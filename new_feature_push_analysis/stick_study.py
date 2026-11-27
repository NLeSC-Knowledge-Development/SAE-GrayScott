"""Find stick directions in a two-regime SAE, then inject stick figures into the main regime.

    ../../../.venv/bin/python stick_study.py --data data/gray_scott_mixed.npz --model-dir model \
                                          --figures figures

The question
------------
The emulator and the SAE are trained on two Gray-Scott regimes at once: the spot-dominated
main regime (``F = 0.026``, ``k = 0.059``) and elongated sticks (``F = 0.030``,
``k = 0.059``). The regime index is an input, so the same network holds two transition rules
and the experiment can choose which one it runs.

Three things are measured.

**1. Does the dictionary have a direction for a stick?** For every cell, the component it
belongs to has a principal-axis elongation (1 for a disc, larger for a stick). Every SAE
direction is scored by how well it ranks cells inside elongated components above cells
inside round ones, on states from one split and then on states from another. A direction
that only knows "occupied versus empty" scores 0.5 on this comparison, because both classes
are occupied.

**2. What does injecting stick figures do in the main regime?** A stick figure is the
selected direction together with the spatial activation footprint the feature has on a real
stick. Three such footprints at distinct sites are added to the emulator's layer on the
first call only, and the conditioned emulator then runs freely in the main regime for twenty
calls. The same footprints are used for three directions -- the stick feature, a
round-component feature, and a random direction of the same size -- and a fourth condition
uses the stick direction on three discs with the same area per site. The comparison
separates *which direction* from *which shape*.

**3. What would the same nudge do to the equations?** Each pair's first-call field difference
is a perturbation of the two fields. It is converted to physical units and added to that
pair's true main-regime state, then integrated with the true equations at ``F = 0.026``,
``k = 0.059`` for the same twenty calls. This is the only part of the experiment that is not
the network talking to itself.

Writes ``<model-dir>/stick.json`` and ``<figures>/stick_features.png``,
``<figures>/stick_injection.png`` and ``<figures>/stick_outcomes.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from generate_data import (ELONGATION_MIN, SPOT_THRESHOLD, advance, connected_components,
                           coverage, describe_field, elongation_map, load_dataset, spot_centres,
                           stick_mask)
from interpret import auroc
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder

# How many states each measurement uses. Selection and evaluation are different splits.
N_DISCOVERY_STATES = 40
N_EVAL_STATES = 40
# The paired injection: how many (state, site) pairs, and how many calls to follow them.
N_PAIRS = 120
N_CALLS = 20
# A component counts as being at the injection site if its middle is within this many cells.
SITE_RADIUS = 6
# How hard the injection is at its strongest cell, as a fraction of the token's own size.
PUSH_FRACTION = 0.25
# The footprint is blurred so the push has no sharp edges.
FOOTPRINT_SIGMA = 1.2
# A feature must fire on at least this fraction of stick cells to be eligible.
MIN_FEATURE_RATE = 1e-4
# A component at or below this elongation is the "round" class in the discriminative score.
ROUND_ELONGATION = 1.6

STICK_COLOUR = "#c44e52"
BLOB_COLOUR = "#4c72b0"
RANDOM_COLOUR = "#999999"
MODEL_COLOUR = "#dd8452"


# ------------------------------------------------------------------- labels and scores


def states_for_regime(block, regime, count, seed=0):
    """``count`` states of one regime, spread through the block in time."""
    pool = np.flatnonzero(block["regime"] == regime)
    if len(pool) == 0:
        raise ValueError(f"the split holds no states for regime {regime}")
    count = min(count, len(pool))
    positions = np.linspace(0, len(pool) - 1, count).round().astype(int)
    return block["inputs"][pool[positions]], block["regime"][pool[positions]]


def round_mask(field, threshold=SPOT_THRESHOLD, max_elongation=ROUND_ELONGATION):
    """Cells belonging to a connected component that is compact, not elongated."""
    out = np.zeros_like(field, dtype=bool)
    for component in connected_components(field, threshold):
        if component["elongation"] <= max_elongation:
            rows, cols = component["cells"].T
            out[rows, cols] = True
    return out


def cell_labels(states, mean, std):
    """Per-cell elongation, the stick class and the round class for every state."""
    v = states[:, -1, 1] * std[1] + mean[1]
    elongation = np.stack([elongation_map(field) for field in v])
    sticks = np.stack([stick_mask(field) for field in v])
    rounds = np.stack([round_mask(field) for field in v])
    return v, elongation, sticks, rounds


def feature_matrix(model, sae, layer, states, regime):
    """The SAE's activations for every cell of every state: one row per cell."""
    tokens = token_activations(model, states, layer, regime)
    with torch.no_grad():
        return sae.encode(tokens).numpy()


def per_feature_auroc(features, labels):
    """AUROC of every feature as a score for the binary label."""
    return np.array([auroc(features[:, index], labels)
                     for index in range(features.shape[1])])


def select_features(model, sae, layer, discovery, evaluation, mean, std,
                    stick_regime, blob_regime):
    """Score every direction on stick-versus-round, choose two, and score them out of sample."""
    discovery_states, discovery_regime = discovery
    evaluation_states, evaluation_regime = evaluation
    _, elongation, sticks, rounds = cell_labels(discovery_states, mean, std)
    _, eval_elongation, eval_sticks, eval_rounds = cell_labels(evaluation_states, mean, std)
    features = feature_matrix(model, sae, layer, discovery_states, discovery_regime)
    eval_features = feature_matrix(model, sae, layer, evaluation_states, evaluation_regime)

    occupied = (sticks | rounds).reshape(-1)
    if occupied.sum() < 100 or not 0 < sticks.reshape(-1)[occupied].sum() < occupied.sum():
        raise ValueError("too few occupied cells, or only one class, to compare stick and "
                         "round components")
    stick_target = sticks.reshape(-1)[occupied]
    feature_occupied = features[occupied]
    eval_occupied = (eval_sticks | eval_rounds).reshape(-1)
    if eval_occupied.sum() < 100 or not 0 < eval_sticks.reshape(-1)[eval_occupied].sum() \
            < eval_occupied.sum():
        raise ValueError("the evaluation states do not hold both shape classes")
    eval_target = eval_sticks.reshape(-1)[eval_occupied]
    eval_feature_occupied = eval_features[eval_occupied]
    eval_elongation_occupied = eval_elongation.reshape(-1)[eval_occupied]

    stick_vs_blob = per_feature_auroc(feature_occupied, stick_target)
    blob_vs_stick = 1.0 - stick_vs_blob
    # Direction that carries the stick class rather than the round one, and the reverse.
    stick_rate = features[sticks.reshape(-1)].mean(axis=0)
    blob_rate = features[rounds.reshape(-1)].mean(axis=0)
    eligible = (stick_rate > MIN_FEATURE_RATE) & (blob_rate > MIN_FEATURE_RATE)
    if not eligible.any():
        raise ValueError("no feature fires on both classes; lower MIN_FEATURE_RATE")
    stick_choice = int(np.argmax(np.where(eligible, stick_vs_blob, -np.inf)))
    blob_choice = int(np.argmax(np.where(eligible, blob_vs_stick, -np.inf)))

    scores = {
        "stick_feature": stick_choice,
        "blob_feature": blob_choice,
        "stick_vs_blob_discovery": stick_vs_blob.tolist(),
        "blob_vs_stick_discovery": blob_vs_stick.tolist(),
        "selected": {
            "stick": {
                "feature": stick_choice,
                "discovery_auroc_stick_vs_blob": float(stick_vs_blob[stick_choice]),
                "evaluation_auroc_stick_vs_blob": float(auroc(
                    eval_feature_occupied[:, stick_choice], eval_target)),
                "evaluation_correlation_with_elongation": float(np.corrcoef(
                    eval_feature_occupied[:, stick_choice],
                    eval_elongation_occupied)[0, 1]),
                "evaluation_auroc_stick_anywhere": float(auroc(
                    eval_features[:, stick_choice].reshape(-1), eval_sticks.reshape(-1))),
                "mean_activation_on_stick_cells": float(stick_rate[stick_choice]),
                "mean_activation_on_round_cells": float(blob_rate[stick_choice]),
            },
            "blob": {
                "feature": blob_choice,
                "discovery_auroc_blob_vs_stick": float(blob_vs_stick[blob_choice]),
                "evaluation_auroc_blob_vs_stick": float(auroc(
                    eval_feature_occupied[:, blob_choice], 1 - eval_target)),
                "evaluation_auroc_round_anywhere": float(auroc(
                    eval_features[:, blob_choice].reshape(-1), eval_rounds.reshape(-1))),
                "mean_activation_on_stick_cells": float(stick_rate[blob_choice]),
                "mean_activation_on_round_cells": float(blob_rate[blob_choice]),
            },
        },
        "class_sizes": {
            "discovery_stick_cells": int(sticks.sum()),
            "discovery_round_cells": int(rounds.sum()),
            "evaluation_stick_cells": int(eval_sticks.sum()),
            "evaluation_round_cells": int(eval_rounds.sum()),
        },
    }
    return scores


# ---------------------------------------------------------------------------- injection


def periodic_distance(n, a, b):
    """Distance on a periodic axis."""
    d = abs(a - b) % n
    return min(d, n - d)


def site_distance(n, row, col):
    """Distance from every cell to a site, on the periodic grid."""
    rows, cols = np.mgrid[0:n, 0:n]
    dy = np.abs(rows - row) % n
    dy = np.minimum(dy, n - dy)
    dx = np.abs(cols - col) % n
    dx = np.minimum(dx, n - dx)
    return np.hypot(dy, dx)


def disc_mask(n, row, col, area):
    """A disc at the site with about the same number of cells as the stick footprint."""
    radius = np.sqrt(area / np.pi)
    return site_distance(n, row, col) <= radius


def choose_sites(rng, n, count):
    """``count`` distinct cells, as (row, col) pairs."""
    flat = rng.choice(n * n, size=min(count, n * n), replace=False)
    return [(int(index // n), int(index % n)) for index in flat]


def periodic_blur(field, sigma):
    """Gaussian blur on the periodic grid, without a scipy dependency."""
    radius = int(3 * sigma)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-offsets ** 2 / (2 * sigma ** 2))
    weights = weights / weights.sum()
    rows = np.zeros_like(field)
    for shift, weight in zip(offsets, weights):
        rows += weight * np.roll(field, shift, axis=0)
    blurred = np.zeros_like(field)
    for shift, weight in zip(offsets, weights):
        blurred += weight * np.roll(rows, shift, axis=1)
    return blurred


def stick_figure(model, sae, layer, stick_states, stick_regime, feature, target_row,
                 target_col, rng, mean, std):
    """A real stick's footprint and the selected feature's activation on it, moved to a site.

    Returns the amplitude map, the binary footprint and the elongation of the component used.
    The amplitude is the feature's activation multiplied by the component mask, scaled so its
    largest value is 1, then blurred so the push has no sharp edges; if the feature is silent
    there, the binary footprint is used.
    """
    order = rng.permutation(len(stick_states))
    for index in order:
        state = stick_states[index:index + 1]
        # The stored fields are standardised; the component threshold is in physical units.
        v = state[0, -1, 1] * std[1] + mean[1]
        components = [component for component in connected_components(v)
                      if component["elongation"] >= ELONGATION_MIN]
        if not components:
            continue
        component = components[int(rng.integers(len(components)))]
        mask = np.zeros(v.shape, dtype=bool)
        rows, cols = component["cells"].T
        mask[rows, cols] = True
        tokens = token_activations(model, state, layer, np.array([stick_regime]))
        with torch.no_grad():
            activation = sae.encode(tokens).reshape(*v.shape, -1)[:, :, feature].numpy()
        amplitude = np.clip(activation * mask, 0.0, None)
        if amplitude.max() <= 0.0:
            amplitude = mask.astype(np.float64)
        amplitude = periodic_blur(amplitude, FOOTPRINT_SIGMA)
        amplitude = amplitude / amplitude.max()
        centre_row, centre_col = component["centroid"]
        shift_row = int(round(target_row - centre_row))
        shift_col = int(round(target_col - centre_col))
        amplitude = np.roll(np.roll(amplitude, shift_row, axis=0), shift_col, axis=1)
        mask = np.roll(np.roll(mask, shift_row, axis=0), shift_col, axis=1)
        return amplitude, mask, float(component["elongation"])
    raise ValueError("no elongated component in any stick state; lower ELONGATION_MIN")


def patched_forward(model, inputs, layer, patch, regime):
    """Run the model with ``patch`` applied to one layer's activations."""
    handle = model.get_submodule(layer).register_forward_hook(
        lambda module, args, output: patch(output))
    try:
        with torch.no_grad():
            return model(inputs, regime)
    finally:
        handle.remove()


def patched_rollout(model, layer, state, regime, amplitude, direction, n_calls):
    """A push on the first call only, then the model runs on its own predictions.

    Returned are the pushed fields, the untouched fields, and the difference the push made on
    the first call -- the perturbation the equations can be given.
    """
    window = torch.as_tensor(state[None], dtype=torch.float32)
    regime_tensor = torch.as_tensor([regime], dtype=torch.long)
    amplitude_tensor = torch.as_tensor(amplitude, dtype=torch.float32)
    direction_tensor = torch.as_tensor(direction, dtype=torch.float32)

    def patch(output):
        return output + (amplitude_tensor[None, None, :, :]
                         * direction_tensor[None, :, None, None])

    pushed_state = patched_forward(model, window, layer, patch, regime_tensor)
    with torch.no_grad():
        plain_state = model(window, regime_tensor)
    effective = (pushed_state - plain_state).numpy()[0]
    pushed_window = torch.cat([window[:, 1:], pushed_state[:, None]], dim=1)
    plain_window = torch.cat([window[:, 1:], plain_state[:, None]], dim=1)
    pushed_fields = [pushed_state.numpy()[0]]
    plain_fields = [plain_state.numpy()[0]]
    for _ in range(n_calls - 1):
        with torch.no_grad():
            pushed_state = model(pushed_window, regime_tensor)
            plain_state = model(plain_window, regime_tensor)
        pushed_window = torch.cat([pushed_window[:, 1:], pushed_state[:, None]], dim=1)
        plain_window = torch.cat([plain_window[:, 1:], plain_state[:, None]], dim=1)
        pushed_fields.append(pushed_state.numpy()[0])
        plain_fields.append(plain_state.numpy()[0])
    return np.stack(pushed_fields), np.stack(plain_fields), effective


def sae_pushed_rollout(model, layer, sae, feature, state, regime, strength, amplitude, n_calls,
                       push_calls=None, field_delta=None):
    """A push through the SAE, then the model on its own predictions.

    At every call the layer output is encoded into SAE latents, the selected feature's latent
    is increased by ``strength`` times ``amplitude`` at each cell, and the modified latents are
    decoded back into the token. ``strength = 0`` is the paired control: the same encode/decode
    path without the push. ``push_calls`` stops the push after that many calls and keeps running
    the encode/decode path with no added activation. ``field_delta`` is added to the ``v`` field
    of the current state at every call, for a non-SAE control.
    """
    window = torch.as_tensor(state[None], dtype=torch.float32)
    regime_tensor = torch.as_tensor([regime], dtype=torch.long)
    amplitude_tensor = torch.as_tensor(amplitude.reshape(-1), dtype=torch.float32)
    one_hot = torch.zeros(sae.n_latent)
    one_hot[feature] = 1.0
    delta_tensor = None if field_delta is None else torch.as_tensor(
        field_delta, dtype=torch.float32)

    def patch(output):
        shape = output.shape
        tokens = output.permute(0, 2, 3, 1).reshape(-1, shape[1])
        latents = sae.encode(tokens) + active * amplitude_tensor[:, None] * one_hot[None, :]
        rebuilt = sae.decoder(latents)
        return rebuilt.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)

    fields = []
    for call in range(n_calls):
        active = strength if push_calls is None or call < push_calls else 0.0
        if delta_tensor is not None:
            window = window.clone()
            window[:, -1, 1] += delta_tensor
        state_t = patched_forward(model, window, layer, patch, regime_tensor)
        window = torch.cat([window[:, 1:], state_t[:, None]], dim=1)
        fields.append(state_t.numpy()[0])
    return np.stack(fields)


def compact_near(field, row, col, radius=SITE_RADIUS, max_elongation=ROUND_ELONGATION):
    """Is there a compact component whose middle is within ``radius`` of this cell?"""
    n = field.shape[0]
    for component in connected_components(field):
        if component["elongation"] > max_elongation:
            continue
        centre_row, centre_col = component["centroid"]
        if np.hypot(periodic_distance(n, centre_row, row),
                    periodic_distance(n, centre_col, col)) <= radius:
            return True
    return False


def elongation_near(field, row, col, radius=SITE_RADIUS):
    """The largest elongation among components whose middle is within ``radius``."""
    n = field.shape[0]
    best = 0.0
    for component in connected_components(field):
        centre_row, centre_col = component["centroid"]
        if np.hypot(periodic_distance(n, centre_row, row),
                    periodic_distance(n, centre_col, col)) <= radius:
            best = max(best, component["elongation"])
    return best


def component_near(components, n, row, col, radius=SITE_RADIUS):
    """The most elongated component whose middle is within ``radius`` of a site."""
    best = None
    for component in components:
        centre_row, centre_col = component["centroid"]
        if np.hypot(periodic_distance(n, centre_row, row),
                    periodic_distance(n, centre_col, col)) <= radius:
            if best is None or component["elongation"] > best["elongation"]:
                best = component
    return best


def shape_delta(pushed, control, n, sites, radius=SITE_RADIUS):
    """Largest shape change near the injection sites, as a continuous endpoint."""
    best = {"delta_elongation": 0.0, "delta_area": 0.0, "delta_skeleton": 0.0,
            "pushed_elongation": 0.0, "control_elongation": 0.0}
    for row, col in sites:
        a = component_near(pushed, n, row, col, radius)
        b = component_near(control, n, row, col, radius)
        a_e = a["elongation"] if a else 0.0
        b_e = b["elongation"] if b else 0.0
        candidate = {
            "delta_elongation": a_e - b_e,
            "delta_area": float((a["area"] if a else 0) - (b["area"] if b else 0)),
            "delta_skeleton": float((a["skeleton_length"] if a else 0.0)
                                    - (b["skeleton_length"] if b else 0.0)),
            "pushed_elongation": a_e,
            "control_elongation": b_e,
        }
        if candidate["delta_elongation"] > best["delta_elongation"]:
            best = candidate
    return best


def component_overlap(components, mask):
    """The component with the largest overlap with a reference mask, or None."""
    best, best_overlap = None, 0
    for component in components:
        rows, cols = component["cells"].T
        overlap = int(mask[rows, cols].sum())
        if overlap > best_overlap:
            best, best_overlap = component, overlap
    return best


def overlap_delta(pushed, control, mask):
    """Shape change of the *same* component, identified by overlap with its old mask."""
    a = component_overlap(pushed, mask)
    b = component_overlap(control, mask)
    return {
        "delta_elongation": float((a["elongation"] if a else 0.0)
                                  - (b["elongation"] if b else 0.0)),
        "delta_area": float((a["area"] if a else 0) - (b["area"] if b else 0)),
        "delta_skeleton": float((a["skeleton_length"] if a else 0.0)
                                - (b["skeleton_length"] if b else 0.0)),
    }


def stamp_stick(v, row, col, amplitude, length=12, width=3):
    """Add a horizontal stick of ``v`` at a periodic site, as a physical positive control."""
    n = v.shape[0]
    rows = (np.arange(n) - row + n // 2) % n - n // 2
    cols = (np.arange(n) - col + n // 2) % n - n // 2
    mask = (np.abs(rows)[:, None] <= width // 2) & (np.abs(cols)[None, :] <= length // 2)
    return v + amplitude * mask


def global_shape_metrics(field):
    """Global stick content: maximum E, stick area and stick-cell fraction."""
    components = describe_field(field)
    n = field.shape[0]
    elongations = [component["elongation"] for component in components]
    stick_area = sum(component["area"] for component in components
                     if component["elongation"] >= ELONGATION_MIN)
    return {
        "max_elongation": float(max(elongations)) if elongations else 0.0,
        "stick_area": float(stick_area),
        "stick_fraction": float(stick_area / (n * n)),
    }


def first_call_effect(model, layer, state, regime, amplitude, direction):
    """The field change produced by the patch on the first call only."""
    window = torch.as_tensor(state[None], dtype=torch.float32)
    regime_tensor = torch.as_tensor([regime], dtype=torch.long)
    amplitude_tensor = torch.as_tensor(amplitude, dtype=torch.float32)
    direction_tensor = torch.as_tensor(direction, dtype=torch.float32)

    def patch(output):
        return output + (amplitude_tensor[None, None, :, :]
                         * direction_tensor[None, :, None, None])

    pushed = patched_forward(model, window, layer, patch, regime_tensor)
    with torch.no_grad():
        plain = model(window, regime_tensor)
    return (pushed - plain).numpy()[0]


def bootstrap_interval(values, seed=0, draws=1000):
    """Mean and 95% percentile interval for a paired endpoint."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean())
             for _ in range(draws)]
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def component_amplitude(model, sae, layer, state, component, feature, regime, n):
    """The selected feature's activation map on one existing component, normalised."""
    tokens = token_activations(model, state[None], layer, np.array([regime]))
    with torch.no_grad():
        activation = sae.encode(tokens).reshape(n, n, -1)[:, :, feature].numpy()
    mask = np.zeros((n, n), dtype=bool)
    rows, cols = component["cells"].T
    mask[rows, cols] = True
    amplitude = np.clip(activation * mask, 0.0, None)
    if amplitude.max() <= 0.0:
        amplitude = mask.astype(np.float64)
    return amplitude / amplitude.max(), mask


def load_component_selection(model_dir):
    """Read the component-level selection, if component_study.py has produced it."""
    path = Path(model_dir) / "stick_components.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def injected_conditions(sae, stick_feature, blob_feature, mask, area_per_site, n, sites,
                        generator):
    """The four injections compared on the same footprints and the same sites."""
    stick_direction = sae.directions()[stick_feature].numpy()
    blob_direction = sae.directions()[blob_feature].numpy()
    random_direction = F.normalize(torch.randn(sae.n_input, generator=generator), dim=0).numpy()
    disc = np.zeros((n, n), dtype=np.float64)
    for row, col in sites:
        disc = np.maximum(disc, disc_mask(n, row, col, area_per_site).astype(np.float64))
    return {
        "stick direction, stick shape": (stick_direction, mask),
        "blob direction, stick shape": (blob_direction, mask),
        "random direction, stick shape": (random_direction, mask),
        "stick direction, disc shape": (stick_direction, disc),
    }


# ------------------------------------------------------------------------------- figures


def features_figure(scores, out_path, dpi=150):
    """Every direction on the stick-versus-round comparison, with the two choices marked."""
    stick = np.asarray(scores["stick_vs_blob_discovery"])
    blob = np.asarray(scores["blob_vs_stick_discovery"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))

    axes[0].scatter(blob, stick, s=9, color="#bbbbbb", alpha=0.7,
                    label="every direction")
    axes[0].scatter([blob[scores["stick_feature"]]], [stick[scores["stick_feature"]]],
                    s=42, color=STICK_COLOUR, zorder=3,
                    label=f"stick feature {scores['stick_feature']}")
    axes[0].scatter([blob[scores["blob_feature"]]], [stick[scores["blob_feature"]]],
                    s=42, color=BLOB_COLOUR, zorder=3,
                    label=f"round feature {scores['blob_feature']}")
    axes[0].plot([0.5, 1.0], [1.0, 0.5], color="#888888", lw=1.0, linestyle=":")
    axes[0].set_xlabel("AUROC for round cells against stick cells")
    axes[0].set_ylabel("AUROC for stick cells against round cells")
    axes[0].set_title("which directions separate shape, not occupancy", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    names = ["stick feature", "round feature"]
    evaluation = [scores["selected"]["stick"]["evaluation_auroc_stick_vs_blob"],
                  scores["selected"]["blob"]["evaluation_auroc_blob_vs_stick"]]
    anywhere = [scores["selected"]["stick"]["evaluation_auroc_stick_anywhere"],
                scores["selected"]["blob"]["evaluation_auroc_round_anywhere"]]
    position = np.arange(2)
    width = 0.36
    axes[1].bar(position - width / 2, evaluation, width=width, color=[STICK_COLOUR, BLOB_COLOUR],
                label="its own class against the other shape")
    axes[1].bar(position + width / 2, anywhere, width=width, color="#c9c9c9",
                edgecolor="#8a8a8a", linewidth=0.6,
                label="class against the whole grid")
    axes[1].axhline(0.5, color="#888888", lw=1.0, linestyle=":")
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_ylabel("AUROC on the evaluation states")
    axes[1].set_title("out of sample", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(axis="y", alpha=0.3)

    fig.text(0.5, 0.01,
             "Left: both classes are cells inside a bright component, so 0.5 is the score of a "
             "direction that only knows occupied from empty. Right: the selected directions "
             "on states that took no part in choosing them.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def injection_figure(example, out_path, dpi=150,
                     titles=("without the injection", "with the stick figure"), footer=None):
    """One state: control, pushed, difference, and the footprint used."""
    control, pushed, difference, amplitude = example
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0))
    panels = (
        (control, titles[0], "inferno", 0.0, 0.45, "$v$"),
        (pushed, titles[1], "inferno", 0.0, 0.45, "$v$"),
        (difference, "the difference", "coolwarm", None, None, "$\\Delta v$"),
        (amplitude, "the injected footprint", "viridis", 0.0, 1.0, "amplitude"),
    )
    for axis, (field, title, cmap, low, high, label) in zip(axes, panels):
        kwargs = {} if low is None else {"vmin": low, "vmax": high}
        image = axis.imshow(field, cmap=cmap, **kwargs)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, label=label)
    fig.text(0.5, 0.01, footer or f"the fields after {N_CALLS} calls.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


SHORT_LABELS = {
    "stick direction, stick shape": "stick\nstick",
    "blob direction, stick shape": "round\nstick",
    "non-shape direction, stick shape": "non-shape\nstick",
    "signed round direction, stick shape": "signed\nstick",
    "probe direction, stick shape": "probe\nstick",
    "random direction, stick shape": "random\nstick",
    "stick direction, disc shape": "stick\ndisc",
}


def _short(label):
    return SHORT_LABELS.get(label, label.replace(", ", ",\n"))


def outcome_figure(model_rates, labels, out_path, n_pairs, dpi=150):
    """What followed each injection: binary outcomes, RMS, and the continuous shape change."""
    position = np.arange(len(labels))
    width = 0.26
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 4.8),
                             gridspec_kw={"width_ratios": (1.5, 1.0, 1.0, 1.0)})

    def interval(key):
        low = np.array([model_rates[label][key + "_ci"][0] for label in labels])
        high = np.array([model_rates[label][key + "_ci"][1] for label in labels])
        mean = np.array([model_rates[label][key] for label in labels])
        return mean, np.vstack([mean - low, high - mean])

    mean, _ = interval("new_blob")
    stick_mean, _ = interval("stick_near")
    axes[0].bar(position - width / 2, mean, width=width, color=MODEL_COLOUR,
                label="new round blob")
    axes[0].bar(position + width / 2, stick_mean, width=width, color=STICK_COLOUR,
                label="elongated component")
    axes[0].set_xticks(position)
    axes[0].set_xticklabels([_short(label) for label in labels], fontsize=8)
    axes[0].set_ylabel(f"share of {n_pairs} pairs")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("what followed the injection", fontsize=10)
    axes[0].legend(fontsize=7.6, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    rms, rms_error = interval("change_rms")
    axes[1].bar(position, rms, yerr=rms_error, color=MODEL_COLOUR, capsize=2)
    axes[1].set_xticks(position)
    axes[1].set_xticklabels([_short(label) for label in labels], fontsize=8)
    axes[1].set_ylabel("RMS change (standardised)")
    axes[1].set_title("how far the forecast moved", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    delta, delta_error = interval("delta_elongation")
    axes[2].bar(position, delta, yerr=delta_error, color=STICK_COLOUR, capsize=2)
    axes[2].axhline(0.0, color="#888888", lw=0.9)
    axes[2].set_xticks(position)
    axes[2].set_xticklabels([_short(label) for label in labels], fontsize=8)
    axes[2].set_ylabel("Δ elongation near the site")
    axes[2].set_title("continuous shape change", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    axes[3].bar(position, [model_rates[label]["stick_fraction_change"] for label in labels],
                color=STICK_COLOUR)
    axes[3].axhline(0.0, color="#888888", lw=0.9)
    axes[3].set_xticks(position)
    axes[3].set_xticklabels([_short(label) for label in labels], fontsize=8)
    axes[3].set_ylabel("Δ cells in elongated components")
    axes[3].set_title("added stick content", fontsize=10)
    axes[3].grid(axis="y", alpha=0.3)

    fig.text(0.5, 0.01,
             "Shapes are paired: same site, state and footprint in every condition. Error bars "
             "are 95% pair-bootstrap intervals. Equation results are shown separately.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def twin_figure(twin, twin_direct, out_path, dpi=150):
    """True-equation results on empty sites, with the direct physical stick as a reference."""
    peaks = list(next(iter(twin.values())).keys())
    x = np.arange(len(peaks))
    display = {"non-shape direction, stick shape": "control direction, stick shape",
               "signed round direction, stick shape": "signed blob direction, stick shape"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
    for label, entry in twin.items():
        axes[0].plot(x, [entry[peak]["new_blob_rate"] for peak in peaks], marker="o",
                     alpha=0.7, label=display.get(label, label))
        axes[1].plot(x, [entry[peak]["mean_max_elongation"] for peak in peaks], marker="o",
                     alpha=0.7, label=display.get(label, label))
    axes[0].plot(x, [twin_direct[peak]["new_blob_rate"] for peak in peaks], marker="s",
                 color="black", lw=2.5, label="direct physical stick")
    axes[1].plot(x, [twin_direct[peak]["mean_max_elongation"] for peak in peaks], marker="s",
                 color="black", lw=2.5, label="direct physical stick")
    for axis, title in ((axes[0], "new component on an empty site"),
                        (axes[1], "maximum elongation on an empty site")):
        axis.set_xticks(x)
        axis.set_xticklabels([f"peak |Δv| = {peak}" for peak in peaks])
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("share of blank sites")
    axes[1].set_ylabel("mean maximum elongation")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def shape_outcome_figure(dose, stick_intervention, out_path, dpi=150):
    """Push strength and stick-regime ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
    fractions = sorted(dose, key=float)
    for key, label, colour in (("stick direction", "stick direction", STICK_COLOUR),
                               ("non-shape direction", "control direction", RANDOM_COLOUR)):
        axes[0].plot([float(f) for f in fractions],
                     [dose[f][key]["delta_elongation"] for f in fractions],
                     marker="o", color=colour, label=label)
    axes[0].axhline(0.0, color="#888888", lw=0.9)
    axes[0].set_xlabel("push fraction $\\alpha$")
    axes[0].set_ylabel("Δ elongation near the site")
    axes[0].set_title("push strength in the blob regime", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    labels = list(stick_intervention)
    display = {"ablate non-shape direction": "ablate control direction",
               "boost signed round feature": "boost signed blob feature"}
    position = np.arange(len(labels))
    means = [stick_intervention[label]["delta_max_elongation"] for label in labels]
    low = [stick_intervention[label]["delta_max_elongation_ci"][0] for label in labels]
    high = [stick_intervention[label]["delta_max_elongation_ci"][1] for label in labels]
    axes[1].bar(position, means, yerr=[np.array(means) - np.array(low),
                                       np.array(high) - np.array(means)],
                color=[MODEL_COLOUR if "physical" in label else
                       (STICK_COLOUR if "stick" in label else RANDOM_COLOUR) for label in labels],
                capsize=2)
    axes[1].axhline(0.0, color="#888888", lw=0.9)
    axes[1].set_xticks(position)
    axes[1].set_xticklabels([display.get(label, label).replace(" ", "\n") for label in labels],
                            fontsize=8)
    axes[1].set_ylabel("Δ maximum elongation (global)")
    axes[1].set_title("stick-regime ablation and boost", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------------- main


def parse_floats(text):
    return [float(value) for value in text.split(",") if value.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Find a stick direction, inject it, and compare the equations.")
    parser.add_argument("--data", default="data/gray_scott_mixed.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--blob-regime", type=int, default=0)
    parser.add_argument("--stick-regime", type=int, default=1)
    parser.add_argument("--discovery-split", choices=("val", "test"), default="val")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test")
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--calls", type=int, default=N_CALLS)
    parser.add_argument("--push-fraction", type=float, default=PUSH_FRACTION)
    parser.add_argument("--random-pairs", type=int, default=40)
    parser.add_argument("--random-directions", type=int, default=8)
    parser.add_argument("--stick-pairs", type=int, default=40)
    parser.add_argument("--dose-pairs", type=int, default=24)
    parser.add_argument("--dose-fractions", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--twin-pairs", type=int, default=24)
    parser.add_argument("--twin-peaks", default="0.05,0.15,0.25")
    parser.add_argument("--matched-rms", action="store_true",
                        help="scale every direction to the stick direction's first-call RMS")
    parser.add_argument("--output", default="stick.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    n = settings["N"]
    if not 0 <= args.blob_regime < settings["n_regimes"] \
            or not 0 <= args.stick_regime < settings["n_regimes"]:
        parser.error("both regimes must be names the dataset holds")
    if args.blob_regime == args.stick_regime:
        parser.error("the blob and stick regimes must be different")

    model, _ = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    config = checkpoint["config"]
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    layer = config["layer"]

    # ------------------------------------------------------------ 1. selection
    component_selection = load_component_selection(args.model_dir)
    if component_selection is not None:
        selected = component_selection["selected"]
        stick_feature = int(selected["stick_feature"])
        blob_feature = int(selected["round_feature"])
        signed_feature = int(selected["signed_feature"])
        signed_sign = float(selected.get("signed_sign", 1.0))
        nonshape_feature = int(selected["matched_nonshape"])
        probe_direction = np.asarray(selected["probe_direction"], dtype=np.float64)
        probe_direction /= max(np.linalg.norm(probe_direction), 1e-12)
        random_directions = [np.asarray(value, dtype=np.float64)
                             for value in selected["random_directions"]]
        selection_source = "component"
        scores = {"source": "component",
                  "selected": selected,
                  "primary": component_selection["primary"],
                  "secondary": component_selection["secondary"]}
    else:
        discovery_block = dataset["splits"][args.discovery_split]
        evaluation_block = dataset["splits"][args.evaluation_split]
        discovery_states, discovery_regime = states_for_regime(
            discovery_block, args.stick_regime, N_DISCOVERY_STATES, args.seed)
        evaluation_states, evaluation_regime = states_for_regime(
            evaluation_block, args.stick_regime, N_EVAL_STATES, args.seed + 1)
        scores = select_features(model, sae, layer,
                                 (discovery_states, discovery_regime),
                                 (evaluation_states, evaluation_regime), mean, std,
                                 args.stick_regime, args.blob_regime)
        stick_feature = scores["stick_feature"]
        blob_feature = scores["blob_feature"]
        signed_feature = blob_feature
        signed_sign = -1.0
        nonshape_feature = None
        probe_direction = None
        random_directions = []
        selection_source = "cell"

    stick_direction = sae.directions()[stick_feature].numpy()
    blob_direction = sae.directions()[blob_feature].numpy()
    signed_direction = signed_sign * sae.directions()[signed_feature].numpy()
    nonshape_direction = (sae.directions()[nonshape_feature].numpy()
                          if nonshape_feature is not None else None)
    if not random_directions:
        rng = np.random.default_rng(args.seed)
        direction = rng.normal(size=sae.n_input)
        random_directions = [direction / np.linalg.norm(direction)]
    random_direction = random_directions[0]
    if nonshape_direction is None:
        nonshape_direction = random_direction

    # ------------------------------------------------------------ 2. main-regime injection
    rng = np.random.default_rng(args.seed)
    evaluation_block = dataset["splits"][args.evaluation_split]
    blob_states, _ = states_for_regime(evaluation_block, args.blob_regime, args.pairs,
                                       args.seed + 2)
    stick_examples, _ = states_for_regime(evaluation_block, args.stick_regime,
                                          max(N_DISCOVERY_STATES, args.pairs), args.seed + 3)
    sites_per_state = [choose_sites(rng, n, args.sites) for _ in blob_states]

    labels = ["stick direction, stick shape", "blob direction, stick shape",
              "non-shape direction, stick shape", "signed round direction, stick shape"]
    if probe_direction is not None:
        labels.append("probe direction, stick shape")
    labels += ["random direction, stick shape", "stick direction, disc shape"]

    model_rates = {label: {"new_blob_values": [], "stick_near_values": [],
                           "change_rms": [], "coverage_change": [],
                           "stick_fraction_change": [], "delta_elongation": [],
                           "delta_area": [], "delta_skeleton": []}
                   for label in labels}
    effective_fields = {label: [] for label in labels}
    examples = {}
    amplitudes, scales = [], []

    for index, state in enumerate(blob_states):
        sites = sites_per_state[index]
        amplitude = np.zeros((n, n), dtype=np.float64)
        areas = []
        for row, col in sites:
            site_amplitude, footprint, _ = stick_figure(
                model, sae, layer, stick_examples, args.stick_regime, stick_feature,
                row, col, rng, mean, std)
            amplitude = np.maximum(amplitude, site_amplitude)
            areas.append(int(footprint.sum()))
        amplitudes.append(amplitude)
        area_per_site = float(np.mean(areas))
        scale = float(token_activations(model, state[None], layer,
                                        np.array([args.blob_regime])).norm(dim=1).mean())
        scales.append(scale)
        disc = np.zeros((n, n), dtype=np.float64)
        for row, col in sites:
            disc = np.maximum(disc, disc_mask(n, row, col, area_per_site).astype(np.float64))
        conditions = {
            labels[0]: (stick_direction, amplitude),
            labels[1]: (blob_direction, amplitude),
            labels[2]: (nonshape_direction, amplitude),
            labels[3]: (signed_direction, amplitude),
            labels[-1]: (stick_direction, disc),
        }
        if probe_direction is not None:
            conditions["probe direction, stick shape"] = (probe_direction, amplitude)
        conditions["random direction, stick shape"] = (random_direction, amplitude)

        target_rms = None
        if args.matched_rms:
            effect = first_call_effect(model, layer, state, args.blob_regime,
                                       args.push_fraction * scale * amplitude, stick_direction)
            target_rms = float(np.sqrt(np.mean(effect ** 2)))
        for label in labels:
            direction, shape = conditions[label]
            push = args.push_fraction * scale * shape
            if target_rms is not None:
                effect = first_call_effect(model, layer, state, args.blob_regime, push, direction)
                current = float(np.sqrt(np.mean(effect ** 2)))
                push = push * (target_rms / max(current, 1e-12))
            pushed, control, difference = patched_rollout(
                model, layer, state, args.blob_regime, push, direction, args.calls)
            pushed_v = pushed[-1, 1] * std[1] + mean[1]
            control_v = control[-1, 1] * std[1] + mean[1]
            pushed_components = describe_field(pushed_v)
            control_components = describe_field(control_v)
            delta = shape_delta(pushed_components, control_components, n, sites)
            appeared = any(compact_near(pushed_v, row, col)
                           and not compact_near(control_v, row, col) for row, col in sites)
            pushed_elongation = max(elongation_near(pushed_v, row, col) for row, col in sites)
            control_elongation = max(elongation_near(control_v, row, col) for row, col in sites)
            model_rates[label]["new_blob_values"].append(int(appeared))
            model_rates[label]["stick_near_values"].append(int(
                pushed_elongation >= ELONGATION_MIN and control_elongation < ELONGATION_MIN))
            model_rates[label]["change_rms"].append(float(
                np.sqrt(np.mean((pushed[-1] - control[-1]) ** 2))))
            model_rates[label]["coverage_change"].append(coverage(pushed_v) - coverage(control_v))
            model_rates[label]["stick_fraction_change"].append(
                (float(stick_mask(pushed_v).sum()) - float(stick_mask(control_v).sum()))/ (n * n))
            model_rates[label]["delta_elongation"].append(delta["delta_elongation"])
            model_rates[label]["delta_area"].append(delta["delta_area"])
            model_rates[label]["delta_skeleton"].append(delta["delta_skeleton"])
            effective_fields[label].append(difference)
            if index == 0:
                examples[label] = (control_v, pushed_v,
                                   (pushed[-1, 1] - control[-1, 1]) * std[1], shape)

    for label in labels:
        for key in ("change_rms", "coverage_change", "stick_fraction_change",
                    "delta_elongation", "delta_area", "delta_skeleton"):
            average, low, high = bootstrap_interval(model_rates[label][key], args.seed)
            model_rates[label][key] = average
            model_rates[label][key + "_ci"] = [low, high]
        for key in ("new_blob_values", "stick_near_values"):
            average, low, high = bootstrap_interval(model_rates[label][key], args.seed)
            model_rates[label][key.replace("_values", "")] = average
            model_rates[label][key.replace("_values", "") + "_ci"] = [low, high]

    # ------------------------------------------------------------ 3. random-direction null
    random_null = []
    random_subset = blob_states[:min(args.random_pairs, len(blob_states))]
    for direction in random_directions[:args.random_directions]:
        hits, sticks, rms, elongations = 0, 0, [], []
        for index, state in enumerate(random_subset):
            sites = sites_per_state[index]
            push = args.push_fraction * scales[index] * amplitudes[index]
            pushed, control, _ = patched_rollout(model, layer, state, args.blob_regime,
                                                 push, direction, args.calls)
            pushed_v = pushed[-1, 1] * std[1] + mean[1]
            control_v = control[-1, 1] * std[1] + mean[1]
            delta = shape_delta(describe_field(pushed_v), describe_field(control_v), n, sites)
            hits += int(any(compact_near(pushed_v, row, col)
                            and not compact_near(control_v, row, col) for row, col in sites))
            pushed_elongation = max(elongation_near(pushed_v, row, col) for row, col in sites)
            control_elongation = max(elongation_near(control_v, row, col) for row, col in sites)
            sticks += int(pushed_elongation >= ELONGATION_MIN and control_elongation < ELONGATION_MIN)
            rms.append(float(np.sqrt(np.mean((pushed[-1] - control[-1]) ** 2))))
            elongations.append(delta["delta_elongation"])
        random_null.append({"new_blob": hits / len(random_subset),
                           "stick_near": sticks / len(random_subset),
                           "change_rms": float(np.mean(rms)),
                           "delta_elongation": float(np.mean(elongations))})

    # ------------------------------------------------------------ 4. dose-response
    dose = {}
    dose_subset = list(range(min(args.dose_pairs, len(blob_states))))
    for fraction in parse_floats(args.dose_fractions):
        dose[str(fraction)] = {}
        for name, direction in (("stick direction", stick_direction),
                                ("non-shape direction", nonshape_direction)):
            hits, sticks, rms, elongations = 0, 0, [], []
            for index in dose_subset:
                state, sites = blob_states[index], sites_per_state[index]
                push = fraction * args.push_fraction * scales[index] * amplitudes[index]
                pushed, control, _ = patched_rollout(model, layer, state, args.blob_regime,
                                                     push, direction, args.calls)
                pushed_v = pushed[-1, 1] * std[1] + mean[1]
                control_v = control[-1, 1] * std[1] + mean[1]
                delta = shape_delta(describe_field(pushed_v), describe_field(control_v), n, sites)
                hits += int(any(compact_near(pushed_v, row, col)
                                and not compact_near(control_v, row, col) for row, col in sites))
                pushed_elongation = max(elongation_near(pushed_v, row, col) for row, col in sites)
                control_elongation = max(elongation_near(control_v, row, col) for row, col in sites)
                sticks += int(pushed_elongation >= ELONGATION_MIN and control_elongation < ELONGATION_MIN)
                rms.append(float(np.sqrt(np.mean((pushed[-1] - control[-1]) ** 2))))
                elongations.append(delta["delta_elongation"])
            dose[str(fraction)][name] = {
                "new_blob": hits / len(dose_subset),
                "stick_near": sticks / len(dose_subset),
                "change_rms": float(np.mean(rms)),
                "delta_elongation": float(np.mean(elongations)),
            }

    # ------------------------------------------------------------ 5. stick-regime ablation
    stick_states, _ = states_for_regime(evaluation_block, args.stick_regime,
                                        args.stick_pairs, args.seed + 4)
    stick_conditions = {
        "ablate stick feature": -stick_direction,
        "boost stick feature": stick_direction,
        "ablate non-shape direction": -nonshape_direction,
        "boost signed round feature": signed_direction,
    }
    stick_labels = list(stick_conditions) + ["physical erase component"]
    stick_intervention = {label: {"new_blob_values": [], "stick_near_values": [],
                                 "delta_elongation": [], "delta_area": [],
                                 "delta_skeleton": [], "delta_max_elongation": [],
                                 "delta_stick_area": [], "delta_stick_fraction": []}
                          for label in stick_labels}
    used_stick_states = 0
    for state in stick_states:
        v = state[-1, 1] * std[1] + mean[1]
        components = describe_field(v)
        sticks = [component for component in components
                  if component["elongation"] >= ELONGATION_MIN
                  and component["solidity"] >= 0.70]
        if not sticks:
            continue
        component = max(sticks, key=lambda item: item["elongation"])
        amplitude, original_mask = component_amplitude(model, sae, layer, state, component,
                                                       stick_feature, args.stick_regime, n)
        scale = float(token_activations(model, state[None], layer,
                                        np.array([args.stick_regime])).norm(dim=1).mean())
        row, col = component["centroid"]
        used_stick_states += 1
        baseline_control_v = None
        for label, direction in stick_conditions.items():
            pushed, control, _ = patched_rollout(
                model, layer, state, args.stick_regime,
                args.push_fraction * scale * amplitude, direction, args.calls)
            pushed_v = pushed[-1, 1] * std[1] + mean[1]
            control_v = control[-1, 1] * std[1] + mean[1]
            if baseline_control_v is None:
                baseline_control_v = control_v
            pushed_components = describe_field(pushed_v)
            control_components = describe_field(control_v)
            pushed_global = global_shape_metrics(pushed_v)
            control_global = global_shape_metrics(control_v)
            delta = overlap_delta(pushed_components, control_components, original_mask)
            appeared = compact_near(pushed_v, row, col) and not compact_near(control_v, row, col)
            pushed_elongation = elongation_near(pushed_v, row, col)
            control_elongation = elongation_near(control_v, row, col)
            stick_intervention[label]["new_blob_values"].append(int(appeared))
            stick_intervention[label]["stick_near_values"].append(int(
                pushed_elongation >= ELONGATION_MIN and control_elongation < ELONGATION_MIN))
            stick_intervention[label]["delta_elongation"].append(delta["delta_elongation"])
            stick_intervention[label]["delta_area"].append(delta["delta_area"])
            stick_intervention[label]["delta_skeleton"].append(delta["delta_skeleton"])
            stick_intervention[label]["delta_max_elongation"].append(
                pushed_global["max_elongation"] - control_global["max_elongation"])
            stick_intervention[label]["delta_stick_area"].append(
                pushed_global["stick_area"] - control_global["stick_area"])
            stick_intervention[label]["delta_stick_fraction"].append(
                pushed_global["stick_fraction"] - control_global["stick_fraction"])

        # Physical positive control: erase the component itself, then roll out.
        erased = np.array(state, copy=True)
        erased[-1, 0][original_mask] = (1.0 - mean[0]) / std[0]
        erased[-1, 1][original_mask] = (0.0 - mean[1]) / std[1]
        erased_pushed, _, _ = patched_rollout(
            model, layer, erased, args.stick_regime,
            np.zeros_like(amplitude), stick_direction, args.calls)
        erased_v = erased_pushed[-1, 1] * std[1] + mean[1]
        erased_components = describe_field(erased_v)
        control_components = describe_field(baseline_control_v)
        erased_global = global_shape_metrics(erased_v)
        control_global = global_shape_metrics(baseline_control_v)
        delta = overlap_delta(erased_components, control_components, original_mask)
        label = "physical erase component"
        appeared = (compact_near(erased_v, row, col)
                    and not compact_near(baseline_control_v, row, col))
        pushed_elongation = elongation_near(erased_v, row, col)
        control_elongation = elongation_near(baseline_control_v, row, col)
        stick_intervention[label]["new_blob_values"].append(int(appeared))
        stick_intervention[label]["stick_near_values"].append(int(
            pushed_elongation >= ELONGATION_MIN and control_elongation < ELONGATION_MIN))
        stick_intervention[label]["delta_elongation"].append(delta["delta_elongation"])
        stick_intervention[label]["delta_area"].append(delta["delta_area"])
        stick_intervention[label]["delta_skeleton"].append(delta["delta_skeleton"])
        stick_intervention[label]["delta_max_elongation"].append(
            erased_global["max_elongation"] - control_global["max_elongation"])
        stick_intervention[label]["delta_stick_area"].append(
            erased_global["stick_area"] - control_global["stick_area"])
        stick_intervention[label]["delta_stick_fraction"].append(
            erased_global["stick_fraction"] - control_global["stick_fraction"])
    for label in stick_labels:
        for key in ("delta_elongation", "delta_area", "delta_skeleton",
                    "delta_max_elongation", "delta_stick_area", "delta_stick_fraction"):
            average, low, high = bootstrap_interval(stick_intervention[label][key], args.seed)
            stick_intervention[label][key] = average
            stick_intervention[label][key + "_ci"] = [low, high]
        for key in ("new_blob_values", "stick_near_values"):
            average, low, high = bootstrap_interval(stick_intervention[label][key], args.seed)
            stick_intervention[label][key.replace("_values", "")] = average
            stick_intervention[label][key.replace("_values", "") + "_ci"] = [low, high]

    # ------------------------------------------------------------ 6. equation twin
    stride = settings["stride"]
    F = settings["regime_F"][args.blob_regime]
    k = settings["regime_k"][args.blob_regime]
    twin_requested = min(args.twin_pairs, len(blob_states))
    physical_all = blob_states[:, -1] * std[None, :, None, None] \
        + mean[None, :, None, None]
    # Keep only sites with no compact component before the nudge, so a new component is
    # observable. This is the only place where the equations are asked a clean question.
    twin_indices = [index for index in range(len(blob_states))
                    if not any(compact_near(physical_all[index, 1], row, col)
                               for row, col in sites_per_state[index])]
    twin_indices = twin_indices[:twin_requested]
    if not twin_indices:
        twin_indices = list(range(twin_requested))
    twin_pairs = len(twin_indices)
    truth_rates, truth_elongation, truth_peak, twin = {}, {}, {}, {}
    for label in labels:
        hits, elongations, peaks = 0, [], []
        for index in twin_indices:
            sites = sites_per_state[index]
            profile = effective_fields[label][index] * std[:, None, None]
            u, v = physical_all[index, 0].copy(), physical_all[index, 1].copy()
            u = u + profile[0]
            v = v + profile[1]
            u, v = advance(u, v, args.calls * stride, F, k, settings["dt"])
            hits += int(any(compact_near(v, row, col) for row, col in sites))
            elongations.append(max(elongation_near(v, row, col) for row, col in sites))
            peaks.append([float(np.abs(profile[0]).max()), float(np.abs(profile[1]).max())])
        truth_rates[label] = hits / twin_pairs
        truth_elongation[label] = float(np.mean(elongations))
        truth_peak[label] = [float(np.mean([peak[0] for peak in peaks])),
                             float(np.mean([peak[1] for peak in peaks]))]
        twin[label] = {}
        for peak in parse_floats(args.twin_peaks):
            hits, elongations = 0, []
            for index in twin_indices:
                sites = sites_per_state[index]
                profile = effective_fields[label][index] * std[:, None, None]
                current = max(float(np.abs(profile[1]).max()), 1e-12)
                profile = profile * (peak / current)
                u, v = physical_all[index, 0].copy(), physical_all[index, 1].copy()
                u = u + profile[0]
                v = v + profile[1]
                u, v = advance(u, v, args.calls * stride, F, k, settings["dt"])
                hits += int(any(compact_near(v, row, col) for row, col in sites))
                elongations.append(max(elongation_near(v, row, col) for row, col in sites))
            twin[label][str(peak)] = {"new_blob_rate": hits / twin_pairs,
                                     "mean_max_elongation": float(np.mean(elongations))}

    # Positive control for the twin: a direct 12 x 3 physical stick of v at each site.
    twin_direct = {}
    for peak in parse_floats(args.twin_peaks):
        hits, elongations = 0, []
        for index in twin_indices:
            sites = sites_per_state[index]
            v = physical_all[index, 1].copy()
            for row, col in sites:
                v = stamp_stick(v, row, col, peak)
            _, v = advance(physical_all[index, 0].copy(), v, args.calls * stride,
                           F, k, settings["dt"])
            hits += int(any(compact_near(v, row, col) for row, col in sites))
            elongations.append(max(elongation_near(v, row, col) for row, col in sites))
        twin_direct[str(peak)] = {"new_blob_rate": hits / twin_pairs,
                                 "mean_max_elongation": float(np.mean(elongations))}

    results = {
        "layer": layer,
        "blob_regime": int(args.blob_regime),
        "stick_regime": int(args.stick_regime),
        "regime_names": [str(name) for name in settings["regime_names"]],
        "discovery_split": args.discovery_split,
        "evaluation_split": args.evaluation_split,
        "selection_source": selection_source,
        "n_pairs": int(len(blob_states)),
        "n_sites": int(args.sites),
        "n_calls": int(args.calls),
        "n_twin_pairs": int(twin_pairs),
        "n_stick_pairs": int(used_stick_states),
        "push_fraction": float(args.push_fraction),
        "matched_rms": bool(args.matched_rms),
        "select": scores,
        "model": model_rates,
        "random_null": random_null,
        "dose_response": dose,
        "stick_intervention": stick_intervention,
        "truth": {"new_blob_rate": truth_rates, "mean_max_elongation_near_site": truth_elongation},
        "effective_perturbation_peak": truth_peak,
        "twin": twin,
        "twin_direct": twin_direct,
    }
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / args.output).write_text(json.dumps(results, indent=2))
    suffix = "" if args.output == "stick.json" else "_" + Path(args.output).stem
    if selection_source == "cell":
        features_figure(scores, Path(args.figures) / f"stick_features{suffix}.png")
    injection_figure(examples[labels[0]], Path(args.figures) / f"stick_injection{suffix}.png")
    outcome_figure(model_rates, labels, Path(args.figures) / f"stick_outcomes{suffix}.png",
                   len(blob_states))
    shape_outcome_figure(dose, stick_intervention,
                         Path(args.figures) / f"stick_shape_outcomes{suffix}.png")
    twin_figure(twin, twin_direct, Path(args.figures) / f"stick_twin{suffix}.png")

    # ---------------------------------------------------------------------- what we found
    print(f"\nselection: {selection_source}")
    if component_selection is not None:
        primary = component_selection["primary"]
        print(f"  component direction {stick_feature}: main-regime stick-vs-round AUROC "
              f"{primary['selected_auroc']:.3f} discovery, "
              f"{primary['selected_evaluation_auroc']:.3f} test; "
              f"permutation p {primary['null']['p_positive']:.4f}")
        print(f"  matched non-shape direction {nonshape_feature}; signed direction "
              f"{signed_feature} x {signed_sign:+.0f}")
    else:
        print(f"  stick feature {stick_feature}, round feature {blob_feature}")

    print(f"\ninjection in the {settings['regime_names'][args.blob_regime]} regime, "
          f"{len(blob_states)} pairs, {args.sites} sites, {args.calls} calls")
    print(f"  {'condition':<34s}{'new blob':>10s}{'stick near':>12s}{'RMS change':>12s}"
          f"{'dE':>8s}{'d stick':>10s}")
    for label in labels:
        print(f"  {label:<34s}{model_rates[label]['new_blob']:10.0%}"
              f"{model_rates[label]['stick_near']:12.0%}"
              f"{model_rates[label]['change_rms']:12.4f}"
              f"{model_rates[label]['delta_elongation']:8.3f}"
              f"{model_rates[label]['stick_fraction_change']:10.4f}")
    if random_null:
        print(f"  random directions: new blob {np.mean([r['new_blob'] for r in random_null]):.0%}, "
              f"ΔE {np.mean([r['delta_elongation'] for r in random_null]):+.3f}")
    print("\nstick-regime intervention:")
    for label, entry in stick_intervention.items():
        print(f"  {label:<30s} ΔE {entry['delta_elongation']:+.3f} "
              f"[{entry['delta_elongation_ci'][0]:+.3f}, {entry['delta_elongation_ci'][1]:+.3f}], "
              f"ΔmaxE {entry['delta_max_elongation']:+.3f} "
              f"[{entry['delta_max_elongation_ci'][0]:+.3f}, "
              f"{entry['delta_max_elongation_ci'][1]:+.3f}], "
              f"Δstick {entry['delta_stick_fraction']:+.4f}")
    print(f"\nequation twin on {twin_pairs} blank sites (new blob rate by peak |Δv|):")
    for label in labels:
        values = ", ".join(f"{peak}: {entry['new_blob_rate']:.0%}"
                           for peak, entry in twin[label].items())
        print(f"  {label:<34s}{values}")
    print("  direct physical stick: " + ", ".join(
        f"{peak}: {entry['new_blob_rate']:.0%}" for peak, entry in twin_direct.items()))

    print(f"\nwrote {model_dir / args.output}")
    print(f"wrote {Path(args.figures) / f'stick_injection{suffix}.png'}")
    print(f"wrote {Path(args.figures) / f'stick_outcomes{suffix}.png'}")
    print(f"wrote {Path(args.figures) / f'stick_shape_outcomes{suffix}.png'}")
    print(f"wrote {Path(args.figures) / f'stick_twin{suffix}.png'}")


if __name__ == "__main__":
    main()
