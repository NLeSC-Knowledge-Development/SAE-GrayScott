"""Component-level test for an elongation direction, with proper nulls and probes.

    ../../../.venv/bin/python component_study.py --data data/gray_scott_mixed.npz --model-dir model \
                                              --figures figures

Why this exists
---------------
The cell-level comparison in ``stick_study.py`` treats every cell as an independent sample.
Cells inside one component are not independent, and the stick regime is dominated by a
labyrinth whose elongation is low even though it looks elongated. This script therefore:

1. measures elongation, area, solidity, skeleton length/width and branch points per connected
   component;
2. compares components *within one regime* -- sticks against round spots in the main regime,
   and sticks against labyrinth pieces in the stick regime -- so the regime cannot explain the
   separation;
3. matches the two classes on area and mean ``v``, so occupancy and size cannot explain it
   either;
4. scores each SAE direction on component-mean activations, with a max-over-directions
   permutation null and a component bootstrap;
5. fits a linear probe from the 32-number token to the label, as a positive control, and
   repeats the probe at several layers.

Writes ``<model-dir>/stick_components.json`` and ``<figures>/stick_components.png``.
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

from generate_data import ELONGATION_MIN, describe_field, load_dataset
from interpret import auroc, _ranks
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder

# A component is a stick if it is elongated *and* compact enough to be a stick, not a
# labyrinth. Solidity is area divided by convex-hull area.
STICK_SOLIDITY = 0.70
ROUND_MAX_ELONGATION = 1.6
ROUND_MIN_AREA = 8
ROUND_MAX_AREA = 200
LABYRINTH_SOLIDITY = 0.50
MIN_FIRING_RATE = 1e-4

STICK_COLOUR = "#c44e52"
ROUND_COLOUR = "#4c72b0"


# ------------------------------------------------------------------ component harvesting


def classify(component):
    """Stick, round, labyrinth or other, from the component's shape only."""
    elongation = component["elongation"]
    solidity = component["solidity"]
    if solidity < LABYRINTH_SOLIDITY:
        return "labyrinth"
    if elongation >= ELONGATION_MIN and solidity >= STICK_SOLIDITY:
        return "stick"
    if (elongation <= ROUND_MAX_ELONGATION and solidity >= STICK_SOLIDITY
            and ROUND_MIN_AREA <= component["area"] <= ROUND_MAX_AREA):
        return "round"
    return "other"


def harvest(block, regime, model, sae, layers, feature_layer, mean, std, count=None):
    """Every component of ``count`` states, with mean tokens and mean SAE activations."""
    states = block["inputs"][block["regime"] == regime]
    if count is not None:
        count = min(count, len(states))
        positions = np.linspace(0, len(states) - 1, count).round().astype(int)
        states = states[positions]
    n = states.shape[-1]
    records = []
    firing = np.zeros(sae.n_latent if sae is not None else 0, dtype=np.float64)
    cells = 0
    for state_index, state in enumerate(states):
        v = state[-1, 1] * std[1] + mean[1]
        components = describe_field(v)
        if not components:
            continue
        token_by_layer = {}
        for layer in layers:
            tokens = token_activations(model, state[None], layer,
                                       np.array([regime])).reshape(n, n, -1)
            token_by_layer[layer] = tokens
        features = None
        if sae is not None and feature_layer in token_by_layer:
            flat = token_by_layer[feature_layer].reshape(-1, token_by_layer[feature_layer].shape[-1])
            with torch.no_grad():
                features = sae.encode(flat).reshape(n, n, -1)
            firing += (features > 0).sum(dim=(0, 1)).cpu().numpy()
            cells += n * n
        for component in components:
            rows, cols = component["cells"].T
            record = {
                "regime": int(regime),
                "state": int(state_index),
                "class": classify(component),
                "elongation": float(component["elongation"]),
                "area": int(component["area"]),
                "solidity": float(component["solidity"]),
                "skeleton_length": float(component["skeleton_length"]),
                "skeleton_width": float(component["skeleton_width"]),
                "branch_points": int(component["branch_points"]),
                "endpoints": int(component["endpoints"]),
                "mean_v": float(component["mean_v"]),
                "centroid": [float(component["centroid"][0]), float(component["centroid"][1])],
            }
            record["mean_token"] = {
                layer: token_by_layer[layer][rows, cols].mean(dim=0).cpu().numpy()
                for layer in layers
            }
            if features is not None:
                record["mean_feature"] = features[rows, cols].mean(dim=0).cpu().numpy()
            records.append(record)
    firing_rate = firing / max(cells, 1)
    return records, firing_rate


# ------------------------------------------------------------------ component statistics


def matched_indices(records, positive_class, negative_class, seed=0):
    """Greedy nearest-neighbour match on log area and mean v, within one regime."""
    positive = [i for i, record in enumerate(records) if record["class"] == positive_class]
    negative = [i for i, record in enumerate(records) if record["class"] == negative_class]
    if not positive or not negative:
        return [], []
    values = np.array([[np.log(max(record["area"], 1)), record["mean_v"]]
                       for record in records], dtype=np.float64)
    subset = values[positive + negative]
    centre = subset.mean(axis=0)
    scale = subset.std(axis=0) + 1e-8
    z = (values - centre) / scale
    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    used = set()
    out_positive, out_negative = [], []
    for i in positive:
        best, best_distance = None, np.inf
        for j in negative:
            if j in used:
                continue
            distance = float(np.sum((z[i] - z[j]) ** 2))
            if distance < best_distance:
                best, best_distance = j, distance
        if best is None:
            break
        used.add(best)
        out_positive.append(i)
        out_negative.append(best)
    return out_positive, out_negative


def average_ranks(matrix):
    """1-based average ranks, one column per direction, computed once for permutations."""
    ranks = np.empty_like(matrix, dtype=np.float64)
    for column in range(matrix.shape[1]):
        ranks[:, column] = _ranks(matrix[:, column]) + 1.0
    return ranks


def auroc_from_ranks(ranks, labels):
    """AUROC of every column from precomputed ranks."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(ranks.shape[1], np.nan)
    return (ranks[labels].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def permutation_null(ranks, labels, n_permutations, seed=0):
    """Max-over-directions null for positive and signed separation."""
    labels = np.asarray(labels, dtype=bool)
    observed = auroc_from_ranks(ranks, labels)
    observed_positive = float(np.nanmax(observed))
    observed_signed = float(np.nanmax(np.maximum(observed, 1.0 - observed)))
    rng = np.random.default_rng(seed)
    null_positive, null_signed = [], []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        scores = auroc_from_ranks(ranks, shuffled)
        null_positive.append(float(np.nanmax(scores)))
        null_signed.append(float(np.nanmax(np.maximum(scores, 1.0 - scores))))
    null_positive = np.asarray(null_positive)
    null_signed = np.asarray(null_signed)
    return {
        "observed_positive": observed_positive,
        "observed_signed": observed_signed,
        "p_positive": float((1 + np.sum(null_positive >= observed_positive))
                            / (1 + n_permutations)),
        "p_signed": float((1 + np.sum(null_signed >= observed_signed))
                          / (1 + n_permutations)),
        "null_positive_95": float(np.quantile(null_positive, 0.95)),
        "null_signed_95": float(np.quantile(null_signed, 0.95)),
    }


def bootstrap_auroc(scores, labels, n_bootstrap, seed=0):
    """Component bootstrap for a small set of directions."""
    labels = np.asarray(labels, dtype=bool)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        take_positive = rng.choice(positive, size=len(positive), replace=True)
        take_negative = rng.choice(negative, size=len(negative), replace=True)
        index = np.concatenate([take_positive, take_negative])
        target = np.concatenate([np.ones(len(take_positive), dtype=bool),
                                 np.zeros(len(take_negative), dtype=bool)])
        draws.append([auroc(scores[index, column], target)
                      for column in range(scores.shape[1])])
    return np.asarray(draws)


def correlation_with_elongation(features, elongations):
    elongations = np.asarray(elongations, dtype=np.float64)
    values = []
    for column in range(features.shape[1]):
        x = features[:, column]
        if x.std() == 0:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(x, elongations)[0, 1]))
    return np.array(values)


# ------------------------------------------------------------------ linear probes


def ridge_fit(features, target, penalty=1.0):
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    centre = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-8
    design = np.c_[np.ones(len(features)), (features - centre) / scale]
    regulariser = np.eye(design.shape[1]) * penalty
    regulariser[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regulariser, design.T @ target)
    return {"centre": centre, "scale": scale, "weights": weights}


def ridge_predict(model, features):
    features = np.asarray(features, dtype=np.float64)
    design = np.c_[np.ones(len(features)), (features - model["centre"]) / model["scale"]]
    return design @ model["weights"]


def r2_score(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return float(1.0 - residual / max(total, 1e-12))


def probe_layer(discovery_records, evaluation_records, layer, penalty=1.0):
    """Continuous and binary probes for one layer, fit on discovery and scored on evaluation."""
    train = [i for i, record in enumerate(discovery_records)
             if record["class"] in ("stick", "round", "other")]
    test = [i for i, record in enumerate(evaluation_records)
            if record["class"] in ("stick", "round", "other")]
    train_x = np.stack([discovery_records[i]["mean_token"][layer] for i in train])
    test_x = np.stack([evaluation_records[i]["mean_token"][layer] for i in test])
    train_e = np.array([discovery_records[i]["elongation"] for i in train])
    test_e = np.array([evaluation_records[i]["elongation"] for i in test])
    continuous = ridge_fit(train_x, train_e, penalty)
    continuous_prediction = ridge_predict(continuous, test_x)
    train_y = np.array([discovery_records[i]["class"] == "stick" for i in train], dtype=float)
    test_y = np.array([evaluation_records[i]["class"] == "stick" for i in test], dtype=bool)
    binary = ridge_fit(train_x, train_y, penalty)
    binary_prediction = ridge_predict(binary, test_x)
    return {
        "continuous_r2": r2_score(test_e, continuous_prediction),
        "binary_auroc": float(auroc(binary_prediction, test_y)),
        "continuous_weights_raw": (continuous["weights"][1:] / continuous["scale"]).tolist(),
        "binary_weights_raw": (binary["weights"][1:] / binary["scale"]).tolist(),
    }


def raw_direction(probe):
    weights = np.asarray(probe["binary_weights_raw"], dtype=np.float64)
    norm = np.linalg.norm(weights)
    return weights / max(norm, 1e-12)


# ------------------------------------------------------------------ figures


def snapshot_state(block, regime, model, sae, layer, feature, mean, std, class_name):
    """The state of one regime with the most components of a class, its field and activation."""
    states = block["inputs"][block["regime"] == regime]
    best = None
    for state in states:
        v = state[-1, 1] * std[1] + mean[1]
        components = describe_field(v)
        if class_name == "stick":
            chosen = [c for c in components if c["elongation"] >= ELONGATION_MIN
                      and c["solidity"] >= STICK_SOLIDITY]
        else:
            chosen = [c for c in components if c["elongation"] <= ROUND_MAX_ELONGATION
                      and c["solidity"] >= STICK_SOLIDITY
                      and ROUND_MIN_AREA <= c["area"] <= ROUND_MAX_AREA]
        if best is None or len(chosen) > best[0]:
            best = (len(chosen), state, v, components)
    _, state, v, components = best
    tokens = token_activations(model, state[None], layer, np.array([regime]))
    with torch.no_grad():
        activation = sae.encode(tokens).reshape(*v.shape, -1)[:, :, feature].numpy()
    return {"v": v, "activation": activation, "components": components}


def class_cells(components, shape, predicate):
    """A binary mask of the components that satisfy ``predicate``."""
    mask = np.zeros(shape, dtype=bool)
    for component in components:
        if predicate(component):
            rows, cols = component["cells"].T
            mask[rows, cols] = True
    return mask


def component_mean_map(components, activation):
    """Each component filled with the feature's mean activation over its cells."""
    out = np.full(activation.shape, np.nan)
    for component in components:
        rows, cols = component["cells"].T
        out[rows, cols] = float(activation[rows, cols].mean())
    return out


def make_snapshots(model, sae, layer, block, mean, std, feature):
    """One stick-regime and one blob-regime test state, for the qualitative row."""
    stick = snapshot_state(block, 1, model, sae, layer, feature, mean, std, "stick")
    blob = snapshot_state(block, 0, model, sae, layer, feature, mean, std, "round")
    return stick, blob


def is_stick(component):
    return component["elongation"] >= ELONGATION_MIN and component["solidity"] >= STICK_SOLIDITY


def is_blob(component):
    return (component["elongation"] <= ROUND_MAX_ELONGATION
            and component["solidity"] >= STICK_SOLIDITY
            and ROUND_MIN_AREA <= component["area"] <= ROUND_MAX_AREA)


def component_figure(primary, secondary, probe, activation, snapshots, out_path):
    fig = plt.figure(figsize=(13.0, 12.6))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.05])
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 0:2]), fig.add_subplot(grid[1, 2:4])]

    axis = axes[0]
    scores = np.asarray(primary["scores"], dtype=float)
    axis.hist(scores, bins=60, color="#bbbbbb", edgecolor="#888888")
    axis.set_yscale("log")
    axis.set_ylim(0.8, None)
    axis.axvline(0.5, color="#444444", lw=1.0, linestyle=":", label="chance")
    null95 = primary["null"]["null_positive_95"]
    axis.axvline(null95, color="#d95f02", lw=1.2, linestyle="--",
                 label=f"null 95th pct {null95:.3f}")
    axis.axvline(primary["selected_auroc"], color=STICK_COLOUR, lw=1.5,
                 label=f"selected {primary['selected_feature']} = "
                       f"{primary['selected_auroc']:.3f}")
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel("component AUROC (stick vs blob, stick regime)")
    axis.set_ylabel("directions (log scale)")
    axis.set_title("512 direction scores", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    axis = axes[1]
    names = ["direction 159", "direction 74"]
    stick = [activation["stick_feature"]["stick"], activation["round_feature"]["stick"]]
    blob = [activation["stick_feature"]["round"], activation["round_feature"]["round"]]
    positions = np.arange(len(names))
    axis.bar(positions - 0.18, stick, width=0.36, color=STICK_COLOUR,
             label="stick components")
    axis.bar(positions + 0.18, blob, width=0.36, color=ROUND_COLOUR,
             label="blob components")
    for position, value in zip(positions - 0.18, stick):
        axis.text(position, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for position, value in zip(positions + 0.18, blob):
        axis.text(position, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_ylabel("mean activation")
    axis.set_title("Mean activation by class", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)

    axis = axes[2]
    transfer = (secondary["evaluation_scores"][primary["selected_feature"]]
                if secondary and "evaluation_scores" in secondary else np.nan)
    values = [primary["selected_evaluation_auroc"], secondary["selected_evaluation_auroc"],
              transfer]
    labels = ["159\nstick regime", "74\nblob regime", "159\nin blob regime"]
    axis.bar(np.arange(len(values)), values, width=0.6,
             color=[STICK_COLOUR, ROUND_COLOUR, "#999999"])
    axis.axhline(0.5, color="#444444", lw=1.0, linestyle=":")
    for position, value in enumerate(values):
        axis.text(position, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("AUROC on test components")
    axis.set_title("Test AUROC within and across regimes", fontsize=10)
    axis.grid(axis="y", alpha=0.3)

    axis = axes[3]
    layers = list(probe["layers"])
    positions = np.arange(len(layers))
    axis.bar(positions - 0.18, [probe["layers"][layer]["continuous_r2"] for layer in layers],
             width=0.36, color="#4c72b0", label="continuous R²")
    axis.bar(positions + 0.18, [probe["layers"][layer]["binary_auroc"] for layer in layers],
             width=0.36, color=STICK_COLOUR, label="binary AUROC")
    axis.set_xticks(positions)
    axis.set_xticklabels(layers, rotation=20, ha="right", fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("score on test components")
    axis.set_title("Linear probe from the token", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)

    feature = primary["selected_feature"]
    stick_snapshot, blob_snapshot = snapshots
    mean_maps = [component_mean_map(snapshot["components"], snapshot["activation"])
                 for snapshot in (stick_snapshot, blob_snapshot)]
    finite = np.concatenate([values[np.isfinite(values)] for values in mean_maps])
    low, high = np.quantile(finite, [0.02, 0.98])
    activation_axes = []
    activation_images = []
    for position, snapshot, values, title, is_field in (
            (grid[2, 0], stick_snapshot, mean_maps[0], "stick regime: $v$", True),
            (grid[2, 1], stick_snapshot, mean_maps[0],
             f"direction {feature}: mean activation", False),
            (grid[2, 2], blob_snapshot, mean_maps[1], "blob regime: $v$", True),
            (grid[2, 3], blob_snapshot, mean_maps[1],
             f"direction {feature}: mean activation", False)):
        panel = fig.add_subplot(position)
        panel.imshow(snapshot["v"], cmap="gray", vmin=0.0, vmax=0.45)
        if not is_field:
            image = panel.imshow(values, cmap="viridis", vmin=low, vmax=high)
            activation_axes.append(panel)
            activation_images.append(image)
        sticks = class_cells(snapshot["components"], snapshot["v"].shape, is_stick)
        blobs = class_cells(snapshot["components"], snapshot["v"].shape, is_blob)
        panel.contour(sticks, levels=[0.5], colors=["white"], linewidths=1.0)
        panel.contour(blobs, levels=[0.5], colors=[ROUND_COLOUR], linewidths=0.8)
        panel.set_title(title, fontsize=9)
        panel.set_xticks([])
        panel.set_yticks([])
    fig.colorbar(activation_images[-1], ax=activation_axes[-1], fraction=0.046, pad=0.02,
                 label=f"direction {feature}: mean activation")

    fig.text(0.5, 0.005,
             "Top: component-mean activation; the probe uses the 32-number token, not the SAE.\n"
             "Bottom: the same mean for each component; white outlines are sticks, blue "
             "outlines blobs.",
             ha="center", va="bottom", fontsize=9, color="#444444", linespacing=1.6)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(
        description="Component-level elongation analysis with nulls and linear probes.")
    parser.add_argument("--data", default="data/gray_scott_mixed.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--layer", default="blocks.6")
    parser.add_argument("--probe-layers", default="blocks.4,blocks.5,blocks.6")
    parser.add_argument("--discovery-split", choices=("val", "test"), default="val")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test")
    parser.add_argument("--states", type=int, default=None,
                        help="states per regime; default uses every state in the block")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-directions", type=int, default=64)
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw the figure from the saved measurements")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    model, _ = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    config = checkpoint["config"]
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    layers = [layer.strip() for layer in args.probe_layers.split(",") if layer.strip()]
    if args.layer not in layers:
        layers.append(args.layer)

    discovery_block = dataset["splits"][args.discovery_split]
    evaluation_block = dataset["splits"][args.evaluation_split]

    if args.figure_only:
        results = json.loads((Path(args.model_dir) / "stick_components.json").read_text())
        figure_probes = {layer: {"continuous_r2": results["probe"]["continuous"][layer]["continuous_r2"],
                                 "binary_auroc": results["probe"]["binary"][layer]["auroc"]}
                         for layer in results["probe_layers"]}
        snapshots = make_snapshots(model, sae, args.layer, evaluation_block, mean, std,
                                   results["primary"]["selected_feature"])
        component_figure(results["primary"], results["secondary"],
                         {"layers": figure_probes}, results["activation"], snapshots,
                         Path(args.figures) / "stick_components.png")
        print(f"redrew {Path(args.figures) / 'stick_components.png'}")
        return

    print(f"\nharvesting components at {args.layer} and probes at {layers}")
    records = {}
    firing_rates = {}
    for split, block in (("discovery", discovery_block), ("evaluation", evaluation_block)):
        for regime in (0, 1):
            key = (split, regime)
            records[key], firing_rates[key] = harvest(
                block, regime, model, sae, layers, args.layer, mean, std, args.states)
            counts = {name: sum(record["class"] == name for record in records[key])
                      for name in ("stick", "round", "labyrinth", "other")}
            print(f"  {split:10s} regime {regime}: {len(records[key]):5d} components {counts}")

    primary_regime = 1
    secondary_regime = 0
    firings = firing_rates[("discovery", primary_regime)]
    by_name = {}
    for name, regime, positive, negative in (
            ("primary", primary_regime, "stick", "round"),
            ("secondary", secondary_regime, "stick", "round")):
        discovery_records = records[("discovery", regime)]
        evaluation_records = records[("evaluation", regime)]
        discovery_positive, discovery_negative = matched_indices(
            discovery_records, positive, negative, args.seed)
        evaluation_positive, evaluation_negative = matched_indices(
            evaluation_records, positive, negative, args.seed + 1)
        if not discovery_positive or not evaluation_positive:
            print(f"  {name}: not enough {positive}/{negative} components")
            by_name[name] = {}
            continue
        discovery_index = discovery_positive + discovery_negative
        evaluation_index = evaluation_positive + evaluation_negative
        discovery_features = np.stack([discovery_records[i]["mean_feature"]
                                       for i in discovery_index])
        evaluation_features = np.stack([evaluation_records[i]["mean_feature"]
                                        for i in evaluation_index])
        discovery_labels = np.array([discovery_records[i]["class"] == positive
                                     for i in discovery_index])
        evaluation_labels = np.array([evaluation_records[i]["class"] == positive
                                      for i in evaluation_index])
        ranks = average_ranks(discovery_features)
        scores = auroc_from_ranks(ranks, discovery_labels)
        null = permutation_null(ranks, discovery_labels, args.permutations, args.seed)
        # The selected direction is chosen on discovery, scored on evaluation.
        choice = int(np.nanargmax(scores))
        evaluation_scores = auroc(evaluation_features[:, choice], evaluation_labels)
        correlations = correlation_with_elongation(
            np.stack([record["mean_feature"] for record in discovery_records]),
            np.array([record["elongation"] for record in discovery_records]))
        entry = {
            "regime": int(regime),
            "positive_class": positive,
            "negative_class": negative,
            "n_positive": int(len(discovery_positive)),
            "n_negative": int(len(discovery_negative)),
            "matched_regime_counts": {
                "positive": sum(discovery_records[i]["regime"] for i in discovery_positive),
                "negative": sum(discovery_records[i]["regime"] for i in discovery_negative),
            },
            "selected_feature": choice,
            "selected_auroc": float(scores[choice]),
            "selected_evaluation_auroc": float(evaluation_scores),
            "selected_correlation": float(correlations[choice]),
            "selected_firing_rate": float(firings[choice]) if len(firings) else 0.0,
            "scores": [float(value) for value in scores],
            "evaluation_scores": [float(auroc(evaluation_features[:, k], evaluation_labels))
                                  for k in range(evaluation_features.shape[1])],
            "correlations": [float(value) for value in correlations],
            "null": null,
        }
        by_name[name] = entry

    primary = by_name.get("primary", {})
    secondary = by_name.get("secondary", {})

    # ---------------------------------------------------- selection and controls
    if not primary or "scores" not in primary:
        raise SystemExit("the primary contrast has too few components to select a direction")
    scores = np.asarray(primary["scores"])
    correlations = np.asarray(primary["correlations"])
    stick_choice = int(np.nanargmax(scores))
    round_choice = int(np.nanargmax(1.0 - scores))
    signed_choice = int(np.nanargmax(np.maximum(scores, 1.0 - scores)))
    signed_sign = 1.0 if scores[signed_choice] >= 0.5 else -1.0

    # A non-shape control: low |correlation|, AUROC near 0.5, firing rate near the selected.
    eligible = (firings > MIN_FIRING_RATE) & (np.abs(correlations) < 0.10) \
        & (np.abs(scores - 0.5) < 0.08)
    if not eligible.any():
        eligible = (firings > MIN_FIRING_RATE) & (np.abs(correlations) < 0.20)
    eligible[[stick_choice, round_choice, signed_choice]] = False
    candidates = np.flatnonzero(eligible)
    matched_choice = int(candidates[np.argmin(np.abs(firings[candidates] - firings[stick_choice]))]) \
        if len(candidates) else int(np.nanargmin(np.abs(scores - 0.5)))

    # Random directions in token space, for the injection null.
    discovery_records = records[("discovery", primary_regime)]
    tokens = np.stack([record["mean_token"][args.layer] for record in discovery_records])
    labels = np.array([record["class"] == "stick" for record in discovery_records])
    rng = np.random.default_rng(args.seed)
    random_scores, random_directions = [], []
    for _ in range(args.random_directions):
        direction = rng.normal(size=tokens.shape[1])
        direction /= np.linalg.norm(direction)
        random_scores.append(float(auroc(tokens @ direction, labels)))
        random_directions.append(direction.tolist())
    random_scores = np.asarray(random_scores)
    order = np.argsort(np.abs(random_scores - 0.5))[:8]
    random_directions = [random_directions[i] for i in order]
    random_scores = [float(random_scores[i]) for i in order]

    # Bootstrap confidence intervals for the selected directions.
    primary_positive, primary_negative = matched_indices(
        records[("discovery", primary_regime)], "stick", "round", args.seed)
    primary_index = primary_positive + primary_negative
    primary_features = np.stack([records[("discovery", primary_regime)][i]["mean_feature"]
                                 for i in primary_index])
    primary_labels = np.array([records[("discovery", primary_regime)][i]["class"] == "stick"
                               for i in primary_index])
    chosen = np.array([stick_choice, round_choice, signed_choice])
    bootstrap = bootstrap_auroc(primary_features[:, chosen], primary_labels,
                                args.bootstrap, args.seed)
    bootstrap_summary = {
        "stick_feature": [float(np.quantile(bootstrap[:, 0], 0.025)),
                          float(np.quantile(bootstrap[:, 0], 0.975))],
        "round_feature": [float(np.quantile(bootstrap[:, 1], 0.025)),
                          float(np.quantile(bootstrap[:, 1], 0.975))],
        "signed_feature": [float(np.quantile(bootstrap[:, 2], 0.025)),
                           float(np.quantile(bootstrap[:, 2], 0.975))],
    }

    # ---------------------------------------------------- probes
    # For continuous R2, use every component in the primary regime.
    probes = {}
    for layer in layers:
        probes[layer] = probe_layer(records[("discovery", primary_regime)],
                                    records[("evaluation", primary_regime)], layer)
    # Binary probe on the matched primary sets at each layer.
    binary_probes = {}
    for layer in layers:
        train_x = np.stack([records[("discovery", primary_regime)][i]["mean_token"][layer]
                            for i in primary_index])
        test_positive, test_negative = matched_indices(
            records[("evaluation", primary_regime)], "stick", "round", args.seed + 1)
        test_index = test_positive + test_negative
        test_x = np.stack([records[("evaluation", primary_regime)][i]["mean_token"][layer]
                           for i in test_index])
        y_train = np.array([records[("discovery", primary_regime)][i]["class"] == "stick"
                            for i in primary_index], dtype=float)
        y_test = np.array([records[("evaluation", primary_regime)][i]["class"] == "stick"
                           for i in test_index], dtype=bool)
        fitted = ridge_fit(train_x, y_train)
        binary_probes[layer] = {
            "auroc": float(auroc(ridge_predict(fitted, test_x), y_test)),
            "direction": raw_direction({"binary_weights_raw": fitted["weights"][1:]
                                        / fitted["scale"]}).tolist(),
        }
    probe_direction = np.asarray(binary_probes[args.layer]["direction"], dtype=np.float64)
    probe_direction /= max(np.linalg.norm(probe_direction), 1e-12)

    def means(feature, indices):
        return float(np.mean([records[("discovery", primary_regime)][i]["mean_feature"][feature]
                              for i in indices]))

    activation = {
        name: {"stick": means(feature, primary_positive),
               "round": means(feature, primary_negative)}
        for name, feature in (("stick_feature", stick_choice),
                              ("round_feature", round_choice),
                              ("signed_feature", signed_choice))
    }

    def descriptors(indices):
        return {key: float(np.mean([records[("discovery", primary_regime)][i][key] for i in indices]))
                for key in ("elongation", "area", "solidity", "skeleton_length",
                            "skeleton_width", "branch_points")}

    descriptors_by_class = {"stick": descriptors(primary_positive),
                            "round": descriptors(primary_negative)}

    # ---------------------------------------------------- write and report
    results = {
        "layer": args.layer,
        "probe_layers": layers,
        "thresholds": {
            "stick_solidity": STICK_SOLIDITY,
            "round_max_elongation": ROUND_MAX_ELONGATION,
            "round_area": [ROUND_MIN_AREA, ROUND_MAX_AREA],
            "labyrinth_solidity": LABYRINTH_SOLIDITY,
        },
        "class_counts": {
            f"{split}_{regime}": {name: sum(record["class"] == name
                                            for record in records[(split, regime)])
                                  for name in ("stick", "round", "labyrinth", "other")}
            for split in ("discovery", "evaluation") for regime in (0, 1)
        },
        "primary": primary,
        "secondary": secondary,
        "selected": {
            "stick_feature": stick_choice,
            "round_feature": round_choice,
            "signed_feature": signed_choice,
            "signed_sign": signed_sign,
            "matched_nonshape": matched_choice,
            "probe_direction": probe_direction.tolist(),
            "random_directions": random_directions,
            "random_scores": random_scores,
        },
        "activation": activation,
        "descriptors_by_class": descriptors_by_class,
        "bootstrap": bootstrap_summary,
        "secondary_by_primary_selection": {
            "feature": stick_choice,
            "evaluation_auroc": (float(secondary["evaluation_scores"][stick_choice])
                                 if secondary and "evaluation_scores" in secondary else None),
        },
        "probe": {
            "continuous": probes,
            "binary": binary_probes,
        },
    }
    model_dir = Path(args.model_dir)
    (model_dir / "stick_components.json").write_text(json.dumps(results, indent=2))
    figure_probes = {layer: {"continuous_r2": probes[layer]["continuous_r2"],
                             "binary_auroc": binary_probes[layer]["auroc"]}
                     for layer in layers}
    snapshots = make_snapshots(model, sae, args.layer, evaluation_block, mean, std,
                               stick_choice)
    component_figure(primary, secondary, {"layers": figure_probes}, activation, snapshots,
                     Path(args.figures) / "stick_components.png")

    print(f"\nprimary contrast (stick regime): {primary['n_positive']} stick vs "
          f"{primary['n_negative']} matched round components")
    print(f"  selected positive direction {stick_choice}: "
          f"AUROC {primary['selected_auroc']:.3f} discovery, "
          f"{primary['selected_evaluation_auroc']:.3f} test")
    print(f"  permutation p (max over {len(scores)} directions): "
          f"{primary['null']['p_positive']:.4f}; null 95th percentile "
          f"{primary['null']['null_positive_95']:.3f}")
    print(f"  bootstrap 95% CI: {bootstrap_summary['stick_feature'][0]:.3f}"
          f"-{bootstrap_summary['stick_feature'][1]:.3f}")
    print(f"  round direction {round_choice}: 1-AUROC "
          f"{1.0 - scores[round_choice]:.3f} discovery, "
          f"{1.0 - primary['evaluation_scores'][round_choice]:.3f} test")
    print(f"  signed direction {signed_choice} (sign {signed_sign:+.0f})")
    print(f"  matched non-shape control {matched_choice}: AUROC {scores[matched_choice]:.3f}")
    if secondary:
        print(f"\nsecondary contrast (blob regime): {secondary['n_positive']} stick vs "
              f"{secondary['n_negative']} matched round components")
        print(f"  primary direction {stick_choice}: AUROC {scores[stick_choice]:.3f} "
              f"discovery, {secondary['evaluation_scores'][stick_choice]:.3f} test")
    print("\nprobe (continuous R2 / binary AUROC):")
    for layer in layers:
        print(f"  {layer:10s} {probes[layer]['continuous_r2']:.3f} / "
              f"{binary_probes[layer]['auroc']:.3f}")
    print(f"\nwrote {model_dir / 'stick_components.json'}")
    print(f"wrote {Path(args.figures) / 'stick_components.png'}")


if __name__ == "__main__":
    main()
