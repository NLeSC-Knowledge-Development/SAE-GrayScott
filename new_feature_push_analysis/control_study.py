"""Control: add v directly at the same sites instead of pushing the SAE feature.

The feature push adds a constant multiple of the mean token norm to the stick latent at every
call. Here we build two non-SAE controls that change the v field at the same three sites, with
the same per-call RMS change as the feature push on its first call:

* ``random_v``: a smooth random pattern on the footprint;
* ``footprint_v``: the footprint itself, added to v.

Both use the same SAE round-trip with no added activation as the feature control, so all
conditions share the reconstruction path. We follow near-site stick content for twenty calls.

Writes ``model/stick_controls.json`` and ``figures/stick_controls.png``.
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

from generate_data import ELONGATION_MIN, connected_components, coverage, load_dataset
from stick_study import (FOOTPRINT_SIGMA, N_DISCOVERY_STATES, STICK_COLOUR, bootstrap_interval,
                         choose_sites, periodic_blur, sae_pushed_rollout, states_for_regime,
                         stick_figure)
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder

CONTROL_COLOUR = "#4c72b0"
FOOTPRINT_COLOUR = "#dd8452"
RANDOM_COLOUR = "#999999"
NEAR_RADIUS = 8
STRENGTH = 4.0
CONDITIONS = ("feature", "sae_control", "footprint_v", "random_v")
LABELS = {"feature": "SAE push", "sae_control": "SAE without the push",
          "footprint_v": "v in the same footprint", "random_v": "random v"}
COLOURS = {"feature": STICK_COLOUR, "sae_control": CONTROL_COLOUR,
           "footprint_v": FOOTPRINT_COLOUR, "random_v": RANDOM_COLOUR}


def near_cells(shape, sites, radius):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    near = np.zeros(shape, dtype=bool)
    for row, col in sites:
        dr = np.minimum(np.abs(rows - row), shape[0] - np.abs(rows - row))
        dc = np.minimum(np.abs(cols - col), shape[1] - np.abs(cols - col))
        near |= (dr ** 2 + dc ** 2) <= radius ** 2
    return near


def near_stats(field, near):
    """Near-site share, count, mean area and global share of elongated components."""
    area = count = global_area = 0
    areas = []
    for component in connected_components(field):
        if component["elongation"] < ELONGATION_MIN:
            continue
        global_area += component["area"]
        rows, cols = component["cells"].T
        if near[rows, cols].any():
            area += component["area"]
            areas.append(component["area"])
            count += 1
    mean_area = float(np.mean(areas)) if areas else 0.0
    return area / near.sum(), count, mean_area, global_area / field.size


def random_pattern(rng, support, sigma, near, target):
    values = rng.random(support.shape) * support
    values = periodic_blur(values, sigma)
    values = np.clip(values, 0.0, None)
    rms = np.sqrt(np.mean(values[near] ** 2))
    return values * (target / max(rms, 1e-12))


def scaled_pattern(pattern, near, target):
    rms = np.sqrt(np.mean(pattern[near] ** 2))
    return pattern * (target / max(rms, 1e-12))


def curve(values, seed):
    mean = values.mean(axis=0)
    low = np.empty(values.shape[1])
    high = np.empty(values.shape[1])
    for call in range(values.shape[1]):
        _, low[call], high[call] = bootstrap_interval(values[:, call], seed)
    return mean, low, high


def control_figure(history, calls, out_path, seed=0):
    calls_axis = np.arange(1, calls + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
    series = (("near_share", "share of near cells in elongated components", axes[0]),
              ("near_area", "mean area of those components", axes[1]),
              ("coverage", "coverage", axes[2]))
    for key, label, axis in series:
        for name in CONDITIONS:
            mean, low, high = curve(history[name][key], seed)
            axis.plot(calls_axis, mean, color=COLOURS[name], lw=2, label=LABELS[name])
            axis.fill_between(calls_axis, low, high, color=COLOURS[name], alpha=0.15, lw=0)
        axis.set_xlabel("call")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("stick content near the sites", fontsize=10)
    axes[1].set_title("size of the sticks near the sites", fontsize=10)
    axes[2].set_title("coverage of the field", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(args):
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    n = settings["N"]

    model, _ = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    config = checkpoint["config"]
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    layer = config["layer"]
    selection = json.loads(
        (Path(args.model_dir) / "stick_components.json").read_text())["selected"]
    feature = int(selection["stick_feature"])

    block = dataset["splits"][args.evaluation_split]
    blob_states, _ = states_for_regime(block, args.blob_regime, args.pairs, args.seed + 2)
    stick_states, _ = states_for_regime(block, args.stick_regime,
                                        max(N_DISCOVERY_STATES, args.pairs), args.seed + 3)
    rng = np.random.default_rng(args.seed)
    pattern_rng = np.random.default_rng(args.seed + 100)
    sites_per_state = [choose_sites(rng, n, args.sites) for _ in blob_states]

    history = {name: {key: [] for key in ("near_share", "near_count", "near_area", "coverage")}
               for name in CONDITIONS}
    for state, sites in zip(blob_states, sites_per_state):
        amplitude = np.zeros((n, n), dtype=np.float64)
        for row, col in sites:
            site_amplitude, _, _ = stick_figure(
                model, sae, layer, stick_states, args.stick_regime, feature,
                row, col, rng, mean, std)
            amplitude = np.maximum(amplitude, site_amplitude)
        near = near_cells((n, n), sites, args.radius)
        scale = float(token_activations(model, state[None], layer,
                                        np.array([args.blob_regime])).norm(dim=1).mean())
        pushed = sae_pushed_rollout(model, layer, sae, feature, state, args.blob_regime,
                                    STRENGTH * scale, amplitude, 1)
        control = sae_pushed_rollout(model, layer, sae, feature, state, args.blob_regime,
                                     0.0, amplitude, 1)
        target = float(np.sqrt(np.mean((pushed[0, 1] - control[0, 1])[near] ** 2)))
        support = amplitude > 0
        deltas = {
            "random_v": random_pattern(pattern_rng, support, FOOTPRINT_SIGMA, near, target),
            "footprint_v": scaled_pattern(amplitude, near, target),
        }
        runs = {
            "feature": dict(strength=STRENGTH * scale, field_delta=None),
            "sae_control": dict(strength=0.0, field_delta=None),
            **{name: dict(strength=0.0, field_delta=delta) for name, delta in deltas.items()},
        }
        for name, options in runs.items():
            fields = sae_pushed_rollout(model, layer, sae, feature, state, args.blob_regime,
                                        options["strength"], amplitude, args.calls,
                                        field_delta=options["field_delta"])
            share, count, area, cover = [], [], [], []
            for call in range(args.calls):
                v = fields[call, 1] * std[1] + mean[1]
                s, c, a, _ = near_stats(v, near)
                share.append(s)
                count.append(c)
                area.append(a)
                cover.append(coverage(v))
            history[name]["near_share"].append(share)
            history[name]["near_count"].append(count)
            history[name]["near_area"].append(area)
            history[name]["coverage"].append(cover)

    for name in CONDITIONS:
        for key in history[name]:
            history[name][key] = np.asarray(history[name][key])

    def endpoint(name, key, call):
        average, low, high = bootstrap_interval(history[name][key][:, call - 1], args.seed)
        return {"mean": float(average), "ci": [low, high]}

    def difference(name, key, call, reference="sae_control"):
        values = (history[name][key][:, call - 1] - history[reference][key][:, call - 1])
        average, low, high = bootstrap_interval(values, args.seed)
        return {"mean": float(average), "ci": [low, high]}

    endpoints = {}
    for call in (8, args.calls):
        endpoints[str(call)] = {
            name: {key: endpoint(name, key, call)
                   for key in ("near_share", "near_area", "coverage")}
            for name in CONDITIONS
        }
    differences = {}
    for call in (8, args.calls):
        differences[str(call)] = {
            name: {key: difference(name, key, call)
                   for key in ("near_share", "near_area")}
            for name in ("feature", "footprint_v", "random_v")
        }
    controls = {
        "calls": list(range(1, args.calls + 1)),
        "conditions": {
            name: {key: history[name][key].mean(axis=0).tolist() for key in history[name]}
            for name in CONDITIONS
        },
        "endpoints": endpoints,
        "differences_vs_sae_control": differences,
    }

    results = {
        "layer": layer,
        "blob_regime": int(args.blob_regime),
        "stick_regime": int(args.stick_regime),
        "regime_names": [str(name) for name in settings["regime_names"]],
        "evaluation_split": args.evaluation_split,
        "selected_feature": feature,
        "n_pairs": int(len(blob_states)),
        "n_sites": int(args.sites),
        "n_calls": int(args.calls),
        "near_radius": int(args.radius),
        "push_strength": float(STRENGTH),
        "controls": controls,
    }
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "stick_controls.json").write_text(json.dumps(results, indent=2))

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    control_figure(history, args.calls, figures / "stick_controls.png", args.seed)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the SAE push with matched v additions at the same sites.")
    parser.add_argument("--data", default="data/gray_scott_mixed.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--blob-regime", type=int, default=0)
    parser.add_argument("--stick-regime", type=int, default=1)
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test")
    parser.add_argument("--pairs", type=int, default=120)
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--radius", type=int, default=NEAR_RADIUS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
