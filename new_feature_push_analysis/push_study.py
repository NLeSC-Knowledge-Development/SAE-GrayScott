"""The constant-push study: hold the stick feature on and watch sticks grow.

At every call we encode the emulator's block-6 tokens with the SAE, add a constant multiple
of the mean token norm to the selected stick latent inside a stick-shaped footprint, decode
the modified latents, and run the emulator on the decoded token. The control runs the same
encode/decode path with no added activation, so the only difference is the stick latent.

We follow what happens near the three sites for twenty calls: the share of the grid in
elongated components, and the number of elongated components. A sweep over the push strength
shows whether more push gives more sticks.

Writes ``model/stick_push.json`` and ``figures/stick_push.png``.
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
from stick_study import (N_DISCOVERY_STATES, STICK_COLOUR, bootstrap_interval, choose_sites,
                         sae_pushed_rollout, states_for_regime, stick_figure)
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder

CONTROL_COLOUR = "#4c72b0"
NEAR_RADIUS = 8
SWEEP = (0.0, 2.0, 4.0, 6.0)
PUSH_STRENGTH = 4.0


def near_cells(shape, sites, radius):
    """Cells within ``radius`` of any site, on the periodic grid."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    near = np.zeros(shape, dtype=bool)
    for row, col in sites:
        dr = np.minimum(np.abs(rows - row), shape[0] - np.abs(rows - row))
        dc = np.minimum(np.abs(cols - col), shape[1] - np.abs(cols - col))
        near |= (dr ** 2 + dc ** 2) <= radius ** 2
    return near


def field_sticks(field, near):
    """Near-site share, near-site count and global share of cells in elongated components."""
    area = count = global_area = 0
    for component in connected_components(field):
        if component["elongation"] < ELONGATION_MIN:
            continue
        global_area += component["area"]
        rows, cols = component["cells"].T
        if near[rows, cols].any():
            area += component["area"]
            count += 1
    return area / near.sum(), count, global_area / field.size


def curve(values, seed):
    """Mean and 95% band at every call for an array of shape (pairs, calls)."""
    mean = values.mean(axis=0)
    low = np.empty(values.shape[1])
    high = np.empty(values.shape[1])
    for call in range(values.shape[1]):
        _, low[call], high[call] = bootstrap_interval(values[:, call], seed)
    return mean, low, high


def trajectory_figure(history, strengths, calls, out_path, seed=0):
    calls_axis = np.arange(1, calls + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))

    series = (("near_share", "share of near cells in elongated components", axes[0]),
              ("near_count", "elongated components near a site", axes[1]))
    for key, label, axis in series:
        for name, strength, colour in (("without the push", 0.0, CONTROL_COLOUR),
                                       ("with the push", PUSH_STRENGTH, STICK_COLOUR)):
            mean, low, high = curve(history[strength][key], seed)
            axis.plot(calls_axis, mean, color=colour, lw=2, label=name)
            axis.fill_between(calls_axis, low, high, color=colour, alpha=0.18, lw=0)
        axis.set_xlabel("call")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_title("stick content near the sites", fontsize=10)
    axes[1].set_title("number of sticks near the sites", fontsize=10)

    axis = axes[2]
    for strength in strengths:
        final = history[strength]["near_share"][:, -1]
        mean, low, high = bootstrap_interval(final, seed)
        colour = STICK_COLOUR if strength == PUSH_STRENGTH else "#777777"
        axis.errorbar([strength], [mean], yerr=[[mean - low], [high - mean]],
                      color=colour, marker="o", capsize=3)
    axis.set_xlabel("push strength (mean token norms)")
    axis.set_ylabel("share of cells in elongated components")
    axis.set_title("final stick content by strength", fontsize=10)
    axis.grid(alpha=0.3)

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
    sites_per_state = [choose_sites(rng, n, args.sites) for _ in blob_states]
    near_per_state = [near_cells((n, n), sites, args.radius) for sites in sites_per_state]

    amplitudes = []
    scales = []
    for state, sites in zip(blob_states, sites_per_state):
        amplitude = np.zeros((n, n), dtype=np.float64)
        for row, col in sites:
            site_amplitude, _, _ = stick_figure(
                model, sae, layer, stick_states, args.stick_regime, feature,
                row, col, rng, mean, std)
            amplitude = np.maximum(amplitude, site_amplitude)
        amplitudes.append(amplitude)
        scales.append(float(token_activations(model, state[None], layer,
                                              np.array([args.blob_regime])).norm(dim=1).mean()))

    strengths = SWEEP
    history = {strength: {key: [] for key in ("near_share", "near_count",
                                              "global_share", "coverage")}
               for strength in strengths}
    for state, amplitude, scale, near in zip(blob_states, amplitudes, scales, near_per_state):
        for strength in strengths:
            fields = sae_pushed_rollout(model, layer, sae, feature, state, args.blob_regime,
                                        strength * scale, amplitude, args.calls)
            near_share, near_count, global_share, cover = [], [], [], []
            for call in range(args.calls):
                v = fields[call, 1] * std[1] + mean[1]
                share, count, global_cells = field_sticks(v, near)
                near_share.append(share)
                near_count.append(count)
                global_share.append(global_cells)
                cover.append(coverage(v))
            history[strength]["near_share"].append(near_share)
            history[strength]["near_count"].append(near_count)
            history[strength]["global_share"].append(global_share)
            history[strength]["coverage"].append(cover)

    for strength in strengths:
        for key in history[strength]:
            history[strength][key] = np.asarray(history[strength][key])

    control = history[0.0]
    push = history[PUSH_STRENGTH]

    def summary(metric):
        difference = push[metric][:, -1] - control[metric][:, -1]
        average, low, high = bootstrap_interval(difference, args.seed)
        return {
            "control": float(control[metric][:, -1].mean()),
            "push": float(push[metric][:, -1].mean()),
            "difference": average,
            "difference_ci": [low, high],
        }

    new_stick = (push["near_count"][:, -1] > control["near_count"][:, -1]).astype(float)
    new_share, new_low, new_high = bootstrap_interval(new_stick, args.seed)

    time_series = {"calls": list(range(1, args.calls + 1))}
    for key in ("near_share", "near_count", "global_share", "coverage"):
        for name, values in (("control", control[key]), ("push", push[key])):
            average, low, high = curve(values, args.seed)
            time_series[f"{name}_{key}"] = average.tolist()
            time_series[f"{name}_{key}_ci"] = [low.tolist(), high.tolist()]

    sweep = {"strengths": list(strengths), "near_share": [], "near_share_ci": [],
             "near_count": [], "near_count_ci": []}
    for strength in strengths:
        for key, target, target_ci in (("near_share", sweep["near_share"],
                                        sweep["near_share_ci"]),
                                       ("near_count", sweep["near_count"],
                                        sweep["near_count_ci"])):
            average, low, high = bootstrap_interval(history[strength][key][:, -1], args.seed)
            target.append(float(average))
            target_ci.append([low, high])

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
        "push_strength": float(PUSH_STRENGTH),
        "outcomes": {
            "near_share": summary("near_share"),
            "near_count": summary("near_count"),
            "global_share": summary("global_share"),
            "coverage": summary("coverage"),
            "new_stick": {"share": new_share, "ci": [new_low, new_high]},
        },
        "time_series": time_series,
        "sweep": sweep,
    }
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "stick_push.json").write_text(json.dumps(results, indent=2))

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    trajectory_figure(history, strengths, args.calls, figures / "stick_push.png", args.seed)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hold the stick feature on through the SAE and follow the sticks.")
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
