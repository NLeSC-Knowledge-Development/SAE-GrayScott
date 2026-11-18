"""Explore how selected SAE features relate to continuous physical quantities."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import load_dataset
from physical_probes import quantities
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


QUESTIONS = (
    (117, "Local v"),
    (159, "V Laplacian"),
    (402, "Horizontal v gradient"),
    (371, "Vertical v gradient"),
    (463, "V gradient magnitude"),
    (45, "Distance to nearest centre"),
    (149, "Distance to second centre"),
    (94, "Previous v change"),
    (244, "Grid-mean v (control)"),
    (429, "Next true v change"),
)


def bin_statistics(quantity, activation, bins=10):
    """Summarise activity in quantile bins of a physical quantity."""
    x = np.asarray(quantity).reshape(-1)
    z = np.asarray(activation).reshape(-1)
    if len(x) != len(z):
        raise ValueError("each physical value must have one feature activation")
    valid = np.isfinite(x) & np.isfinite(z)
    x, z = x[valid], z[valid]
    if len(x) < 2 or bins < 1:
        raise ValueError("at least two finite observations and one bin are required")
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise ValueError("physical quantity is constant; bins are undefined")
    indices = np.searchsorted(edges[1:-1], x, side="right")
    rows = []
    for index in range(len(edges) - 1):
        mask = indices == index
        if not mask.any():
            continue
        active = z[mask & (z > 0)]
        median, low, high = (np.quantile(active, [0.5, 0.25, 0.75])
                             if len(active) else (np.nan, np.nan, np.nan))
        rows.append({"low": float(edges[index]), "high": float(edges[index + 1]),
                     "centre": float(x[mask].mean()), "count": int(mask.sum()),
                     "active_count": int(len(active)), "rate": float(len(active) / mask.sum()),
                     "median_on": float(median), "q25_on": float(low),
                     "q75_on": float(high)})
    x_centered = x.astype(np.float64) - x.mean()
    z_centered = z.astype(np.float64) - z.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(z_centered)
    correlation = float(x_centered @ z_centered / denominator) if denominator else None
    return rows, correlation


def example_index(feature, activation):
    """Select the most active state for 94; use the first state otherwise."""
    if feature != 94:
        return 0
    counts = np.count_nonzero(activation > 0, axis=(1, 2))
    if not counts.max():
        raise ValueError("feature 94 is off in every requested state")
    return int(np.argmax(counts))


def plot_relationship(feature, name, quantity, activation, rows, correlation, seed,
                      start, out_path):
    index = example_index(feature, activation)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    field = quantity[index]
    if np.isfinite(field).any():
        signed = np.nanmin(field) < 0 < np.nanmax(field)
        limit = float(np.nanmax(np.abs(field))) if signed else None
        image = axes[0, 0].imshow(field, cmap="coolwarm" if signed else "inferno",
                                  vmin=-limit if signed else None, vmax=limit)
        fig.colorbar(image, ax=axes[0, 0], fraction=0.046)
    else:
        axes[0, 0].text(0.5, 0.5, "Quantity undefined\nfor this state", ha="center",
                        va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_title(name)
    if np.max(activation[index]) > 0:
        image = axes[0, 1].imshow(activation[index], cmap="viridis", vmin=0)
        fig.colorbar(image, ax=axes[0, 1], fraction=0.046)
    else:
        axes[0, 1].imshow(activation[index], cmap="viridis", vmin=0, vmax=1)
        axes[0, 1].text(0.5, 0.5, "Feature is off\nin this state", ha="center",
                        va="center", color="white", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title(f"Feature {feature} activation")
    for axis in axes[0]:
        axis.set_xticks([])
        axis.set_yticks([])

    centre = [row["centre"] for row in rows]
    axes[1, 0].plot(centre, [100 * row["rate"] for row in rows], "o-", color="#dd8452")
    axes[1, 0].set_ylabel("Cells with feature on (%)")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_title("How often is the feature on?")
    axes[1, 1].plot(centre, [row["median_on"] for row in rows], "-", color="#4c72b0")
    reliable = [row for row in rows if row["active_count"] >= 30]
    sparse = [row for row in rows if 0 < row["active_count"] < 30]
    if reliable:
        axes[1, 1].scatter([row["centre"] for row in reliable],
                           [row["median_on"] for row in reliable], color="#4c72b0",
                           label="Median when on", zorder=3)
    if sparse:
        axes[1, 1].scatter([row["centre"] for row in sparse],
                           [row["median_on"] for row in sparse], facecolors="white",
                           edgecolors="#4c72b0", label="Fewer than 30 active cells", zorder=4)
    axes[1, 1].fill_between(centre, [row["q25_on"] for row in rows],
                            [row["q75_on"] for row in rows], color="#4c72b0", alpha=0.2,
                            label="Middle 50% when on")
    axes[1, 1].set_ylabel("Feature activation when on")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_title("How strong is it when on?")
    axes[1, 1].legend(fontsize=8, loc="best")
    padding = (max(centre) - min(centre)) * 0.02 or 0.01
    for axis in axes[1]:
        axis.set_xlabel(name)
        axis.set_xlim(min(centre) - padding, max(centre) + padding)
        axis.grid(alpha=0.25)
    r_text = "undefined" if correlation is None else f"{correlation:+.3f}"
    selected = " (selected for most active cells)" if feature == 94 else ""
    fig.suptitle(f"Feature {feature} and {name} | simulation seed {seed}", fontsize=14)
    fig.text(0.5, 0.015,
             f"Top: validation state {start + index}{selected}. Bottom: all cells from "
             f"{len(quantity)} validation states.\n"
             f"Raw Pearson r (all cells, including off): {r_text}. Shading shows the middle "
             "50% of active values, not uncertainty.\n"
             "Bins without active cells have no strength value; hollow markers use fewer "
             "than 30 active cells.\n"
             "States and cells are correlated; this simulation has been explored before.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return start + index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures/feature_relationships")
    parser.add_argument("--state-start", type=int, default=180)
    parser.add_argument("--states", type=int, default=54)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.state_start < 0 or min(args.states, args.bins, args.threads) < 1:
        parser.error("state-start must be nonnegative; counts and threads must be positive")
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data)
    settings, val = data["settings"], data["splits"]["val"]
    end = args.state_start + args.states
    if end > len(val["inputs"]) or settings["history"] != 2:
        raise ValueError("requested validation states must fit and have two input frames")
    model_dir = Path(args.model_dir)
    model, _ = load_emulator(model_dir / "emulator.pt")
    checkpoint = torch.load(model_dir / "sae.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["layer"] != "blocks.6":
        raise ValueError("requires a block 6 SAE")
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()

    inputs = val["inputs"][args.state_start:end]
    mean, std = data["mean"], data["std"]
    raw = inputs * std[None, None, :, None, None] + mean[None, None, :, None, None]
    targets = quantities(raw[:, -1], raw[:, -2], val["raw_targets"][args.state_start:end],
                         settings["F"], settings["k"])
    features = [feature for feature, _ in QUESTIONS]
    batches = []
    n = settings["N"]
    for start in range(0, len(inputs), 4):
        tokens = token_activations(model, inputs[start:start + 4], config["layer"], batch_size=4)
        with torch.no_grad():
            z = sae.encode(tokens)[:, features].numpy()
        batches.append(z.reshape(-1, n, n, len(features)))
    z = np.concatenate(batches)

    output = Path(args.figures)
    output.mkdir(parents=True, exist_ok=True)
    bin_rows = []
    for i, (feature, name) in enumerate(QUESTIONS):
        values, activation = targets[name], z[..., i]
        rows, correlation = bin_statistics(values, activation, bins=args.bins)
        out_path = output / f"feature_{feature:03d}_relationship.png"
        state = plot_relationship(feature, name, values, activation, rows, correlation,
                                  settings["seed"], args.state_start, out_path)
        bin_rows.extend({"feature": feature, "quantity": name, **row} for row in rows)
        print(f"Feature {feature}: map state {state}; wrote {out_path}", flush=True)
    columns = ["feature", "quantity", "low", "high", "centre", "count", "active_count",
               "rate", "median_on", "q25_on", "q75_on"]
    with (output / "bins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(bin_rows)
    print(f"Wrote {output / 'bins.csv'}", flush=True)


if __name__ == "__main__":
    main()
