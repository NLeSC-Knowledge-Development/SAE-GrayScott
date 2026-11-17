"""Read continuous physical quantities from SAE features and emulator tokens."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import (DU, DV, SPOT_THRESHOLD, advance, integrate, laplacian,
                           load_dataset, seed_pattern, spot_centres)
from interpret import box_mean
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


RIDGE = 1e-3
BATCH = 4096


def quantities(current, previous, following, feed, kill):
    """One target per cell; centre distances are undefined if too few are detected."""
    u, v = current[:, 0], current[:, 1]
    dx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / 2
    dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / 2
    distances = np.full((len(v), 2, *v.shape[1:]), np.nan)
    rows, cols = np.indices(v.shape[1:])
    for i, field in enumerate(v):
        centres = spot_centres(field)
        if centres:
            all_distances = []
            for row, col in centres:
                dr = np.minimum((rows - row) % field.shape[0], (row - rows) % field.shape[0])
                dc = np.minimum((cols - col) % field.shape[1], (col - cols) % field.shape[1])
                all_distances.append(np.hypot(dr, dc))
            ordered = np.sort(all_distances, axis=0)
            distances[i, :min(2, len(ordered))] = ordered[:2]
    reaction = u * v * v
    return {
        "Local u": u,
        "Local v": v,
        "Reaction (u v²)": reaction,
        "Reaction - drain (no diffusion)": reaction - (feed + kill) * v,
        "V Laplacian": np.stack([laplacian(field) for field in v]),
        "Horizontal v gradient": dx,
        "Vertical v gradient": dy,
        "V gradient magnitude": np.hypot(dx, dy),
        "Mean v in 11×11 patch": np.stack([box_mean(field, 5) for field in v]),
        "Spot coverage in 11×11 patch": np.stack(
            [box_mean((field > SPOT_THRESHOLD).astype(np.float64), 5) for field in v]),
        "Distance to nearest centre": distances[:, 0],
        "Distance to second centre": distances[:, 1],
        "Previous v change": v - previous[:, 1],
        "Next true v change": following[:, 1] - v,
        "Grid-mean v (control)": np.broadcast_to(v.mean(axis=(1, 2))[:, None, None], v.shape),
    }


def fit_probes(features, tokens, targets):
    """Fit on discovery cells, grouping targets with the same finite-cell mask."""
    y = np.column_stack([value.reshape(-1) for value in targets.values()])
    if len(features) != len(y) or len(tokens) != len(y):
        raise ValueError("features, tokens and physical quantities must have one row per cell")
    masks = np.isfinite(y)
    groups = {}
    for index in range(y.shape[1]):
        groups.setdefault(masks[:, index].tobytes(), []).append(index)
    fitted = []
    for indices in groups.values():
        mask = masks[:, indices[0]]
        n = int(mask.sum())
        if n < 2:
            raise ValueError(f"not enough discovery cells for {list(targets)[indices[0]]}")
        values = y[mask][:, indices].astype(np.float64)
        y_mean = values.mean(axis=0)
        centered_y = values - y_mean
        models = []
        for data in (features, tokens):
            size = data.shape[1]
            sums = np.zeros(size)
            gram = np.zeros((size, size))
            cross = np.zeros((size, len(indices)))
            valid_indices = np.flatnonzero(mask)
            for start in range(0, n, BATCH):
                chosen = valid_indices[start:start + BATCH]
                x = np.asarray(data[chosen], dtype=np.float64)
                sums += x.sum(axis=0)
                gram += x.T @ x
                cross += x.T @ centered_y[start:start + len(x)]
            mean = sums / n
            centered_gram = gram - n * np.outer(mean, mean)
            variance = np.maximum(np.diag(centered_gram), 0)
            scale = np.sqrt(variance / n)
            scale[scale == 0] = 1
            normal_gram = centered_gram / np.outer(scale, scale)
            normal_gram.flat[::size + 1] += RIDGE
            coefficients = np.linalg.solve(normal_gram, cross / scale[:, None]) / scale[:, None]
            models.append((mean, coefficients))
            if data is features:
                ranking = np.zeros_like(cross)
                np.divide(cross**2, variance[:, None], out=ranking,
                          where=variance[:, None] > 0)
                selected = np.argmax(ranking, axis=0)
                if np.any(variance[selected] <= 0):
                    raise ValueError("no varying SAE feature for a physical quantity")
                slope = cross[selected, np.arange(len(indices))] / (
                    variance[selected] * (1 + RIDGE / n))
        fitted.append({"indices": indices, "y_mean": y_mean, "models": models,
                       "selected": selected, "slope": slope})
    return fitted


def score_probes(features, tokens, targets, fitted):
    """Return squared errors and target moments for exact per-seed and pooled R²."""
    y = np.column_stack([value.reshape(-1) for value in targets.values()])
    if len(features) != len(y) or len(tokens) != len(y):
        raise ValueError("features, tokens and physical quantities must have one row per cell")
    results = [None] * y.shape[1]
    for group in fitted:
        evaluation_masks = {}
        for local_index, index in enumerate(group["indices"]):
            mask = np.isfinite(y[:, index])
            evaluation_masks.setdefault(mask.tobytes(), []).append(local_index)
        for local_indices in evaluation_masks.values():
            score_group(features, tokens, y, group, local_indices, targets, results)
    return results


def score_group(features, tokens, y, group, local_indices, targets, results):
    indices = [group["indices"][j] for j in local_indices]
    mask = np.isfinite(y[:, indices[0]])
    n = int(mask.sum())
    if n < 2:
        raise ValueError(f"not enough evaluation cells for {list(targets)[indices[0]]}")
    actual = y[mask][:, indices].astype(np.float64)
    errors = np.zeros((3, len(indices)))
    valid_indices = np.flatnonzero(mask)
    for start in range(0, n, BATCH):
        chosen = valid_indices[start:start + BATCH]
        z = np.asarray(features[chosen], dtype=np.float64)
        x = np.asarray(tokens[chosen], dtype=np.float64)
        z_mean, z_coef = group["models"][0]
        x_mean, x_coef = group["models"][1]
        selected = group["selected"][local_indices]
        y_mean = group["y_mean"][local_indices]
        predictions = [
            y_mean + (z[:, selected] - z_mean[selected]) * group["slope"][local_indices],
            y_mean + (z - z_mean) @ z_coef[:, local_indices],
            y_mean + (x - x_mean) @ x_coef[:, local_indices],
        ]
        observed = actual[start:start + len(chosen)]
        for i, prediction in enumerate(predictions):
            errors[i] += np.square(observed - prediction).sum(axis=0)
    for j, index in enumerate(indices):
        column = actual[:, j]
        results[index] = {"n": n, "sum": float(column.sum()),
                          "sum_squares": float(column @ column),
                          "sse": errors[:, j].tolist(),
                          "feature": int(group["selected"][local_indices[j]])}


def r2(result):
    total = result["sum_squares"] - result["sum"] ** 2 / result["n"]
    if total <= 0:
        raise ValueError("evaluation target has zero variance; R² is undefined")
    return [float(1 - value / total) for value in result["sse"]]


def figure(names, rows, out_path):
    fig, axis = plt.subplots(figsize=(11, 8))
    positions = np.arange(len(names))
    width = 0.25
    for j, (label, colour) in enumerate((
            ("Best single SAE feature", "#c44e52"),
            ("All SAE features", "#dd8452"),
            ("Original layer", "#4c72b0"))):
        axis.barh(positions + (j - 1) * width, [row["pooled_r2"][j] for row in rows],
                  height=width, color=colour, label=label)
    axis.set_yticks(positions, [f"{name}  (#{row['feature']})"
                                for name, row in zip(names, rows)], fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_xlabel("R² on independent simulator seeds (1 = perfect; 0 = mean baseline)")
    axis.set_title("Physical quantities in block 6: linear readouts")
    axis.legend(loc="lower right", fontsize=9)
    fig.text(0.5, 0.01, "Feature number = best single feature chosen on discovery states.\n"
             "All readouts fit on discovery states; bars pool cells from three evaluation seeds.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--seeds", type=int, nargs="+", default=[9, 14, 35])
    parser.add_argument("--states", type=int, default=150)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.states < 2 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use at least two states, positive threads and distinct seeds")
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data)
    settings = data["settings"]
    if settings["history"] != 2 or not (np.isclose(settings["Du"], DU)
                                         and np.isclose(settings["Dv"], DV)):
        raise ValueError("requires history=2 and simulator-default diffusion coefficients")
    if settings["seed"] in args.seeds:
        parser.error("evaluation seeds must differ from the dataset seed")
    val = data["splits"]["val"]
    if len(val["inputs"]) < 120:
        raise ValueError("requires at least 120 validation states")
    model_dir = Path(args.model_dir)
    model, _ = load_emulator(model_dir / "emulator.pt")
    checkpoint = torch.load(model_dir / "sae.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["layer"] != "blocks.6":
        raise ValueError("this figure requires an SAE trained on blocks.6")
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    mean, std = data["mean"], data["std"]

    def encode(inputs):
        tokens = token_activations(model, inputs, config["layer"], batch_size=4)
        with torch.no_grad():
            features = sae.encode(tokens)
        return features.numpy(), tokens.numpy()

    print("Fitting probes on validation states 60:120", flush=True)
    inputs = val["inputs"][60:120]
    raw = inputs * std[None, None, :, None, None] + mean[None, None, :, None, None]
    targets = quantities(raw[:, -1], raw[:, -2], val["raw_targets"][60:120],
                         settings["F"], settings["k"])
    features, tokens = encode(inputs)
    fitted = fit_probes(features, tokens, targets)
    names = list(targets)
    del features, tokens, raw, targets
    per_seed = []
    offset = len(data["splits"]["train"]["inputs"]) + settings["gap"] + 60
    for seed in args.seeds:
        print(f"Simulating and evaluating seed {seed}", flush=True)
        u, v = seed_pattern(settings["N"], settings["n_spots"], np.random.default_rng(seed))
        u, v = advance(u, v, settings["warm_up"] + offset * settings["stride"],
                       settings["F"], settings["k"], settings["dt"])
        trajectory = integrate(u, v, (args.states + 1) * settings["stride"], settings["F"],
                               settings["k"], settings["dt"], sample_every=settings["stride"])
        raw_inputs = np.stack([trajectory[:-2], trajectory[1:-1]], axis=1)
        inputs = ((raw_inputs - mean[None, None, :, None, None])
                  / std[None, None, :, None, None]).astype(np.float32)
        targets = quantities(trajectory[1:-1], trajectory[:-2], trajectory[2:],
                             settings["F"], settings["k"])
        features, tokens = encode(inputs)
        per_seed.append(score_probes(features, tokens, targets, fitted))
        del features, tokens, inputs, raw_inputs, trajectory, targets
    rows = []
    for index, name in enumerate(names):
        entries = [seed_results[index] for seed_results in per_seed]
        pooled = {"n": sum(entry["n"] for entry in entries),
                  "sum": sum(entry["sum"] for entry in entries),
                  "sum_squares": sum(entry["sum_squares"] for entry in entries),
                  "sse": np.sum([entry["sse"] for entry in entries], axis=0).tolist()}
        row = {"quantity": name, "feature": entries[0]["feature"], "pooled_r2": r2(pooled),
               "per_seed": [{"seed": seed, "cells": entry["n"], "r2": r2(entry)}
                            for seed, entry in zip(args.seeds, entries)]}
        rows.append(row)
        print(f"{name:32} #{row['feature']:3}  " +
              "  ".join(f"{score:.3f}" for score in row["pooled_r2"]), flush=True)
    results = {"layer": config["layer"], "discovery_validation_states": [60, 120],
               "evaluation_seeds": args.seeds, "states_per_seed": args.states,
               "seed_start_calls_after_warmup": offset, "rows": rows}
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "physical_probes.json").write_text(json.dumps(results, indent=2) + "\n")
    figure(names, rows, Path(args.figures) / "physical_questions.png")
    print(f"Wrote {model_dir / 'physical_probes.json'} and "
          f"{Path(args.figures) / 'physical_questions.png'}")


if __name__ == "__main__":
    main()
