"""Reproduce two-peak examples, matched features and an all-feature rarity screen."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from block6_division_study import (components, deduplicate_candidates, links, periodic_distance,
                                   periodic_patch, significant_peaks, single_successor)
from block6_waist_scan import neighbourhood, waist_on_peak_line
from generate_data import DU, DV, advance, integrate, load_dataset, seed_pattern
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


SEEDS = (217, 228, 235, 246, 251, 252, 279, 296, 297, 304, 308, 311)
STATES, INTERVAL, HORIZON = 256, 10, 7
THRESHOLD, PROMINENCE = 0.2, 0.03
FEATURES = (81, 495, 307, 72, 427)
OUTCOMES = ("persistent", "split", "changed", "ambiguous_split", "lost", "merge_or_ambiguous")


def outcome(label, time, transitions, peak_count):
    """Classify a two-peak component through seven states."""
    track = []
    for step in range(HORIZON):
        outgoing, incoming = transitions[time + step]
        children = outgoing.get(label, [])
        if not children:
            return "lost"
        if any(len(incoming.get(child, [])) != 1 for child in children):
            return "merge_or_ambiguous"
        if len(children) > 1:
            if len(children) != 2 or step > HORIZON - 3:
                return "ambiguous_split"
            for follow in (1, 2):
                out, inc = transitions[time + step + follow]
                children = [single_successor(child, out, inc) for child in children]
                if None in children or len(set(children)) != 2:
                    return "ambiguous_split"
            return "split"
        label = children[0]
        track.append((time + step + 1, label))
    return "persistent" if all(peak_count(t, label) == 2 for t, label in track) else "changed"


def match_events(splits, persistent, shape):
    """Match within a seed, preferring the same state and not reusing controls."""
    used, matches = set(), []
    for candidate in splits:
        parent = candidate["parent"]
        eligible = []
        for index, control in enumerate(persistent):
            if index in used:
                continue
            other = control["parent"]
            difference = abs(candidate["state"] - control["state"])
            distance = periodic_distance((parent["row"], parent["col"]),
                                         (other["row"], other["col"]), shape)
            if difference > 30 or (difference <= HORIZON and distance <= 7):
                continue
            area = abs(np.log(parent["area"] / other["area"]))
            elongation = abs(np.log(parent["elongation"] / other["elongation"]))
            brightness = abs(parent["peak"] - other["peak"])
            separation = abs(np.log(candidate["peak_separation"] / control["peak_separation"]))
            if (area > np.log(1.75) or elongation > np.log(1.5) or brightness > 0.1
                    or separation > np.log(1.5)):
                continue
            score = ((area / np.log(1.75)) ** 2 + (elongation / np.log(1.5)) ** 2
                     + (brightness / 0.1) ** 2 + (separation / np.log(1.5)) ** 2
                     + 0.25 * difference / 30)
            eligible.append((difference != 0, score, index))
        if eligible:
            _, score, index = min(eligible)
            used.add(index)
            matches.append({"split": candidate, "persistent": persistent[index],
                            "match_score": float(score)})
    return matches


def measure_state(encoded, observations):
    """Count active features by waist outcome and in the remaining grid cells."""
    n = encoded.shape[-1]
    active_observations = {kind: np.zeros(n, dtype=np.int64) for kind in OUTCOMES}
    active_cells = {kind: np.zeros(n, dtype=np.int64) for kind in OUTCOMES}
    event_counts = dict.fromkeys(OUTCOMES, 0)
    persistent_mask = np.zeros(encoded.shape[:2], dtype=bool)
    for kind, mask in observations:
        values = encoded[mask]
        if not len(values):
            raise ValueError("empty waist region")
        on = values > 0
        active_observations[kind] += on.any(axis=0)
        active_cells[kind] += on.sum(axis=0)
        event_counts[kind] += 1
        if kind == "persistent":
            persistent_mask |= mask
    background = encoded[~persistent_mask] > 0
    return active_observations, active_cells, event_counts, background.sum(axis=0), len(background)


def example_figure(examples, path):
    offsets = (0, 1, 3, 7)
    fig, axes = plt.subplots(len(examples) * 2, len(offsets), figsize=(9, 2 * len(examples) * 2),
                             squeeze=False)
    for pair_index, pair in enumerate(examples):
        for group_index, kind in enumerate(("persistent", "split")):
            item = pair[kind]
            row = pair_index * 2 + group_index
            for col, offset in enumerate(offsets):
                ax = axes[row, col]
                ax.imshow(item["patches"][col], cmap="inferno", vmin=0, vmax=0.5)
                if offset == 0:
                    mask = item["mask"]
                    if mask.any() and not mask.all():
                        ax.contour(mask, levels=[0.5], colors=["cyan"], linewidths=0.7)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"t+{offset}")
            label = "Two peaks" if kind == "persistent" else "Later split"
            axes[row, 0].set_ylabel(f"Pair {pair_index + 1}: {label}\nseed {pair['seed']}, "
                                    f"state {item['state']}", fontsize=9)
    fig.suptitle("Matched two-peak examples", fontsize=14)
    fig.text(0.5, 0.01,
             "Each row is a fixed patch; one state = 100 simulator steps. "
             "Cyan outlines the initial component.\n"
             "Pairs are from the same seed, but usually not the same time. "
             "Two-peak and split labels depend on the detector.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data)
    settings = data["settings"]
    offset = len(data["splits"]["train"]["inputs"]) + settings["gap"] + 60
    expected = {"N": 64, "stride": 100, "history": 2, "seed": 0, "n_samples": 1600,
                "n_spots": 25, "warm_up": 2000, "gap": 20, "F": 0.022, "k": 0.059,
                "Du": DU, "Dv": DV, "dt": 1.0}
    if offset != 1172 or any(not np.isclose(settings[key], value)
                             for key, value in expected.items()):
        raise ValueError("requires the default seed-0 dataset and simulator settings")
    model_dir = Path(args.model_dir)
    model, _ = load_emulator(model_dir / "emulator.pt")
    checkpoint = torch.load(model_dir / "sae.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["layer"] != "blocks.6" or config["n_latent"] != 512 or config["k"] != 16:
        raise ValueError("requires the 512-feature TopK-16 block-6 SAE")
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, counts, examples = [], [], []
    n = config["n_latent"]
    active_observations = {kind: np.zeros(n, dtype=np.int64) for kind in OUTCOMES}
    active_cells = {kind: np.zeros(n, dtype=np.int64) for kind in OUTCOMES}
    waist_counts = dict.fromkeys(OUTCOMES, 0)
    grid_active = np.zeros(n, dtype=np.int64)
    grid_cells = sampled_states = 0
    for seed in SEEDS:
        print(f"Simulating seed {seed}", flush=True)
        u, v = seed_pattern(settings["N"], settings["n_spots"], np.random.default_rng(seed))
        u, v = advance(u, v, settings["warm_up"] + offset * settings["stride"],
                       settings["F"], settings["k"], settings["dt"])
        trajectory = integrate(u, v, (STATES - 1) * settings["stride"], settings["F"],
                               settings["k"], settings["dt"], sample_every=settings["stride"])
        segmented = [components(frame[1], THRESHOLD) for frame in trajectory]
        transitions = [links(a[0], b[0], a[1]) for a, b in zip(segmented, segmented[1:])]
        cache = {}

        def peaks(t, label):
            key = t, label
            if key not in cache:
                cache[key] = significant_peaks(trajectory[t, 1], segmented[t][0] == label,
                                               THRESHOLD, PROMINENCE)
            return cache[key]

        observations, categories = [], {}
        for t in range(INTERVAL, len(trajectory) - HORIZON, INTERVAL):
            labels, objects = segmented[t]
            for obj in objects:
                points = peaks(t, obj["label"])
                if len(points) != 2:
                    continue
                fate = outcome(obj["label"], t, transitions,
                               lambda state, label: len(peaks(state, label)))
                categories[fate] = categories.get(fate, 0) + 1
                waist = waist_on_peak_line(trajectory[t, 1], labels == obj["label"],
                                            THRESHOLD, PROMINENCE)
                observations.append({"state": t, "parent": obj, "outcome": fate, "waist": waist,
                                     "peak_separation": periodic_distance(
                                         points[0][:2], points[1][:2], labels.shape)})
        split = deduplicate_candidates([x for x in observations
                                        if x["outcome"] == "split" and x["waist"] is not None],
                                       trajectory.shape[-2:], time_radius=7)
        persistent = deduplicate_candidates([x for x in observations
                                             if x["outcome"] == "persistent" and x["waist"] is not None],
                                            trajectory.shape[-2:], time_radius=7)
        matches = match_events(split, persistent, trajectory.shape[-2:])
        match_states = {pair[kind]["state"] for pair in matches for kind in ("split", "persistent")}
        needed = sorted({item["state"] for item in observations})
        encoded = {}
        for start in range(0, len(needed), 4):
            batch = needed[start:start + 4]
            raw = np.stack([trajectory[[t - 1, t]] for t in batch])
            standard = ((raw - data["mean"][None, None, :, None, None])
                        / data["std"][None, None, :, None, None]).astype(np.float32)
            tokens = token_activations(model, standard, config["layer"], batch_size=4)
            with torch.no_grad():
                z = sae.encode(tokens).numpy().reshape(len(batch), settings["N"], settings["N"],
                                                      config["n_latent"])
            for index, t in enumerate(batch):
                if t in match_states:
                    encoded[t] = z[index]
                state_waists = []
                for item in (item for item in observations
                             if item["state"] == t and item["waist"] is not None):
                    region = segmented[t][0] == item["parent"]["label"]
                    state_waists.append((item["outcome"], neighbourhood(region, item["waist"])))
                seen, cells, events, background, background_size = measure_state(z[index], state_waists)
                for kind in OUTCOMES:
                    active_observations[kind] += seen[kind]
                    active_cells[kind] += cells[kind]
                    waist_counts[kind] += events[kind]
                grid_active += background
                grid_cells += background_size
                sampled_states += 1
        for pair in matches:
            row = {"seed": seed, "match_score": pair["match_score"]}
            example = {"seed": seed}
            for kind in ("persistent", "split"):
                item, t = pair[kind], pair[kind]["state"]
                obj = item["parent"]
                region = segmented[t][0] == obj["label"]
                waist = neighbourhood(region, item["waist"])
                if not waist.any():
                    raise ValueError("empty waist region")
                values = encoded[t][waist][:, FEATURES]
                row.update({kind + "_state": t, kind + "_label": obj["label"]})
                for index, feature in enumerate(FEATURES):
                    row[f"{kind}_{feature}_mean"] = float(values[:, index].mean())
                    row[f"{kind}_{feature}_rate"] = float((values[:, index] > 0).mean())
                example[kind] = {"state": t,
                                 "patches": [periodic_patch(trajectory[t + dt, 1], obj["row"],
                                                            obj["col"], 12) for dt in (0, 1, 3, 7)],
                                 "mask": periodic_patch(region, obj["row"], obj["col"], 12)}
            rows.append(row)
            examples.append(example)
        result = {"seed": seed, "outcomes": categories, "matched": len(matches)}
        counts.append(result)
        print(json.dumps(result), flush=True)
    if not rows:
        raise ValueError("no matched two-peak pairs found")
    with (out / "matched_features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    specificity = []
    for feature in range(n):
        row = {"feature": feature}
        for kind in OUTCOMES:
            row[kind + "_active_observations"] = int(active_observations[kind][feature])
            row[kind + "_active_cells"] = int(active_cells[kind][feature])
        row["other_grid_active_cells"] = int(grid_active[feature])
        row["other_grid_firing_rate"] = float(grid_active[feature] / grid_cells)
        specificity.append(row)
    with (out / "all_feature_specificity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(specificity[0]))
        writer.writeheader()
        writer.writerows(specificity)
    nonpersistent = [kind for kind in OUTCOMES if kind != "persistent"]
    exclusive = [row["feature"] for row in specificity
                 if row["persistent_active_observations"] > 0
                 and all(row[kind + "_active_observations"] == 0 for kind in nonpersistent)]
    summary = {"seeds": counts, "matched_pairs": len(rows),
               "same_state_pairs": sum(r["persistent_state"] == r["split_state"] for r in rows),
               "sampled_states": sampled_states, "background_grid_cells": grid_cells,
               "waist_observations": waist_counts,
               "features_active_in_any_persistent_waist": sum(
                   row["persistent_active_observations"] > 0 for row in specificity),
               "features_exclusive_to_persistent_waists_among_sampled_waists": exclusive,
               "features_exclusive_to_persistent_waists_across_sampled_grids": [
                   row["feature"] for row in specificity
                   if row["feature"] in exclusive and row["other_grid_active_cells"] == 0]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    example_figure(examples, out / "dumbbell_examples.png")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
