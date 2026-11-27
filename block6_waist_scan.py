"""Screen all SAE features at candidate-division waists and matched comparison regions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from block6_division_study import CLEAN, classify, components, significant_peaks, trace
from generate_data import DU, DV, advance, integrate, load_dataset, seed_pattern
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


def waist_on_peak_line(field, region, threshold=0.2, prominence=0.03):
    """Approximate the waist by the lowest interior cell on the line between two peaks."""
    peaks = significant_peaks(field, region, threshold, prominence)
    if len(peaks) != 2:
        return None
    shape = np.array(field.shape)
    first = np.array(peaks[0][:2])
    second = np.array(peaks[1][:2])
    delta = (second - first + shape // 2) % shape - shape // 2
    steps = int(np.max(np.abs(delta)))
    if steps < 2:
        return None
    line = np.rint(first + np.linspace(0, 1, steps + 1)[:, None] * delta).astype(int) % shape
    if not region[line[:, 0], line[:, 1]].all():
        return None
    interior = line[1:-1]
    return tuple(int(x) for x in interior[np.argmin(field[interior[:, 0], interior[:, 1]])])


def neighbourhood(region, centre, radius=1):
    """Square neighbourhood clipped to a periodic component."""
    mask = np.zeros_like(region, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            mask[(centre[0] + dy) % region.shape[0], (centre[1] + dx) % region.shape[1]] = True
    return mask & region


def region_summary(regions):
    """Average per-event firing fractions and pool positive strengths within one seed."""
    if not regions or any(len(region) == 0 for region in regions):
        raise ValueError("each event needs a nonempty comparison region")
    rates = np.stack([(region > 0).mean(axis=0) for region in regions])
    all_cells = np.concatenate(regions, axis=0)
    on = all_cells > 0
    strength = np.full(all_cells.shape[1], np.nan)
    for feature in range(all_cells.shape[1]):
        if on[:, feature].any():
            strength[feature] = np.median(all_cells[on[:, feature], feature])
    return {"rate": rates.mean(axis=0), "median": strength,
            "active_cells": on.sum(axis=0), "active_events": (rates > 0).sum(axis=0),
            "cells": len(all_cells), "events": len(regions)}


def select_events(trajectory, segmented, seed_data, threshold, prominence):
    selected = []
    counts = {"strict_candidates": len(seed_data["events"]), "peak_class_pass": 0,
              "waist_found": 0, "valid_within": 0, "matched_control": 0}
    for event in seed_data["events"]:
        t, parent = event["state"], event["parent"]
        labels = segmented[t][0]
        if labels[parent["row"], parent["col"]] != parent["label"]:
            raise ValueError("saved parent component differs from regenerated simulation")
        if classify(trace(trajectory, event, segmented, threshold, prominence, "parent")) != CLEAN:
            continue
        counts["peak_class_pass"] += 1
        region = labels == parent["label"]
        waist = waist_on_peak_line(trajectory[t, 1], region, threshold, prominence)
        if waist is None:
            continue
        counts["waist_found"] += 1
        waist_mask = neighbourhood(region, waist)
        rest_mask = region & ~waist_mask
        if not waist_mask.any() or not rest_mask.any():
            continue
        counts["valid_within"] += 1
        control_mask = None
        control = event["control"]
        if control is not None:
            if labels[control["row"], control["col"]] != control["label"]:
                raise ValueError("saved control component differs from regenerated simulation")
            control_region = labels == control["label"]
            control_mask = neighbourhood(control_region, (control["row"], control["col"]))
            if not control_mask.any():
                raise ValueError("matched control centre has no component cells")
            counts["matched_control"] += 1
        selected.append({"event": event, "waist": waist, "waist_mask": waist_mask,
                         "rest_mask": rest_mask, "control_mask": control_mask})
    return selected, counts


def summarise_seed(selected, encoded, seed, n_latent):
    result = []
    for comparison in ("waist_vs_rest", "waist_vs_control_centre"):
        items = [item for item in selected if comparison == "waist_vs_rest"
                 or item["control_mask"] is not None]
        if not items:
            raise ValueError(f"seed {seed}: no events for {comparison}")
        waist = [encoded[item["event"]["state"]][item["waist_mask"]] for item in items]
        other = [encoded[item["event"]["state"]][item["rest_mask"]
                 if comparison == "waist_vs_rest" else item["control_mask"]] for item in items]
        a, b = region_summary(waist), region_summary(other)
        for f in range(n_latent):
            result.append({"seed": seed, "comparison": comparison, "feature": f,
                           "events": len(items), "waist_cells": a["cells"],
                           "other_cells": b["cells"], "waist_rate": float(a["rate"][f]),
                           "other_rate": float(b["rate"][f]),
                           "difference": float(a["rate"][f] - b["rate"][f]),
                           "waist_active_cells": int(a["active_cells"][f]),
                           "other_active_cells": int(b["active_cells"][f]),
                           "waist_active_events": int(a["active_events"][f]),
                           "other_active_events": int(b["active_events"][f]),
                           "waist_median_on": float(a["median"][f]),
                           "other_median_on": float(b["median"][f])})
    return result


def rank_features(rows, discovery, evaluation, comparison, n_latent):
    by_key = {(row["seed"], row["feature"]): row for row in rows
              if row["comparison"] == comparison}
    output = []
    for f in range(n_latent):
        train = [by_key[(seed, f)] for seed in discovery]
        test = [by_key[(seed, f)] for seed in evaluation]
        output.append({"comparison": comparison, "feature": f,
                       "discovery_difference": float(np.mean([r["difference"] for r in train])),
                       "evaluation_difference": float(np.mean([r["difference"] for r in test])),
                       "evaluation_seed_differences": [r["difference"] for r in test]})
    return output


def figure(rows, ranking, discovery, evaluation, comparison, out):
    ordered = sorted(ranking, key=lambda row: row["discovery_difference"], reverse=True)
    chosen = ordered[:5] + ordered[-3:]
    features = [row["feature"] for row in chosen]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    colours = ("#dd8452", "#4c72b0")
    group_names = ("Waist", "Other parent cells" if comparison == "waist_vs_rest"
                   else "Matched control centre")
    markers = ("o", "s", "^", "D", "P", "X")
    lookup = {(row["seed"], row["feature"]): row for row in rows
              if row["comparison"] == comparison}
    for y, feature in enumerate(features):
        for g, prefix in enumerate(("waist", "other")):
            for marker, seed in zip(markers, evaluation):
                row = lookup[(seed, feature)]
                position = y + (-0.12 if g == 0 else 0.12)
                axes[0].scatter(100 * row[prefix + "_rate"], position, color=colours[g],
                                marker=marker, s=38)
                median = row[prefix + "_median_on"]
                if np.isfinite(median):
                    axes[1].scatter(median, position, color=colours[g], marker=marker, s=38)
    axes[0].set_yticks(range(len(features)), [f"Feature {f}" for f in features])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean % cells on per event")
    axes[1].set_xlabel("Median activation among active cells")
    axes[0].set_xlim(-2, 102)
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    group_handles = [plt.Line2D([0], [0], color=c, marker="o", linestyle="", label=label)
                     for c, label in zip(colours, group_names)]
    seed_handles = [plt.Line2D([0], [0], color="black", marker=marker, linestyle="",
                               label=f"seed {seed}")
                    for marker, seed in zip(markers, evaluation)]
    axes[1].legend(handles=group_handles + seed_handles, fontsize=8, loc="best")
    fig.suptitle("Waist feature screen: evaluation simulation seeds", fontsize=14)
    fig.text(0.5, 0.01,
             f"Selected on discovery seeds {', '.join(map(str, discovery))}: five higher and three "
             "lower waist firing rates.\n"
             "Each point is a seed; rates average events within seed. Strengths pool active cells "
             "within seed and are absent when no cells are on.\n"
             "Candidate divisions and waists are detector-defined; three evaluation seeds are "
             "too few for reliable significance claims.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    source = json.loads(Path(args.source).read_text())
    discovery, evaluation = source["discovery_seeds"], source["evaluation_seeds"]
    if set(discovery) & set(evaluation) or not discovery or not evaluation:
        raise ValueError("discovery and evaluation seeds must be distinct and nonempty")
    data = load_dataset(args.data)
    s = data["settings"]
    if s["history"] != 2 or not (np.isclose(s["Du"], DU) and np.isclose(s["Dv"], DV)):
        raise ValueError("requires history=2 and simulator-default diffusion")
    offset = len(data["splits"]["train"]["inputs"]) + s["gap"] + 60
    if offset != source["simulation_start_calls_after_warmup"] or s["stride"] != source["samples_per_call"]:
        raise ValueError("dataset settings do not match source simulation timing")
    model_dir = Path(args.model_dir)
    model, _ = load_emulator(model_dir / "emulator.pt")
    checkpoint = torch.load(model_dir / "sae.pt", map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if cfg["layer"] != "blocks.6":
        raise ValueError("requires block 6 SAE")
    sae = SparseAutoencoder(cfg["n_input"], cfg["n_latent"], cfg["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    threshold, prominence = source["component_threshold"], 0.03
    all_rows, counts, event_rows = [], {}, []
    for seed in discovery + evaluation:
        seed_data = next((entry for entry in source["per_seed"] if entry["seed"] == seed), None)
        if seed_data is None:
            raise ValueError(f"seed {seed} missing from source")
        print(f"Simulating and screening seed {seed}", flush=True)
        u, v = seed_pattern(s["N"], s["n_spots"], np.random.default_rng(seed))
        u, v = advance(u, v, s["warm_up"] + offset * s["stride"], s["F"], s["k"], s["dt"])
        trajectory = integrate(u, v, (seed_data["state_count"] - 1) * s["stride"], s["F"], s["k"],
                               s["dt"], sample_every=s["stride"])
        segmented = [components(frame[1], threshold) for frame in trajectory]
        selected, seed_counts = select_events(trajectory, segmented, seed_data, threshold, prominence)
        counts[seed] = seed_counts
        needed = sorted({item["event"]["state"] for item in selected})
        encoded = {}
        for start in range(0, len(needed), 4):
            states = needed[start:start + 4]
            raw = np.stack([trajectory[np.array([t - 1, t])] for t in states])
            inputs = ((raw - data["mean"][None, None, :, None, None])
                      / data["std"][None, None, :, None, None]).astype(np.float32)
            tokens = token_activations(model, inputs, cfg["layer"], batch_size=4)
            with torch.no_grad():
                z = sae.encode(tokens).numpy().reshape(len(states), s["N"], s["N"], cfg["n_latent"])
            encoded.update(zip(states, z))
        all_rows.extend(summarise_seed(selected, encoded, seed, cfg["n_latent"]))
        for item in selected:
            event_rows.append({"seed": seed, "state": item["event"]["state"],
                               "waist_row": item["waist"][0], "waist_col": item["waist"][1],
                               "waist_cells": int(item["waist_mask"].sum()),
                               "rest_cells": int(item["rest_mask"].sum()),
                               "control_centre_cells": (int(item["control_mask"].sum())
                                                        if item["control_mask"] is not None else 0)})
        print(f"Seed {seed}: {seed_counts}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_seed_features.csv", all_rows)
    write_csv(out / "events.csv", event_rows)
    rankings = []
    for comparison in ("waist_vs_rest", "waist_vs_control_centre"):
        ranking = rank_features(all_rows, discovery, evaluation, comparison, cfg["n_latent"])
        rankings.extend(ranking)
        figure(all_rows, ranking, discovery, evaluation, comparison, out / f"{comparison}.png")
    write_csv(out / "all_feature_rankings.csv", rankings)
    summary = {"discovery_seeds": discovery, "evaluation_seeds": evaluation,
               "counts": counts, "region": "3x3 neighbourhood around the minimum-v interior "
               "cell on the straight line between exactly two significant parent peaks, "
               "clipped to the parent component",
               "rest": "other cells of the same thresholded component",
               "control": "3x3 around the brightest cell of the pre-existing matched control, "
               "clipped to its component; no detected split over the next three states",
               "selection": "strict candidates with fewer than two significant peaks at t-2, two peaks "
               "at t+1..t+3 and increasing separation; valid waist at t",
               "limitations": "Exploratory; seeds survival-selected and previously explored, "
               "not a fresh blind test. Candidates and waist are detector-defined. "
               "Cells and events are correlated. Three evaluation seeds cannot support "
               "reliable multiple-testing-corrected significance claims."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
