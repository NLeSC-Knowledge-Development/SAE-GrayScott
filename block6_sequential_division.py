"""Reproduce the sequential-division examples and feature-213 triplet comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from block6_division_study import (components, deduplicate_candidates, links,
                                   match_controls, periodic_patch, single_successor)
from generate_data import DU, DV, advance, integrate, load_dataset, seed_pattern
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


SEEDS = (110, 116, 124, 127, 132, 137, 140, 148, 165, 174, 199, 205)
STATES = 256
THRESHOLD = 0.2
HORIZON = 7
FEATURE = 213


def stable_history(label, time, transitions):
    for previous in (time - 1, time - 2):
        outgoing, incoming = transitions[previous]
        parents = incoming.get(label, [])
        if len(parents) != 1 or single_successor(parents[0], outgoing, incoming) != label:
            return False
        label = parents[0]
    return True


def follow_lineage(label, time, transitions):
    """Track exclusive descendants, requiring a first split and seven-state follow-up."""
    lineage = [[label]]
    for step in range(HORIZON):
        outgoing, incoming = transitions[time + step]
        children = []
        for parent in lineage[-1]:
            next_labels = outgoing.get(parent, [])
            if not next_labels or any(len(incoming.get(child, [])) != 1
                                      for child in next_labels):
                return None
            children.extend(next_labels)
        if len(set(children)) != len(children) or len(children) > 3:
            return None
        if step == 0 and len(children) not in (2, 3):
            return None
        lineage.append(children)
    if len(lineage[-1]) == 2:
        return "two", lineage
    if len(lineage[-1]) != 3 or len(lineage[5]) != 3:
        return None
    return ("direct_three" if len(lineage[1]) == 3 else "sequential_three"), lineage


def find_candidates(segmented):
    transitions = [links(a[0], b[0], a[1]) for a, b in zip(segmented, segmented[1:])]
    groups = {"two": [], "direct_three": [], "sequential_three": []}
    for t in range(2, len(segmented) - HORIZON):
        for parent in segmented[t][1]:
            if not stable_history(parent["label"], t, transitions):
                continue
            result = follow_lineage(parent["label"], t, transitions)
            if result is not None:
                kind, lineage = result
                groups[kind].append({"state": t, "parent": parent, "lineage": lineage})
    for kind in groups:
        groups[kind] = deduplicate_candidates(groups[kind], segmented[0][0].shape)
    return groups, transitions


def second_split(event, segmented, transitions):
    """Identify the splitting daughter and its sister just before the second split."""
    lineage = event["lineage"]
    for offset in range(1, 5):
        if len(lineage[offset + 1]) != 3:
            continue
        t = event["state"] + offset
        outgoing, _ = transitions[t]
        parents = lineage[offset]
        splitting = [label for label in parents if len(outgoing[label]) == 2]
        if len(parents) != 2 or len(splitting) != 1:
            raise ValueError("unexpected sequential lineage")
        objects = {obj["label"]: obj for obj in segmented[t][1]}
        sister = next(label for label in parents if label != splitting[0])
        return {"state": t, "parent": objects[splitting[0]], "sister": objects[sister]}
    raise ValueError("second split not found in sequential lineage")


def make_triplets(groups, segmented, transitions):
    targets = [second_split(event, segmented, transitions) for event in groups["sequential_three"]]
    excluded = {(event["state"], event["parent"]["label"]) for event in targets}
    pool = {}
    for event in groups["two"]:
        if (event["state"], event["parent"]["label"]) not in excluded:
            pool.setdefault(event["state"], []).append(event["parent"])
    for event in targets:
        pool.setdefault(event["state"], [])
    return match_controls(targets, pool)


def example_sheet(trajectory, segmented, events, path):
    selected = events[:4]
    if not selected:
        raise ValueError("no sequential candidates found for the example sheet")
    offsets = (-2, -1, 0, 1, 2, 3, 5, 7)
    fig, axes = plt.subplots(len(selected), len(offsets),
                             figsize=(2 * len(offsets), 2.2 * len(selected)), squeeze=False)
    for row, event in enumerate(selected):
        t, parent = event["state"], event["parent"]
        for col, offset in enumerate(offsets):
            ax = axes[row, col]
            ax.imshow(periodic_patch(trajectory[t + offset, 1], parent["row"], parent["col"], 16),
                      cmap="inferno", vmin=0, vmax=0.5)
            ax.scatter([16], [16], marker="+", color="white", s=18)
            if offset >= 0:
                mask = np.isin(segmented[t + offset][0], event["lineage"][offset])
                patch = periodic_patch(mask, parent["row"], parent["col"], 16)
                if patch.any() and not patch.all():
                    ax.contour(patch, levels=[0.5], colors=["cyan"], linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t{offset:+d}" if offset else "t: parent")
        axes[row, 0].set_ylabel(f"seed 116, t={t}", fontsize=9)
    fig.suptitle("Sequential division candidates: cyan outlines track descendants", fontsize=13)
    fig.text(0.5, 0.01,
             "One frame = 100 simulator steps. The white + marks the original parent peak.\n"
             "Threshold-dependent tracking is approximate, not proof of a physical split.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def triplet_figure(rows, path):
    values = np.asarray([[row["sister_mean"], row["daughter_mean"], row["control_mean"]]
                         for row in rows])
    means = values.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in values:
        ax.plot(range(3), row, color="0.6", alpha=0.35, linewidth=1)
        ax.scatter(range(3), row, color="0.4", alpha=0.45, s=22)
    ax.plot(range(3), means, color="#bd592e", linewidth=2, marker="D", markersize=8,
            label="Mean across triplets")
    for x, mean in enumerate(means):
        ax.annotate(f"{mean:.3f}", (x, mean), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(3), ["Sister\n(not splitting now)", "Daughter\n(second split)",
                             "Control blob\n(ordinary split)"])
    ax.set_ylabel("Mean feature activation across blob cells")
    ax.set_ylim(bottom=0, top=max(values.max(), means.max()) * 1.18 or 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")
    ax.set_title(f"Feature {FEATURE}: three blobs at the same time\n"
                 f"{len(rows)} matched triplets from evaluation seeds", fontsize=13)
    fig.text(0.5, 0.02,
             "Each grey line connects three different blobs in the same snapshot. "
             "Values average all cells inside each blob, including zeros.\n"
             "Measured immediately before the daughter splits; not on the products. "
             "Samples are correlated and do not establish causality.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(path, dpi=170)
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
    rows, counts = [], []
    for seed in SEEDS:
        print(f"Simulating seed {seed}", flush=True)
        u, v = seed_pattern(settings["N"], settings["n_spots"], np.random.default_rng(seed))
        u, v = advance(u, v, settings["warm_up"] + offset * settings["stride"],
                       settings["F"], settings["k"], settings["dt"])
        trajectory = integrate(u, v, (STATES - 1) * settings["stride"],
                               settings["F"], settings["k"], settings["dt"],
                               sample_every=settings["stride"])
        segmented = [components(frame[1], THRESHOLD) for frame in trajectory]
        groups, transitions = find_candidates(segmented)
        if seed == 116:
            example_sheet(trajectory, segmented, groups["sequential_three"],
                          out / "sequential_division_examples.png")
        targets = make_triplets(groups, segmented, transitions)
        matched = [item for item in targets if item["control"] is not None]
        counts.append({"seed": seed, **{kind: len(events) for kind, events in groups.items()},
                       "matched_triplets": len(matched)})
        states = sorted({item["state"] for item in matched})
        encoded = {}
        for start in range(0, len(states), 4):
            batch = states[start:start + 4]
            raw = np.stack([trajectory[[t - 1, t]] for t in batch])
            standard = ((raw - data["mean"][None, None, :, None, None])
                        / data["std"][None, None, :, None, None]).astype(np.float32)
            tokens = token_activations(model, standard, config["layer"], batch_size=4)
            with torch.no_grad():
                z = sae.encode(tokens).numpy().reshape(len(batch), settings["N"],
                                                      settings["N"], config["n_latent"])
            encoded.update(zip(batch, z))
        for item in matched:
            t = item["state"]
            labels = segmented[t][0]
            objects = (item["sister"], item["parent"], item["control"])
            if len({obj["label"] for obj in objects}) != 3:
                raise ValueError("matched triplet contains repeated components")
            means = []
            for obj in objects:
                cells = encoded[t][:, :, FEATURE][labels == obj["label"]]
                if not len(cells):
                    raise ValueError(f"empty component for seed {seed}, state {t}")
                means.append(float(cells.mean()))
            rows.append({"seed": seed, "state": t, "feature": FEATURE,
                         "sister_label": objects[0]["label"], "daughter_label": objects[1]["label"],
                         "control_label": objects[2]["label"], "sister_mean": means[0],
                         "daughter_mean": means[1], "control_mean": means[2]})
        print(f"  {counts[-1]}", flush=True)
    if not rows:
        raise ValueError("no matched second-split triplets found")
    with (out / "feature_213_triplets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"seeds": counts}, indent=2) + "\n")
    triplet_figure(rows, out / "feature_213_triplets.png")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
