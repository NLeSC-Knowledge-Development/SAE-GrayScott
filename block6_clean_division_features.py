"""Block-6 SAE feature ranking on peak-filtered division candidates versus controls.

Peak filtering requires fewer than two significant v peaks at t-2 and two at t+1..t+3
with growing separation. Features are ranked on discovery seeds and checked on separate
evaluation seeds, with activation traced from t-2 to t+3.
Detector-defined events; associations only.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from block6_division_study import (CLEAN, OFFSETS, classify, components,
                                   deduplicate_candidates, dilate, division_candidates,
                                   match_controls, trace)
from generate_data import DU, DV, advance, integrate, load_dataset, seed_pattern
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


def region_mask(segmented, t, offset, parent):
    region_t = dilate(segmented[t][0] == parent["label"], 2)
    labels = segmented[t + offset][0]
    touching = np.unique(labels[region_t])
    return np.isin(labels, touching[touching > 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--states", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--prominence", type=float, default=0.03)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data)
    s = data["settings"]
    if s["history"] != 2 or not (np.isclose(s["Du"], DU) and np.isclose(s["Dv"], DV)):
        raise ValueError("requires history=2 and simulator-default diffusion")
    model_dir = Path(args.model_dir)
    model, _ = load_emulator(model_dir / "emulator.pt")
    ckpt = torch.load(model_dir / "sae.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["layer"] != "blocks.6":
        raise ValueError("requires block 6 SAE")
    sae = SparseAutoencoder(cfg["n_input"], cfg["n_latent"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    rates = np.asarray(ckpt["metrics"]["firing_rate"])
    mean, std = data["mean"], data["std"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offset = len(data["splits"]["train"]["inputs"]) + s["gap"] + 60
    records, counts = [], {}
    for seed in args.discovery_seeds + args.evaluation_seeds:
        u, v = seed_pattern(s["N"], s["n_spots"], np.random.default_rng(seed))
        u, v = advance(u, v, s["warm_up"] + offset * s["stride"], s["F"], s["k"], s["dt"])
        traj = integrate(u, v, (args.states + 5) * s["stride"], s["F"], s["k"], s["dt"],
                         sample_every=s["stride"])
        segmented = [components(f[1], args.threshold) for f in traj]
        events, pool = division_candidates(segmented)
        events = deduplicate_candidates(events, traj.shape[-2:])
        clean = [e for e in events
                 if classify(trace(traj, e, segmented, args.threshold, args.prominence,
                                   "parent")) == CLEAN]
        match_controls(clean, pool)
        clean = [e for e in clean if e["control"] is not None]
        counts[seed] = {"candidates": len(events), "clean_matched": len(clean)}
        needed = sorted({e["state"] + o for e in clean for o in OFFSETS})
        z = {}
        for start in range(0, len(needed), 4):
            states = needed[start:start + 4]
            raw = np.stack([traj[np.array([t - 1, t])] for t in states])
            standard = ((raw - mean[None, None, :, None, None])
                        / std[None, None, :, None, None]).astype(np.float32)
            tokens = token_activations(model, standard, cfg["layer"], batch_size=4)
            with torch.no_grad():
                code = sae.encode(tokens).numpy().reshape(len(states), s["N"], s["N"],
                                                          cfg["n_latent"])
            z.update(zip(states, code))
        for e in clean:
            t = e["state"]
            cand, ctrl = [], []
            for o in OFFSETS:
                cand.append(z[t + o][region_mask(segmented, t, o, e["parent"])].mean(0))
                ctrl.append(z[t + o][region_mask(segmented, t, o, e["control"])].mean(0))
            records.append({"seed": seed, "candidate": np.stack(cand), "control": np.stack(ctrl)})
        print(f"seed {seed}: {counts[seed]}", flush=True)

    train = [r for r in records if r["seed"] in args.discovery_seeds]
    test = [r for r in records if r["seed"] in args.evaluation_seeds]
    at = OFFSETS.index(0)
    a = np.stack([r["candidate"][at] for r in train])
    b = np.stack([r["control"][at] for r in train])
    scale = np.std(np.concatenate((a, b)), axis=0)
    score = np.divide((a - b).mean(0), scale, out=np.zeros_like(scale), where=scale > 1e-8)
    order = np.argsort(-score)
    low = np.argsort(score)
    rare = [int(i) for i in order if 0 < rates[i] <= 0.01]
    selected = list(dict.fromkeys([int(i) for i in order[:8]] + rare[:4]
                                  + [int(i) for i in low[:4]]))
    rows = []
    for f in selected:
        d = np.array([r["candidate"][at][f] - r["control"][at][f] for r in test])
        per_seed = {sd: float(np.mean([r["candidate"][at][f] > r["control"][at][f]
                                       for r in test if r["seed"] == sd]))
                    for sd in args.evaluation_seeds}
        rows.append({"feature": f, "training_firing_fraction": float(rates[f]),
                     "discovery_standardized_difference": float(score[f]),
                     "evaluation_mean_difference": float(d.mean()),
                     "evaluation_fraction_candidate_higher": float((d > 0).mean()),
                     "per_seed_fraction_higher": per_seed})
    summary = {"counts": counts, "discovery_pairs": len(train), "evaluation_pairs": len(test),
               "offsets": OFFSETS, "features": rows,
               "note": "Ranking at offset 0 on discovery seeds; checked on separate evaluation "
                       "seeds previously explored. Peak-filtered candidates; associations only."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False)
    for ax, row in zip(axes.ravel(), rows[:12]):
        f = row["feature"]
        for name, color in (("candidate", "tab:red"), ("control", "tab:blue")):
            tr = np.stack([r[name][:, f] for r in test])
            err = tr.std(0) / np.sqrt(len(tr))
            ax.plot(OFFSETS, tr.mean(0), "o-", color=color, label=name)
            ax.fill_between(OFFSETS, tr.mean(0) - err, tr.mean(0) + err, color=color, alpha=0.2)
        ax.set_title(f"feature {f} (rate {row['training_firing_fraction']:.1%}, "
                     f"higher in {row['evaluation_fraction_candidate_higher']:.0%} pairs)",
                     fontsize=8)
        ax.set_xlabel("states relative to parent t")
        ax.set_ylabel("mean activation in blob")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Evaluation seeds ({len(test)} pairs): division candidates vs matched controls; "
                 "bands = standard error over pairs (not independent)")
    fig.tight_layout()
    fig.savefig(out / "clean_division_feature_timecourse.png", dpi=130)
    plt.close(fig)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
