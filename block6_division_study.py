"""Simulate and screen natural blob-division candidates and matched controls.

Connected components, overlap and short-term persistence are operational proxies,
not verified physical object tracking. Results require visual validation.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_data import DU, DV, advance, integrate, load_dataset, seed_pattern


OFFSETS = (-2, -1, 0, 1, 2, 3)
CLEAN = "single peak at t-2, two peaks later"


def periodic_patch(field, row, col, radius):
    offsets = np.arange(-radius, radius + 1)
    return field[np.ix_((row + offsets) % field.shape[0],
                        (col + offsets) % field.shape[1])]


def periodic_distance(a, b, shape):
    dy = min((a[0] - b[0]) % shape[0], (b[0] - a[0]) % shape[0])
    dx = min((a[1] - b[1]) % shape[1], (b[1] - a[1]) % shape[1])
    return float(np.hypot(dy, dx))


def components(field, threshold, minimum_area=4):
    """Four-connected components on a periodic grid, with their brightest cell."""
    mask = field > threshold
    labels = np.zeros(mask.shape, dtype=np.int32)
    n, m = mask.shape
    objects = []
    for row, col in np.argwhere(mask):
        if labels[row, col]:
            continue
        label = len(objects) + 1
        stack = [(int(row), int(col))]
        labels[row, col] = label
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in (((r - 1) % n, c), ((r + 1) % n, c),
                           (r, (c - 1) % m), (r, (c + 1) % m)):
                if mask[rr, cc] and labels[rr, cc] == 0:
                    labels[rr, cc] = label
                    stack.append((rr, cc))
        if len(cells) < minimum_area:
            for r, c in cells:
                labels[r, c] = -1
            continue
        coords = np.asarray(cells)
        brightest = coords[np.argmax(field[coords[:, 0], coords[:, 1]])]
        dr = (coords[:, 0] - brightest[0] + n // 2) % n - n // 2
        dc = (coords[:, 1] - brightest[1] + m // 2) % m - m // 2
        covariance = np.cov(np.stack((dr, dc)), bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt((eigenvalues[-1] + 1) / (eigenvalues[0] + 1)))
        objects.append({"label": label, "area": len(cells),
                        "row": int(brightest[0]), "col": int(brightest[1]),
                        "peak": float(field[tuple(brightest)]), "elongation": elongation})
    labels[labels < 0] = 0
    return labels, objects


def links(before, after, objects, radius=2):
    """Link components if the after-mask overlaps a periodic dilation of the before-mask."""
    outgoing, incoming = {}, {}
    for obj in objects:
        region = before == obj["label"]
        expanded = np.zeros_like(region)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    expanded |= np.roll(region, (dy, dx), axis=(0, 1))
        connected = [int(x) for x in np.unique(after[expanded]) if x > 0]
        outgoing[obj["label"]] = connected
        for label in connected:
            incoming.setdefault(label, []).append(obj["label"])
    return outgoing, incoming


def single_successor(label, outgoing, incoming):
    children = outgoing.get(label, [])
    if len(children) != 1 or len(incoming.get(children[0], [])) != 1:
        return None
    return children[0]


def division_candidates(segmented):
    """One persistent parent and two exclusive children with persistent successors."""
    links_by_time = [links(a[0], b[0], a[1]) for a, b in zip(segmented, segmented[1:])]
    events, control_pool = [], {}
    for t in range(2, len(segmented) - 3):
        outgoing, incoming = links_by_time[t]
        next_out, next_in = links_by_time[t + 1]
        final_out, final_in = links_by_time[t + 2]

        def stable_history(label):
            for time in (t - 1, t - 2):
                old_out, old_in = links_by_time[time]
                parents = old_in.get(label, [])
                if len(parents) != 1 or single_successor(parents[0], old_out, old_in) != label:
                    return False
                label = parents[0]
            return True

        event_labels = set()
        for obj in segmented[t][1]:
            if not stable_history(obj["label"]):
                continue
            children = outgoing.get(obj["label"], [])
            if len(children) != 2 or any(len(incoming.get(c, [])) != 1 for c in children):
                continue
            successors = [single_successor(c, next_out, next_in) for c in children]
            if None in successors or len(set(successors)) != 2:
                continue
            final = [single_successor(c, final_out, final_in) for c in successors]
            if None in final or len(set(final)) != 2:
                continue
            event_labels.add(obj["label"])
            events.append({"state": t, "parent": obj, "children": children,
                           "persistent_children": successors})
        possible = []
        for obj in segmented[t][1]:
            if obj["label"] in event_labels or not stable_history(obj["label"]):
                continue
            first = single_successor(obj["label"], outgoing, incoming)
            second = (single_successor(first, next_out, next_in)
                      if first is not None else None)
            third = (single_successor(second, final_out, final_in)
                     if second is not None else None)
            if third is not None:
                possible.append(obj)
        control_pool[t] = possible
    return events, control_pool


def deduplicate_candidates(events, shape, time_radius=6, space_radius=7):
    """Suppress nearby repeated detections, without claiming physical identity tracking."""
    kept = []
    for event in events:
        point = (event["parent"]["row"], event["parent"]["col"])
        if any(event["state"] - other["state"] <= time_radius
               and periodic_distance(point, (other["parent"]["row"], other["parent"]["col"]),
                                     shape) <= space_radius for other in kept):
            continue
        kept.append(event)
    return kept


def match_controls(events, pool):
    """Same-state, roughly similar size, elongation and brightness; no reused controls."""
    used = set()
    for event in events:
        parent = event["parent"]
        eligible = []
        for control in pool[event["state"]]:
            if (event["state"], control["label"]) in used:
                continue
            area_difference = abs(np.log(control["area"] / parent["area"]))
            shape_difference = abs(np.log(control["elongation"] / parent["elongation"]))
            brightness_difference = abs(control["peak"] - parent["peak"])
            if area_difference <= np.log(1.75) and shape_difference <= np.log(1.5) \
                    and brightness_difference <= 0.1:
                score = (area_difference / np.log(1.75))**2 \
                        + (shape_difference / np.log(1.5))**2 \
                        + (brightness_difference / 0.1)**2
                eligible.append((score, control["label"], control))
        if eligible:
            score, _, control = min(eligible, key=lambda x: (x[0], x[1]))
            used.add((event["state"], control["label"]))
            event["control"] = control
            event["control_match_score"] = float(score)
        else:
            event["control"] = None
    return events


def dilate(mask, radius):
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out |= np.roll(mask, (dy, dx), axis=(0, 1))
    return out


def significant_peaks(field, region, base, prominence, step=0.005):
    """Find peaks resolved in a level sweep with the requested prominence."""
    masked = np.where(region, field, 0.0)
    best = []
    for level in np.arange(base, masked.max(), step):
        labels, objects = components(masked, level, minimum_area=2)
        found = [(o["row"], o["col"], o["peak"]) for o in objects
                 if o["peak"] - level >= prominence]
        if len(found) > len(best):
            best = found
    return best


def trace(trajectory, event, segmented, threshold, prominence, parent_label_key):
    t = event["state"]
    parent = event[parent_label_key]
    region_t = dilate(segmented[t][0] == parent["label"], 2)
    rows = []
    for offset in OFFSETS:
        labels = segmented[t + offset][0]
        touching = np.unique(labels[region_t])
        region = np.isin(labels, touching[touching > 0])
        peaks = significant_peaks(trajectory[t + offset, 1], region, threshold, prominence)
        separation = (periodic_distance(peaks[0][:2], peaks[1][:2], region.shape)
                      if len(peaks) >= 2 else 0.0)
        rows.append({"peaks": len(peaks), "separation": separation})
    return rows


def classify(rows):
    before = [rows[0]["peaks"]]
    after = [r["peaks"] for r in rows[3:]]
    separations = [r["separation"] for r in rows[3:]]
    if max(before) >= 2:
        return "two-peaked already at t-2"
    if min(after) < 2:
        return "not two-peaked afterwards"
    if separations[-1] <= separations[0]:
        return "separation not growing"
    return CLEAN


def event_sheet(trajectory, events, seed, out, page_size=8):
    for page, start in enumerate(range(0, len(events), page_size), 1):
        selected = events[start:start + page_size]
        fig, axes = plt.subplots(len(selected), 6, figsize=(13, 2 * len(selected)), squeeze=False)
        for row, event in enumerate(selected):
            t, parent = event["state"], event["parent"]
            for col, offset in enumerate((-2, -1, 0, 1, 2, 3)):
                patch = periodic_patch(trajectory[t + offset, 1], parent["row"], parent["col"], 12)
                ax = axes[row, col]
                ax.imshow(patch, cmap="inferno", vmin=0, vmax=0.5)
                ax.scatter([12], [12], marker="+", s=22, color="white")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"t{offset:+d}" if offset else "t: parent")
                if col == 0:
                    ax.set_ylabel(f"seed {seed}\nstate {t}\n"
                                  f"control {'yes' if event['control'] else 'no'}", fontsize=8)
        fig.suptitle("Automatic division candidates: inspect before assigning physical labels")
        fig.text(0.5, 0.01, "25x25 periodic patches centered at parent peak at t. "
                 "Fixed v scale 0-0.5; time between frames = 100 model steps.\n"
                 "A split candidate has two component successors persisting two further frames; "
                 "detector output is not proof of division.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(out / f"candidates_seed_{seed}_page_{page:02d}.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--seed-selection", required=True)
    parser.add_argument("--states", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    seeds = args.discovery_seeds + args.evaluation_seeds
    if len(set(seeds)) != len(seeds) or args.states < 10:
        parser.error("use distinct seeds and at least 10 states")
    data = load_dataset(args.data)
    settings = data["settings"]
    if settings["seed"] in seeds:
        parser.error("simulated seeds must differ from the original dataset")
    if settings["history"] != 2 or not (np.isclose(settings["Du"], DU)
                                         and np.isclose(settings["Dv"], DV)):
        raise ValueError("requires history=2 and simulator-default diffusion")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    offset = len(data["splits"]["train"]["inputs"]) + settings["gap"] + 60
    summary_seeds = []
    for seed in seeds:
        print(f"Simulating and screening seed {seed}", flush=True)
        u, v = seed_pattern(settings["N"], settings["n_spots"], np.random.default_rng(seed))
        u, v = advance(u, v, settings["warm_up"] + offset * settings["stride"],
                       settings["F"], settings["k"], settings["dt"])
        trajectory = integrate(u, v, (args.states + 5) * settings["stride"], settings["F"],
                               settings["k"], settings["dt"], sample_every=settings["stride"])
        segmented = [components(frame[1], args.threshold) for frame in trajectory]
        events, pool = division_candidates(segmented)
        raw_count = len(events)
        events = deduplicate_candidates(events, trajectory.shape[-2:])
        match_controls(events, pool)
        event_sheet(trajectory, events, seed, out)
        summary_seeds.append({"seed": seed, "raw_candidate_count": raw_count,
                              "candidate_count": len(events),
                              "matched_count": sum(event["control"] is not None
                                                   for event in events),
                              "state_count": len(trajectory),
                              "states_with_candidates": len({event["state"] for event in events}),
                              "component_observations": sum(len(frame[1])
                                                            for frame in segmented[2:-3]),
                              "events": events})
        print(f"seed {seed}: {len(events)} candidates, "
              f"{summary_seeds[-1]['matched_count']} matched controls", flush=True)
    summary = {
        "seed_selection": args.seed_selection,
        "discovery_seeds": args.discovery_seeds, "evaluation_seeds": args.evaluation_seeds,
        "simulation_start_calls_after_warmup": offset, "samples_per_call": settings["stride"],
        "component_threshold": args.threshold, "minimum_component_area": 4,
        "definition": "Periodic 4-connected components of v above threshold. A candidate is "
                      "a parent with one exclusive predecessor for two earlier states and "
                      "exactly two exclusive overlapping/dilated successors "
                      "at the next state, each with an exclusive distinct successor one state "
                      "later and another one state after that. Dilation radius 2. Greedy "
                      "duplicate suppression within 6 samples and 7 cells may suppress distinct "
                      "events or miss duplicates. These are candidate detections, not verified "
                      "blob divisions and not necessarily independent events.",
        "control_definition": "One component at the same state, with one exclusive successor "
                              "for each of the next three states and one predecessor each for "
                              "the previous two; area ratio within 1.75, "
                              "elongation ratio within 1.5, peak v within 0.1. Greedy one-to-one "
                              "matching within each state; approximate, not exact matching.",
        "per_seed": summary_seeds,
        "limitations": "Same parameters; survival-selected initializations; correlated cells, "
                       "frames and candidate events. Components may merge, split or change "
                       "threshold without physical division. Short observations only; no "
                       "interventions, no claims of verified births or unique event counts.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
