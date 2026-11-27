"""Draw the two regimes: compact spots, elongated sticks, and the elongation that separates them.

    ../../../.venv/bin/python make_fig1.py --data data/gray_scott_mixed.npz --out figures

Nothing is trained here. The figure is the introduction to the two-regime study: the same
seeding recipe at two feed rates, and the measurement used everywhere else -- the
principal-axis elongation of a connected component.

Writes ``<out>/regimes_two.png`` and ``<model-dir>/system.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_data import (ELONGATION_MIN, advance, connected_components, coverage,
                           load_dataset, morphology, seed_pattern)

SNAPSHOTS = 3
STEPS_BETWEEN = 4000
WARM_UP = 2000
# The disc threshold scan: the smallest amplitude of v that still leaves a component.
SCAN_RADII = (2, 3, 4, 5, 6)
SCAN_AMPLITUDES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
SCAN_STEPS = 3000
SURVIVAL_COVERAGE = 0.01


def disc_field(n, radius, amplitude, centre=None):
    """The uniform state with one disc of v raised and u lowered to match."""
    rows, cols = np.mgrid[0:n, 0:n]
    centre_row, centre_col = (n // 2, n // 2) if centre is None else centre
    dy = np.minimum((rows - centre_row) % n, (centre_row - rows) % n)
    dx = np.minimum((cols - centre_col) % n, (centre_col - cols) % n)
    inside = dy**2 + dx**2 <= radius**2
    u = np.ones((n, n))
    v = np.zeros((n, n))
    u[inside] = 1.0 - amplitude
    v[inside] = amplitude
    return u, v


def threshold_scan(n, F, k, dt):
    """Coverage after ``SCAN_STEPS`` for every disc radius and amplitude."""
    final = np.zeros((len(SCAN_RADII), len(SCAN_AMPLITUDES)))
    for row, radius in enumerate(SCAN_RADII):
        for col, amplitude in enumerate(SCAN_AMPLITUDES):
            u, v = disc_field(n, radius, amplitude)
            u, v = advance(u, v, SCAN_STEPS, F, k, dt)
            final[row, col] = coverage(v)
    return final


def run_regime(n, F, k, n_spots, seed, steps_between=STEPS_BETWEEN, warm_up=WARM_UP):
    """Integrate one regime and return its snapshots and the shape measured at each."""
    u, v = seed_pattern(n, n_spots, np.random.default_rng(seed))
    u, v = advance(u, v, warm_up, F, k)
    snapshots, measurements = [], []
    for index in range(SNAPSHOTS):
        u, v = advance(u, v, steps_between, F, k)
        snapshots.append(v.copy())
        measurements.append({"step": (index + 1) * steps_between, **morphology(v)})
    if coverage(v) < 0.005:
        raise ValueError(f"the pattern died for F={F}, k={k}, seed {seed}")
    return snapshots, measurements


def display_name(name):
    """The regime name used in the figure; the dataset calls regime 0 'main'."""
    return "blob" if str(name) == "main" else str(name)


def figure(settings, results, out_path, dpi=150):
    """Two rows, one per regime: the field and the component elongation over the run."""
    names = [display_name(name) for name in settings["regime_names"]]
    fig = plt.figure(figsize=(15.0, 7.2))
    grid = fig.add_gridspec(len(names), SNAPSHOTS + 1, hspace=0.32, wspace=0.22,
                            left=0.05, right=0.98, top=0.92, bottom=0.12)
    for row, name in enumerate(names):
        entry = results[row]
        for col, snapshot in enumerate(entry["snapshots"]):
            axis = fig.add_subplot(grid[row, col])
            image = axis.imshow(snapshot, cmap="inferno", vmin=0.0, vmax=0.45)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{display_name(name)}: {entry['measurements'][col]['step']} steps later",
                           fontsize=9)
            if col == SNAPSHOTS - 1:
                fig.colorbar(image, ax=axis, fraction=0.046, label="$v$")
        axis = fig.add_subplot(grid[row, SNAPSHOTS])
        elongations = [component["elongation"]
                       for component in connected_components(entry["snapshots"][-1])]
        if elongations:
            axis.hist(elongations, bins=np.arange(1.0, max(12.0, max(elongations) + 1.0), 0.5),
                      color="#4c72b0" if row == 0 else "#c44e52", alpha=0.85)
        axis.axvline(ELONGATION_MIN, color="#333333", linestyle="--", lw=1.2,
                     label=f"stick threshold {ELONGATION_MIN:g}")
        axis.set_xlabel("component elongation")
        axis.set_ylabel("components")
        axis.set_title("elongation at the last snapshot", fontsize=9)
        axis.legend(fontsize=7.5)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("the same seeding recipe at two feed rates", fontsize=12)
    fig.text(0.5, 0.015,
             "Elongation is the square root of the ratio of the two principal-axis eigenvalues "
             "of a connected component of $v > 0.25$. A disc is about 1; a stick grows with "
             "its length.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Draw the two Gray-Scott regimes.")
    parser.add_argument("--data", default="data/gray_scott_mixed.npz")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", action="store_true",
                        help="scan the disc threshold at the main regime instead of drawing "
                             "the two-regime figure")
    parser.add_argument("--main-index", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    settings = load_dataset(args.data)["settings"]
    n = settings["N"]
    if args.threshold:
        if not 0 <= args.main_index < settings["n_regimes"]:
            parser.error("--main-index must name one of the dataset's regimes")
        F = float(settings["regime_F"][args.main_index])
        k = float(settings["regime_k"][args.main_index])
        print(f"disc threshold scan at {settings['regime_names'][args.main_index]}: "
              f"F={F}, k={k}, {SCAN_STEPS} steps")
        final = threshold_scan(n, F, k, settings["dt"])
        smallest = {}
        for row, radius in enumerate(SCAN_RADII):
            value = next((SCAN_AMPLITUDES[col] for col, covered in enumerate(final[row])
                          if covered > SURVIVAL_COVERAGE), None)
            smallest[str(radius)] = value
            print(f"  radius {radius}: smallest surviving amplitude {value}")
        record = {"F": F, "k": k, "steps": SCAN_STEPS, "radii": list(SCAN_RADII),
                  "amplitudes": list(SCAN_AMPLITUDES), "coverage": final.tolist(),
                  "smallest_surviving": smallest}
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "threshold.json").write_text(json.dumps(record, indent=2))
        print(f"wrote {model_dir / 'threshold.json'}")
        return
    results = []
    for regime in range(settings["n_regimes"]):
        F = float(settings["regime_F"][regime])
        k = float(settings["regime_k"][regime])
        print(f"integrating {settings['regime_names'][regime]}: F={F}, k={k}")
        snapshots, measurements = run_regime(n, F, k, settings["n_spots"],
                                             args.seed + regime)
        results.append({"name": str(settings["regime_names"][regime]), "F": F, "k": k,
                        "snapshots": snapshots, "measurements": measurements})
        last = measurements[-1]
        print(f"  coverage {last['coverage']:.3f}, {last['components']} components, "
              f"elongation median {last['elongation_median']:.2f} / max "
              f"{last['elongation_max']:.2f}, stick cells {last['stick_cell_fraction']:.2f}")

    output = figure(settings, results, Path(args.out) / "regimes_two.png", args.dpi)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "F": [entry["F"] for entry in results],
        "k": [entry["k"] for entry in results],
        "names": [entry["name"] for entry in results],
        "measurements": [entry["measurements"] for entry in results],
        "steps_between": STEPS_BETWEEN,
        "warm_up": WARM_UP,
    }
    (model_dir / "system.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {output}")
    print(f"wrote {model_dir / 'system.json'}")


if __name__ == "__main__":
    main()
