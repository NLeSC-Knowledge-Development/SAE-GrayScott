"""Simulate Gray-Scott in two regimes and build a prediction dataset for a conditioned emulator.

    ../../../.venv/bin/python generate_data.py --out data
    ../../../.venv/bin/python generate_data.py --explore --figures figures

The system
----------
Gray-Scott is a pair of chemicals on a periodic grid::

    du/dt = Du * lap(u) - u*v^2 + F*(1 - u)
    dv/dt = Dv * lap(v) + u*v^2 - (F + k)*v

``u*v^2`` is the reaction and the only source of new ``v``; ``F*(1-u)`` feeds ``u`` back
towards 1; ``-(F+k)*v`` drains ``v``; ``Du = 0.16`` and ``Dv = 0.08`` diffuse the two fields.
The uniform state ``u = 1, v = 0`` is a fixed point, and a local disturbance above a threshold
becomes a self-sustaining spot instead of decaying.

The two regimes
---------------
``F`` and ``k`` decide what the pattern does. This folder uses two pairs and one emulator
conditioned on which pair produced the state:

* ``main`` -- ``F = 0.026``, ``k = 0.059``. Spot-dominated: on one 64 x 64 run after
  10 000 steps, coverage 0.219, 23 connected components, median elongation 1.16, maximum
  1.96. It is shifted from project_4's ``F = 0.022`` because a stick-shaped perturbation
  survives longer there: a 12 x 3 stick of ``v = 0.5`` leaves maximum elongation 1.9 near
  its site after 500-1000 steps, against 0.50 and 0.38 at ``F = 0.022``. This is the regime
  the injection experiment runs in.
* ``sticks`` -- ``F = 0.030``, ``k = 0.059``. The same seeding recipe grows elongated
  components. Measured on one 64 x 64 run after 10 000 steps: coverage 0.263 of the grid,
  5 connected components, median elongation 6.53, maximum 8.26. The project_4 spot regime
  at ``F = 0.022`` gives coverage 0.049, 10 components, maximum 2.68 and median 1.47.

Elongation is measured on connected components of ``v > 0.25`` as the square root of the
ratio of the two principal-axis eigenvalues of the cell coordinates. A disc gives about 1;
a line gives a number that grows with its length.

The task
--------
The emulator is given the state now, the state one emulator call ago, and a regime index, and
must produce the state one call ahead. One call is ``--stride`` model steps (100 by default).
The regime index is a learned embedding added to every cell's token, so the same network can
hold two transition rules and the experiment can say which rule it is running.

Writes ``<out>/gray_scott_mixed.npz``: the inputs, targets and regime labels of the three
splits, the standardisation statistics, the settings, and the measurements of both patterns.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The system. Du = 2 * Dv is the standard Gray-Scott choice.
DU = 0.16
DV = 0.08
DT = 1.0

DEFAULT_N = 64
# The two regimes. Same k, different F: the feed rate is what turns compact spots into
# elongated components. The main regime is shifted from project_4's 0.022 to 0.026 so that a
# stick-shaped perturbation survives long enough to be measured: on the same 10 000-step run
# the main regime holds 23 round components with maximum elongation 1.96, and a 12 x 3 stick
# of amplitude 0.5 still has maximum elongation 1.9 near its site after 500-1000 steps. At
# F = 0.022 the same perturbation leaves 0.50.
MAIN_F = 0.026
MAIN_K = 0.059
STICK_F = 0.030
STICK_K = 0.059
REGIME_F = (MAIN_F, STICK_F)
REGIME_K = (MAIN_K, STICK_K)
REGIME_NAMES = ("blob", "sticks")
DEFAULT_MAIN_SAMPLES = 1600
DEFAULT_STICK_SAMPLES = 400
DEFAULT_STRIDE = 100
DEFAULT_HISTORY = 2
DEFAULT_SPOTS = 25
SEED_RADIUS = 3.0

# What counts as "inside a component" and how small a component may be to be measured.
SPOT_THRESHOLD = 0.25
SPOT_RADIUS = 4
MIN_COMPONENT_CELLS = 8
# Components at or above this principal-axis ratio are called sticks when a run is summarised.
ELONGATION_MIN = 3.0

EXPLORE_F = (0.014, 0.022, 0.030, 0.038, 0.046, 0.054)
EXPLORE_K = (0.045, 0.052, 0.059, 0.066)


# ------------------------------------------------------------------------------- the system


def laplacian(field):
    """The five-point stencil of the Laplacian, on a periodic grid with ``dx = 1``."""
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
            - 4.0 * field)


def euler_step(u, v, F, k, dt=DT, Du=DU, Dv=DV):
    """One explicit Euler step of the Gray-Scott equations."""
    reaction = u * v * v
    u_next = u + dt * (Du * laplacian(u) - reaction + F * (1.0 - u))
    v_next = v + dt * (Dv * laplacian(v) + reaction - (F + k) * v)
    return u_next, v_next


def advance(u, v, n_steps, F, k, dt=DT):
    """Integrate forward, recording nothing."""
    for _ in range(n_steps):
        u, v = euler_step(u, v, F, k, dt)
    return u, v


def integrate(u, v, n_steps, F, k, dt=DT, sample_every=1):
    """Integrate, keeping the state every ``sample_every`` steps, the initial state first."""
    if n_steps % sample_every != 0:
        raise ValueError(f"n_steps ({n_steps}) must be a multiple of sample_every "
                         f"({sample_every})")
    recorded = np.empty((n_steps // sample_every + 1, 2, *u.shape), dtype=np.float64)
    recorded[0, 0], recorded[0, 1] = u, v
    index = 1
    for step_index in range(1, n_steps + 1):
        u, v = euler_step(u, v, F, k, dt)
        if step_index % sample_every == 0:
            recorded[index, 0], recorded[index, 1] = u, v
            index += 1
    return recorded[:index]


def seed_pattern(n, n_spots, rng, radius=SEED_RADIUS):
    """A periodic grid holding ``u = 1``, ``v = 0``, with ``n_spots`` discs planted in it."""
    u = np.ones((n, n))
    v = np.zeros((n, n))
    side = int(np.ceil(np.sqrt(n_spots)))
    spacing = n / side
    for row in range(side):
        for col in range(side):
            centre_row = (row + 0.5) * spacing + rng.uniform(-0.3, 0.3) * spacing
            centre_col = (col + 0.5) * spacing + rng.uniform(-0.3, 0.3) * spacing
            dy = np.minimum((np.arange(n) - centre_row) % n, (centre_row - np.arange(n)) % n)
            dx = np.minimum((np.arange(n) - centre_col) % n, (centre_col - np.arange(n)) % n)
            inside = dy[:, None] ** 2 + dx[None, :] ** 2 <= radius**2
            u[inside] = 0.50
            v[inside] = 0.25
    return u, v


# --------------------------------------------------------------- morphology of a field


def coverage(v, threshold=SPOT_THRESHOLD):
    """The fraction of the grid above the component threshold."""
    return float((v > threshold).mean())


def connected_components(v, threshold=SPOT_THRESHOLD, min_cells=MIN_COMPONENT_CELLS):
    """Connected components of ``v > threshold`` on the periodic grid, with their shape.

    Returns one dict per component with at least ``min_cells`` cells. ``elongation`` is the
    square root of the ratio of the two principal-axis eigenvalues of the cell coordinates;
    a disc is about 1, a line grows with its length. Coordinates are unwrapped relative to a
    reference cell before the covariance is taken, so a component crossing the periodic edge
    is still measured as one object.
    """
    mask = v > threshold
    n = v.shape[0]
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for start in np.argwhere(mask):
        start = (int(start[0]), int(start[1]))
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        cells = []
        while queue:
            row, col = queue.popleft()
            cells.append((row, col))
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                near = ((row + d_row) % n, (col + d_col) % n)
                if mask[near] and not seen[near]:
                    seen[near] = True
                    queue.append(near)
        if len(cells) < min_cells:
            continue
        cells = np.asarray(cells, dtype=np.float64)
        reference = cells[0]
        offsets = cells - reference
        offsets = (offsets + 0.5 * n) % n - 0.5 * n
        centre = reference + offsets.mean(axis=0)
        centre = centre % n
        covariance = np.cov(offsets.T)
        eigenvalues = np.linalg.eigvalsh(covariance).clip(1e-8, None)
        elongation = float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
        components.append({
            "cells": cells.astype(int),
            "offsets": offsets,
            "eigenvalues": eigenvalues,
            "area": int(len(cells)),
            "centroid": (float(centre[0]), float(centre[1])),
            "elongation": elongation,
            "major_length": float(np.sqrt(eigenvalues[1]) * 4.0),
            "minor_length": float(np.sqrt(eigenvalues[0]) * 4.0),
        })
    return components


def _convex_hull_area(points):
    """Area of the convex hull of a set of 2-D points (monotone chain)."""
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 6), axis=0)
    if len(pts) < 3:
        return float(len(pts))
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return float(len(pts))
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _thin(mask):
    """Zhang-Suen thinning, used only for component shape descriptors."""
    image = mask.astype(np.uint8).copy()
    if image.sum() == 0:
        return image
    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            sequence = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = sum(((sequence[i] == 0) & (sequence[i + 1] == 1)).astype(np.uint8)
                              for i in range(8))
            if step == 0:
                remove = ((image == 1) & (neighbours >= 2) & (neighbours <= 6)
                          & (transitions == 1) & ((p2 * p4 * p6) == 0)
                          & ((p4 * p6 * p8) == 0))
            else:
                remove = ((image == 1) & (neighbours >= 2) & (neighbours <= 6)
                          & (transitions == 1) & ((p2 * p4 * p8) == 0)
                          & ((p2 * p6 * p8) == 0))
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image


def describe_component(field, component):
    """Shape descriptors for one component: solidity and skeleton topology.

    ``solidity`` is area divided by convex-hull area, so a compact blob or a straight stick is
    near 1 and a branched labyrinth is low. The skeleton is the Zhang-Suen thinning of the
    component in its own unwrapped frame; its pixel count is an approximate centerline length,
    and ``area / skeleton_length`` is an approximate width.
    """
    offsets = component.get("offsets")
    if offsets is None:
        cells = component["cells"].astype(np.float64)
        offsets = cells - cells[0]
        offsets = (offsets + 0.5 * field.shape[0]) % field.shape[0] - 0.5 * field.shape[0]
    local = np.rint(offsets - offsets.min(axis=0)).astype(int)
    shape = local.max(axis=0) + 1
    mask = np.zeros(shape, dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    area = int(component["area"])
    hull_area = _convex_hull_area(offsets)
    skeleton_length, branch_points, endpoints = 0, 0, 0
    if mask.sum() > 0:
        skeleton = _thin(mask)
        skeleton_length = int(skeleton.sum())
        if skeleton_length:
            padded = np.pad(skeleton, 1)
            degree = np.zeros_like(skeleton, dtype=np.uint8)
            for row in (-1, 0, 1):
                for col in (-1, 0, 1):
                    if row == 0 and col == 0:
                        continue
                    degree += padded[1 + row:1 + row + skeleton.shape[0],
                                      1 + col:1 + col + skeleton.shape[1]]
            degree = degree * skeleton
            branch_points = int(((degree >= 3) & (skeleton == 1)).sum())
            endpoints = int(((degree == 1) & (skeleton == 1)).sum())
    rows, cols = component["cells"].T
    values = field[rows, cols]
    return {
        "solidity": float(min(1.0, area / max(hull_area, 1.0))),
        "skeleton_length": float(skeleton_length),
        "skeleton_width": float(area / max(skeleton_length, 1)),
        "branch_points": int(branch_points),
        "endpoints": int(endpoints),
        "mean_v": float(values.mean()),
        "max_v": float(values.max()),
    }


def describe_field(v, threshold=SPOT_THRESHOLD, min_cells=MIN_COMPONENT_CELLS):
    """Connected components with their elongation and skeleton/solidity descriptors."""
    described = []
    for component in connected_components(v, threshold, min_cells):
        component.update(describe_component(v, component))
        described.append(component)
    return described


def elongation_map(v, threshold=SPOT_THRESHOLD, min_cells=MIN_COMPONENT_CELLS):
    """One number per cell: the elongation of the component the cell belongs to, else zero.

    This is the label a token is paired with when the question is *does a direction know this
    cell is part of an elongated stick rather than a round spot?*
    """
    out = np.zeros_like(v, dtype=np.float64)
    for component in connected_components(v, threshold, min_cells):
        rows, cols = component["cells"].T
        out[rows, cols] = component["elongation"]
    return out


def stick_mask(v, threshold=SPOT_THRESHOLD, min_elongation=ELONGATION_MIN,
               min_cells=MIN_COMPONENT_CELLS):
    """Cells that belong to a component at or above ``min_elongation``."""
    out = np.zeros_like(v, dtype=bool)
    for component in connected_components(v, threshold, min_cells):
        if component["elongation"] >= min_elongation:
            rows, cols = component["cells"].T
            out[rows, cols] = True
    return out


def morphology(v, threshold=SPOT_THRESHOLD, min_elongation=ELONGATION_MIN):
    """A one-line summary of a field's components, including how much of it is in sticks."""
    components = connected_components(v, threshold)
    elongations = np.array([component["elongation"] for component in components])
    areas = np.array([component["area"] for component in components])
    in_sticks = sum(component["area"] for component in components
                    if component["elongation"] >= min_elongation)
    covered = max(float((v > threshold).sum()), 1.0)
    return {
        "coverage": coverage(v, threshold),
        "components": int(len(components)),
        "elongation_median": float(np.median(elongations)) if len(elongations) else 0.0,
        "elongation_max": float(elongations.max()) if len(elongations) else 0.0,
        "area_median": float(np.median(areas)) if len(areas) else 0.0,
        "stick_coverage": float(np.mean([component["elongation"] >= min_elongation
                                         for component in components]))
        if components else 0.0,
        "stick_cell_fraction": float(in_sticks / covered),
    }


def trajectory_morphology(trajectory, samples=20):
    """What the pattern looks like over a recorded trajectory, not only at its last state.

    A long run can pass through a stretched phase and end in a compact one; the single final
    state is then a poor summary of what the SAE was trained on. This samples ``samples``
    states evenly through the run and reports the median of each shape measurement, plus the
    fraction of sampled states that hold a component at or above ``ELONGATION_MIN``.
    """
    indices = np.linspace(0, len(trajectory) - 1,
                          min(samples, len(trajectory))).round().astype(int)
    shapes = [morphology(trajectory[index, 1]) for index in indices]
    return {
        "coverage": float(np.median([shape["coverage"] for shape in shapes])),
        "components": float(np.median([shape["components"] for shape in shapes])),
        "elongation_median": float(np.median([shape["elongation_median"]
                                               for shape in shapes])),
        "elongation_max": float(np.median([shape["elongation_max"] for shape in shapes])),
        "area_median": float(np.median([shape["area_median"] for shape in shapes])),
        "stick_coverage": float(np.mean([shape["stick_coverage"] for shape in shapes])),
        "stick_cell_fraction": float(np.mean([shape["stick_cell_fraction"]
                                              for shape in shapes])),
        "states_with_sticks": float(np.mean([shape["elongation_max"] >= ELONGATION_MIN
                                             for shape in shapes])),
    }


def spot_centres(v, threshold=SPOT_THRESHOLD, radius=SPOT_RADIUS):
    """The middle of each compact bright region, by greedy non-maximum suppression.

    This is the project_4 detector, kept because the event labels and the coverage curves use
    it. It is not a component detector: on an elongated stick it places several centres along
    the stick, which is why the stick measurements use :func:`connected_components` instead.
    """
    offsets = [(dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if dy * dy + dx * dx <= radius * radius]
    candidates = np.argwhere(v > threshold)
    if len(candidates) == 0:
        return []
    order = np.argsort(-v[candidates[:, 0], candidates[:, 1]])
    taken = np.zeros(v.shape, dtype=bool)
    centres = []
    for index in order:
        row, col = candidates[index]
        if taken[row, col]:
            continue
        centres.append((int(row), int(col)))
        for dy, dx in offsets:
            taken[(row + dy) % v.shape[0], (col + dx) % v.shape[1]] = True
    return centres


def pattern_length(v):
    """The typical spacing between bright regions, from the spatial autocorrelation."""
    n = v.shape[0]
    centred = v - v.mean()
    spectrum = np.fft.fft2(centred)
    correlation = np.real(np.fft.ifft2(spectrum * np.conj(spectrum)))
    correlation = np.fft.fftshift(correlation)
    centre = n // 2
    rows, cols = np.mgrid[0:n, 0:n]
    radius = np.sqrt((rows - centre) ** 2 + (cols - centre) ** 2).astype(int)
    counts = np.bincount(radius.ravel())
    radial = np.bincount(radius.ravel(), correlation.ravel()) / np.maximum(counts, 1)
    below = np.flatnonzero(radial[1:] <= 0.0)
    return float(2.0 * (below[0] + 1)) if below.size else float(n)


def decorrelation_time(recorded, sample_every, threshold=0.5):
    """How many model steps it takes the pattern to half-forget itself, in model steps."""
    signals = recorded[:, 1].reshape(len(recorded), -1)
    centred = signals - signals.mean(axis=0, keepdims=True)
    n = len(centred)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=0)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), axis=0)[:n].mean(axis=1)
    correlation = correlation / correlation[0]
    below = np.flatnonzero(correlation < threshold)
    return float(below[0] * sample_every) if below.size else float("nan")


def verdict(coverage_early, coverage_late, spots_early, spots_late, half_life,
            stick_fraction, elongation_max):
    """A one-word reading of a run: what happened to the pattern while it was integrated."""
    if coverage_late < 0.005:
        return "decayed"
    if coverage_late > 0.6:
        return "filled"
    if elongation_max >= ELONGATION_MIN and stick_fraction >= 0.1:
        return "sticks"
    if spots_late > 1.5 * max(spots_early, 1):
        return "multiplying"
    if np.isnan(half_life):
        return "frozen"
    return "changing"


def explore(n, n_steps, n_spots, seed, out_path, sample_every=20, dpi=150):
    """Walk the (F, k) plane, report what each pair does, and draw the contact sheet."""
    rows, final = [], {}
    for F in EXPLORE_F:
        for k in EXPLORE_K:
            u, v = seed_pattern(n, n_spots, np.random.default_rng(seed))
            recorded = integrate(u, v, n_steps, F, k, sample_every=sample_every)
            early = max(len(recorded) // 8, 1)
            coverage_early = coverage(recorded[early, 1])
            coverage_late = coverage(recorded[-1, 1])
            spots_early = len(spot_centres(recorded[early, 1]))
            spots_late = len(spot_centres(recorded[-1, 1]))
            shape = morphology(recorded[-1, 1])
            half_life = decorrelation_time(recorded, sample_every)
            rows.append({
                "F": F, "k": k, "coverage": coverage_late,
                "components": shape["components"],
                "elongation_median": shape["elongation_median"],
                "elongation_max": shape["elongation_max"],
                "stick_cell_fraction": shape["stick_cell_fraction"],
                "spots": spots_late,
                "spacing": pattern_length(recorded[-1, 1]),
                "half_life": half_life,
                "verdict": verdict(coverage_early, coverage_late, spots_early, spots_late,
                                   half_life, shape["stick_cell_fraction"],
                                   shape["elongation_max"]),
            })
            final[(F, k)] = recorded[-1, 1]

    print(f"\n{n} x {n} cells, {n_steps} steps, {n_spots} seeds, "
          f"Du = {DU}, Dv = {DV}, dt = {DT}")
    print(f"\n  {'F':>7s}{'k':>8s}{'coverage':>10s}{'comps':>7s}{'elong med':>11s}"
          f"{'elong max':>11s}{'stick cell':>12s}{'half-life':>11s}   what happened")
    for row in rows:
        half_life = "static" if np.isnan(row["half_life"]) else f"{row['half_life']:.0f}"
        print(f"  {row['F']:7.4f}{row['k']:8.4f}{row['coverage']:10.3f}"
              f"{row['components']:7d}{row['elongation_median']:11.2f}"
              f"{row['elongation_max']:11.2f}{row['stick_cell_fraction']:12.2f}"
              f"{half_life:>11s}   {row['verdict']}")

    fig, axes = plt.subplots(len(EXPLORE_F), len(EXPLORE_K),
                             figsize=(2.5 * len(EXPLORE_K), 2.5 * len(EXPLORE_F)))
    for row_index, F in enumerate(EXPLORE_F):
        for col_index, k in enumerate(EXPLORE_K):
            entry = next(row for row in rows if row["F"] == F and row["k"] == k)
            axis = axes[row_index, col_index]
            axis.imshow(final[(F, k)], cmap="inferno", vmin=0.0, vmax=0.4)
            axis.set_title(f"F={F:.3f}  k={k:.3f}\n{entry['verdict']}, "
                           f"{entry['components']} comps, elong "
                           f"{entry['elongation_median']:.1f}/{entry['elongation_max']:.1f}",
                           fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("which (F, k) does what: the field v after "
                 f"{n_steps} steps", fontsize=11)
    fig.text(0.5, 0.01, "every panel starts from the same seeded pattern. 'elong' is the "
                        "principal-axis ratio of connected components, median / maximum.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure_path = out_path / "regimes.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi)
    plt.close(fig)

    (out_path / "regimes.json").write_text(json.dumps(
        {"steps": int(n_steps), "cells": int(n), "seeds": int(n_spots),
         "F": list(EXPLORE_F), "k": list(EXPLORE_K), "rows": rows}, indent=2))
    print(f"\nwrote {figure_path}")
    print(f"wrote {out_path / 'regimes.json'}")
    return rows


# ------------------------------------------------------------------------------ the dataset


def build_examples(trajectory, history=DEFAULT_HISTORY):
    """Turn a recorded trajectory into ``(inputs, targets)`` examples, one call apart."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 4:
        raise ValueError(f"trajectory must be (time, 2, n, n), got {trajectory.shape}")
    n_examples = len(trajectory) - history
    windows = np.lib.stride_tricks.sliding_window_view(trajectory, history, axis=0)
    inputs = np.moveaxis(windows[:n_examples], -1, 1)
    return np.ascontiguousarray(inputs), trajectory[history:]


def split_trajectory(trajectory, history, fractions, gap):
    """Three contiguous blocks in time, with ``gap`` recorded samples dropped between them."""
    usable = len(trajectory) - 2 * gap
    n_train = int(usable * fractions[0])
    n_val = int(usable * fractions[1])
    train_slice = trajectory[: n_train + history]
    val_slice = trajectory[n_train + gap : n_train + gap + n_val + history]
    test_slice = trajectory[n_train + gap + n_val + gap :]
    return {
        "train": build_examples(train_slice, history),
        "val": build_examples(val_slice, history),
        "test": build_examples(test_slice, history),
    }


def make_dataset(n_steps, stride, n=DEFAULT_N, regimes=tuple(zip(REGIME_F, REGIME_K)),
                 regime_samples=None, dt=DT, history=DEFAULT_HISTORY, n_spots=DEFAULT_SPOTS,
                 warm_up=2000, gap=20, fractions=(0.7, 0.15, 0.15), seed=0):
    """Simulate one trajectory per regime and build one mixed dataset.

    ``n_steps`` is kept as the main regime's number of recorded states; ``regime_samples``
    gives the number for every regime and defaults to a main-heavy mix, 1600 main states and
    400 stick states. Each regime is split into contiguous train / validation / test blocks on
    its own, so a validation state never sits next to a training state of the same run. The
    per-regime blocks are then concatenated and each example carries an integer ``regime``
    label.
    The standardisation is one mean and one standard deviation per field, taken from the
    combined training inputs: the emulator receives the regime index separately, so the fields
    themselves are put on one common scale.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if regime_samples is None:
        regime_samples = ((DEFAULT_MAIN_SAMPLES, DEFAULT_STICK_SAMPLES)
                          if len(regimes) == 2 else (DEFAULT_MAIN_SAMPLES,) * len(regimes))
    regime_samples = tuple(int(value) for value in regime_samples)
    if len(regime_samples) != len(regimes):
        raise ValueError(f"one sample count per regime: {len(regimes)} regimes but "
                         f"{len(regime_samples)} counts")
    if n_steps != regime_samples[0]:
        raise ValueError(f"n_steps ({n_steps}) must equal the first regime's sample count "
                         f"({regime_samples[0]})")
    parts = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    regime_settings = []
    for regime_index, (F, k) in enumerate(regimes):
        u, v = seed_pattern(n, n_spots, np.random.default_rng(seed + regime_index))
        u, v = advance(u, v, warm_up, F, k, dt)
        trajectory = integrate(u, v, regime_samples[regime_index] * stride, F, k, dt,
                               sample_every=stride)
        blocks = split_trajectory(trajectory, history, fractions, gap)
        for name in parts:
            parts[name].append(blocks[name])
            labels[name].append(np.full(len(blocks[name][0]), regime_index, dtype=np.int64))
        shape = trajectory_morphology(trajectory)
        regime_settings.append({
            "index": int(regime_index),
            "name": REGIME_NAMES[regime_index] if regime_index < len(REGIME_NAMES)
            else f"regime {regime_index}",
            "F": float(F), "k": float(k),
            **shape,
            "spacing": pattern_length(trajectory[-1, 1]),
            "half_life": decorrelation_time(trajectory, stride),
        })

    combined = {}
    for name in parts:
        raw_x = np.concatenate([block[0] for block in parts[name]], axis=0)
        raw_y = np.concatenate([block[1] for block in parts[name]], axis=0)
        combined[name] = (raw_x, raw_y, np.concatenate(labels[name]))

    mean = combined["train"][0].mean(axis=(0, 1, 3, 4))
    std = combined["train"][0].std(axis=(0, 1, 3, 4))
    if np.any(std <= 0.0):
        raise ValueError("the training inputs are constant: the pattern decayed, and this "
                         "dataset has nothing to learn from")

    splits = {}
    for name, (raw_x, raw_y, regime) in combined.items():
        shape = (1, 1, 2, 1, 1)
        splits[name] = {
            "inputs": ((raw_x - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32),
            "targets": ((raw_y - mean.reshape(1, 2, 1, 1))
                        / std.reshape(1, 2, 1, 1)).astype(np.float32),
            "raw_targets": raw_y.astype(np.float32),
            "regime": regime,
        }

    settings = {
        "N": int(n), "stride": int(stride), "history": int(history), "seed": int(seed),
        "n_samples": int(regime_samples[0]), "n_spots": int(n_spots),
        "warm_up": int(warm_up), "gap": int(gap), "dt": float(dt),
        "Du": float(DU), "Dv": float(DV), "n_regimes": int(len(regimes)),
        "regime_samples": np.array(regime_samples, dtype=np.int64),
        "regime_names": np.array([entry["name"] for entry in regime_settings]),
        "regime_F": np.array([entry["F"] for entry in regime_settings], dtype=np.float64),
        "regime_k": np.array([entry["k"] for entry in regime_settings], dtype=np.float64),
        "regime_coverage": np.array([entry["coverage"] for entry in regime_settings]),
        "regime_elongation_median": np.array([entry["elongation_median"]
                                              for entry in regime_settings]),
        "regime_elongation_max": np.array([entry["elongation_max"]
                                           for entry in regime_settings]),
        "regime_stick_cell_fraction": np.array([entry["stick_cell_fraction"]
                                                for entry in regime_settings]),
        "regime_states_with_sticks": np.array([entry["states_with_sticks"]
                                               for entry in regime_settings]),
        "regime_spacing": np.array([entry["spacing"] for entry in regime_settings]),
        "regime_half_life": np.array([entry["half_life"] for entry in regime_settings]),
    }
    return {"splits": splits, "mean": mean, "std": std, "settings": settings,
            "regimes": regime_settings}


def save_dataset(dataset, path):
    """Write the dataset as one ``.npz``."""
    arrays = {}
    for name, block in dataset["splits"].items():
        for key, value in block.items():
            arrays[f"{name}_{key}"] = value
    arrays["mean"] = dataset["mean"]
    arrays["std"] = dataset["std"]
    for key, value in dataset["settings"].items():
        arrays[key] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_dataset(path):
    """Read what :func:`save_dataset` wrote, as a plain dict of numpy arrays."""
    with np.load(path) as handle:
        arrays = {key: handle[key] for key in handle.files}

    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = {
            "inputs": np.asarray(arrays.pop(f"{name}_inputs"), dtype=np.float32),
            "targets": np.asarray(arrays.pop(f"{name}_targets"), dtype=np.float32),
            "raw_targets": np.asarray(arrays.pop(f"{name}_raw_targets"), dtype=np.float32),
            "regime": np.asarray(arrays.pop(f"{name}_regime"), dtype=np.int64),
        }
    integer_keys = ("N", "stride", "history", "seed", "n_samples", "n_spots", "warm_up",
                    "gap", "n_regimes")
    settings = {}
    for key in integer_keys:
        settings[key] = int(arrays.pop(key))
    if "regime_names" in arrays:
        settings["regime_names"] = np.asarray(arrays.pop("regime_names"))
    if "regime_samples" in arrays:
        settings["regime_samples"] = np.asarray(arrays.pop("regime_samples"), dtype=np.int64)
    for key in ("regime_F", "regime_k", "regime_coverage", "regime_elongation_median",
                "regime_elongation_max", "regime_stick_cell_fraction",
                "regime_states_with_sticks", "regime_spacing", "regime_half_life"):
        if key in arrays:
            settings[key] = np.asarray(arrays.pop(key), dtype=np.float64)
    for key in ("dt", "Du", "Dv"):
        if key in arrays:
            settings[key] = float(arrays.pop(key))
    return {"splits": splits,
            "mean": np.asarray(arrays["mean"], dtype=np.float64),
            "std": np.asarray(arrays["std"], dtype=np.float64),
            "settings": settings}


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Gray-Scott in two regimes and write the mixed dataset.")
    parser.add_argument("--out", default="data", help="folder to write the dataset into")
    parser.add_argument("--figures", default="figures", help="folder for the contact sheet")
    parser.add_argument("--explore", action="store_true",
                        help="walk the (F, k) plane and write figures/regimes.png instead of "
                             "building the dataset")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="grid size")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="model steps per emulator call")
    parser.add_argument("--main-f", type=float, default=MAIN_F,
                        help="the main regime's feed rate")
    parser.add_argument("--main-k", type=float, default=MAIN_K,
                        help="the main regime's drain excess")
    parser.add_argument("--stick-f", type=float, default=STICK_F,
                        help="the stick regime's feed rate")
    parser.add_argument("--stick-k", type=float, default=STICK_K,
                        help="the stick regime's drain excess")
    parser.add_argument("--samples", type=int, default=DEFAULT_MAIN_SAMPLES,
                        help="recorded states in the main regime's long run")
    parser.add_argument("--stick-samples", type=int, default=DEFAULT_STICK_SAMPLES,
                        help="recorded states in the stick regime's long run")
    parser.add_argument("--warm-up", type=int, default=2000,
                        help="steps discarded before the first recorded state")
    parser.add_argument("--gap", type=int, default=20,
                        help="recorded samples dropped between the splits")
    parser.add_argument("--spots", type=int, default=DEFAULT_SPOTS)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--steps", type=int, default=8000, help="explore: steps per (F, k)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.explore:
        explore(args.n, args.steps, args.spots, args.seed, Path(args.figures))
        return

    regimes = ((args.main_f, args.main_k), (args.stick_f, args.stick_k))
    regime_samples = (args.samples, args.stick_samples)
    print(f"\nintegrating {len(regimes)} regimes of {args.n} x {args.n} cells: main "
          f"{args.samples} states, stick {args.stick_samples} states, "
          f"{args.stride} steps each ...")
    dataset = make_dataset(args.samples, args.stride, n=args.n, regimes=regimes,
                           regime_samples=regime_samples, history=args.history,
                           n_spots=args.spots, warm_up=args.warm_up,
                           gap=args.gap, seed=args.seed)
    path = save_dataset(dataset, Path(args.out) / "gray_scott_mixed.npz")
    settings = dataset["settings"]

    print(f"  one emulator call advances the system by {settings['stride']} steps\n")
    print(f"  {'split':<12s}{'examples':>8s}{'main':>8s}{'sticks':>8s}{'shape of one input':>22s}")
    for name, block in dataset["splits"].items():
        n_spots = int((block["regime"] == 0).sum())
        n_sticks = int((block["regime"] == 1).sum())
        print(f"  {name:<12s}{len(block['inputs']):8d}{n_spots:8d}{n_sticks:8d}"
              f"{str(block['inputs'].shape[2:]):>22s}")
    print("\n  the patterns the dataset was drawn from, median over 20 sampled states:")
    for entry in dataset["regimes"]:
        print(f"    {entry['name']:<7s} F={entry['F']:.3f} k={entry['k']:.3f}: "
              f"coverage {entry['coverage']:.3f}, {entry['components']:.0f} components, "
              f"elongation median {entry['elongation_median']:.2f} / max "
              f"{entry['elongation_max']:.2f}, stick cells {entry['stick_cell_fraction']:.2f}, "
              f"states with a stick {entry['states_with_sticks']:.0%}")
    print(f"\n  standardisation: u {dataset['mean'][0]:+.3f} +- {dataset['std'][0]:.3f}, "
          f"v {dataset['mean'][1]:+.3f} +- {dataset['std'][1]:.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
