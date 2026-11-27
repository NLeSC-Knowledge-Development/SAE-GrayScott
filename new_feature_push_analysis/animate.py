"""Animations for the two-regime study.

    ../../../.venv/bin/python animate.py --data data/gray_scott_mixed.npz --model-dir model \
                                      --figures figures

The animations that need no network are drawn from the equations:

* ``regimes_compare.gif`` -- the same seeded pattern at the main and stick feed rates;
* ``stick_perturb.gif``  -- a 12 x 3 stick of ``v`` at the main regime, beside the same state
  without it;
* ``disc_threshold.gif`` -- a disc of radius 4 at ``v = 0.10`` and ``v = 0.20``, the two sides
  of the measured threshold.

The animations that need the trained network are:

* ``rollout_main.gif``   -- the true system, the emulator and their difference from the same
  main-regime state;
* ``stick_feature.gif``  -- the selected stick feature on a stick-regime trajectory, beside the
  field and the stick mask;
* ``injection.gif``      -- the selected stick feature held on through the SAE at three sites,
  beside the same rollout with no added activation.

Every animation writes a companion JSON with the settings and the outcome it shows.
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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from generate_data import (ELONGATION_MIN, SPOT_THRESHOLD, advance, connected_components,
                           coverage, elongation_map, load_dataset, morphology, seed_pattern,
                           stick_mask)
from make_fig1 import disc_field
from stick_study import (FOOTPRINT_SIGMA, choose_sites, sae_pushed_rollout, stick_figure)
from control_study import random_pattern, scaled_pattern
from push_study import PUSH_STRENGTH, field_sticks, near_cells
from train_emulator import free_rollout, load_emulator, token_activations, truth_rollout
from train_sae import SparseAutoencoder

V_MIN, V_MAX = 0.0, 0.45
MARKER_COLOUR = "#39c5f0"
ACTIVATION_CMAP = LinearSegmentedColormap.from_list("black_yellow", ["black", "yellow"])
ACTIVATION_GAMMA = 3.0
DPI = 80


def save(animation, path, fps):
    """Write a GIF and report its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {fps} frames per second)")
    return path


def periodic_distance(n, a, b):
    d = abs(a - b) % n
    return min(d, n - d)


def max_elongation_near(field, row, col, radius=10):
    """The largest component elongation whose middle is within ``radius`` of a cell."""
    n = field.shape[0]
    best = 0.0
    for component in connected_components(field):
        centre_row, centre_col = component["centroid"]
        if np.hypot(periodic_distance(n, centre_row, row),
                    periodic_distance(n, centre_col, col)) <= radius:
            best = max(best, component["elongation"])
    return best


def empty_site(field, rng, radius=7, tries=300):
    """A cell with no component within ``radius``, or the middle of the grid."""
    n = field.shape[0]
    for _ in range(tries):
        row, col = int(rng.integers(0, n)), int(rng.integers(0, n))
        d_row = np.minimum(np.abs(np.arange(n) - row), n - np.abs(np.arange(n) - row))
        d_col = np.minimum(np.abs(np.arange(n) - col), n - np.abs(np.arange(n) - col))
        disc = np.hypot(d_row[:, None], d_col[None, :]) <= radius
        if field[disc].max() < SPOT_THRESHOLD:
            return row, col
    return n // 2, n // 2


def stick_perturbation(u, v, row, col, length=12, width=3, amplitude=0.5):
    """A horizontal stick of ``v`` with ``u`` lowered to match, on the periodic grid."""
    n = u.shape[0]
    d_row = np.minimum(np.abs(np.arange(n) - row), n - np.abs(np.arange(n) - row))
    d_col = np.minimum(np.abs(np.arange(n) - col), n - np.abs(np.arange(n) - col))
    inside = (d_row[:, None] <= width / 2) & (d_col[None, :] <= length / 2)
    u = u.copy()
    v = v.copy()
    u[inside] = 1.0 - amplitude
    v[inside] = amplitude
    return u, v


def collect_frames(u, v, F, k, dt, steps, every):
    """Integrate and return the field ``v`` every ``every`` steps, the initial state first."""
    frames = [v.copy()]
    for _ in range(steps // every):
        u, v = advance(u, v, every, F, k, dt)
        frames.append(v.copy())
    return np.stack(frames)


# ------------------------------------------------------------------- equations only


def regimes_compare(settings, out_path, seed=0, warm_up=10000, steps=4000, every=40, fps=10):
    """The same seeded pattern at the two feed rates, side by side."""
    n = settings["N"]
    dt = settings["dt"]
    # The same seed on purpose: the initial condition is the control, F is the only change.
    u, v = seed_pattern(n, settings["n_spots"], np.random.default_rng(seed))
    u, v = advance(u, v, warm_up, settings["regime_F"][0], settings["regime_k"][0], dt)
    main_frames = collect_frames(u, v, settings["regime_F"][0], settings["regime_k"][0],
                                 dt, steps, every)
    u, v = seed_pattern(n, settings["n_spots"], np.random.default_rng(seed))
    u, v = advance(u, v, warm_up, settings["regime_F"][1], settings["regime_k"][1], dt)
    stick_frames = collect_frames(u, v, settings["regime_F"][1], settings["regime_k"][1],
                                  dt, steps, every)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6))
    fig.set_dpi(DPI)
    images = [axis.imshow(frames[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX)
              for axis, frames in zip(axes, (main_frames, stick_frames))]
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

    def draw(index):
        for image, frames in zip(images, (main_frames, stick_frames)):
            image.set_data(frames[index])
        main_shape = morphology(main_frames[index])
        stick_shape = morphology(stick_frames[index])
        axes[0].set_title(f"main: $F$ = {settings['regime_F'][0]}, $k$ = "
                          f"{settings['regime_k'][0]}\n"
                          f"coverage {main_shape['coverage']:.2f}, max elongation "
                          f"{main_shape['elongation_max']:.1f}", fontsize=9)
        axes[1].set_title(f"sticks: $F$ = {settings['regime_F'][1]}, $k$ = "
                          f"{settings['regime_k'][1]}\n"
                          f"coverage {stick_shape['coverage']:.2f}, max elongation "
                          f"{stick_shape['elongation_max']:.1f}", fontsize=9)
        label.set_text(f"step {index * every:,}; one call is {settings['stride']} steps")
        return images

    animation = FuncAnimation(fig, draw, frames=len(main_frames),
                              interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(animation, out_path, fps)
    plt.close(fig)
    return {
        "gif": out_path.name, "same_seed": int(seed), "warm_up_steps": int(warm_up),
        "animated_steps": int(steps), "steps_per_frame": int(every),
        "frames": int(len(main_frames)), "fps": int(fps),
        "main_final_coverage": float(coverage(main_frames[-1])),
        "stick_final_coverage": float(coverage(stick_frames[-1])),
    }


def stick_perturb(settings, out_path, seed=0, warm_up=10000, steps=1200, every=60, fps=6):
    """A stick-shaped perturbation at the main regime, beside the same state without it."""
    n = settings["N"]
    F = settings["regime_F"][0]
    k = settings["regime_k"][0]
    dt = settings["dt"]
    u, v = seed_pattern(n, settings["n_spots"], np.random.default_rng(seed))
    u, v = advance(u, v, warm_up, F, k, dt)
    row, col = empty_site(v, np.random.default_rng(seed + 99))
    control_u, control_v = u.copy(), v.copy()
    pushed_u, pushed_v = stick_perturbation(u, v, row, col)
    control_frames = collect_frames(control_u, control_v, F, k, dt, steps, every)
    pushed_frames = collect_frames(pushed_u, pushed_v, F, k, dt, steps, every)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6))
    fig.set_dpi(DPI)
    images = [axis.imshow(frames[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX)
              for axis, frames in zip(axes, (control_frames, pushed_frames))]
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.plot([col], [row], marker="o", markersize=12, markerfacecolor="none",
                  markeredgecolor=MARKER_COLOUR, markeredgewidth=2.0)
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

    def draw(index):
        for image, frames in zip(images, (control_frames, pushed_frames)):
            image.set_data(frames[index])
        control_elongation = max_elongation_near(control_frames[index], row, col)
        pushed_elongation = max_elongation_near(pushed_frames[index], row, col)
        axes[0].set_title(f"without the stick\nmax elongation near the site "
                          f"{control_elongation:.2f}", fontsize=9)
        axes[1].set_title(f"with a 12 x 3 stick of $v$ = 0.5\nmax elongation near the site "
                          f"{pushed_elongation:.2f}", fontsize=9)
        label.set_text(f"step {index * every:,} at $F$ = {F}, $k$ = {k}; "
                       f"the marked cell is the middle of the stick")
        return images

    animation = FuncAnimation(fig, draw, frames=len(control_frames),
                              interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    save(animation, out_path, fps)
    plt.close(fig)
    return {
        "gif": out_path.name, "row": int(row), "col": int(col),
        "warm_up_steps": int(warm_up), "animated_steps": int(steps),
        "steps_per_frame": int(every), "frames": int(len(control_frames)), "fps": int(fps),
        "control_final_max_elongation": float(max_elongation_near(control_frames[-1], row, col)),
        "pushed_final_max_elongation": float(max_elongation_near(pushed_frames[-1], row, col)),
    }


def disc_threshold(settings, out_path, radius=4, amplitudes=(0.10, 0.20), steps=900, every=15,
                   fps=10):
    """A disc below and above the measured threshold, side by side."""
    n = settings["N"]
    F = settings["regime_F"][0]
    k = settings["regime_k"][0]
    dt = settings["dt"]
    frames = []
    for amplitude in amplitudes:
        u, v = disc_field(n, radius, amplitude)
        frames.append(collect_frames(u, v, F, k, dt, steps, every))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    fig.set_dpi(DPI)
    images = [axis.imshow(stack[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX)
              for axis, stack in zip(axes, frames)]
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

    def draw(index):
        for image, stack in zip(images, frames):
            image.set_data(stack[index])
        for axis, amplitude, stack in zip(axes, amplitudes, frames):
            axis.set_title(f"a disc of $v$ = {amplitude:.2f}\n"
                           f"coverage {coverage(stack[index]):.3f}", fontsize=9)
        label.set_text(f"radius {radius}, {index * every:,} steps at $F$ = {F}, $k$ = {k}")
        return images

    animation = FuncAnimation(fig, draw, frames=len(frames[0]),
                              interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(animation, out_path, fps)
    plt.close(fig)
    return {
        "gif": out_path.name, "radius": int(radius), "amplitudes": list(amplitudes),
        "steps": int(steps), "steps_per_frame": int(every),
        "frames": int(len(frames[0])), "fps": int(fps),
        "final_coverage": [float(coverage(stack[-1])) for stack in frames],
    }


# ------------------------------------------------------------------- with the network


def rollout_main(model, settings, dataset, mean, std, out_path, calls=30, start=0, fps=8):
    """The true system, the emulator and their difference from the same main-regime state."""
    block = dataset["splits"]["test"]
    states = block["inputs"][block["regime"] == 0][start:start + 1]
    if len(states) == 0:
        raise ValueError("the test split holds no main-regime states")
    predicted = free_rollout(model, states, calls, np.array([0]))
    physical = states * std.reshape(1, 1, 2, 1, 1) + mean.reshape(1, 1, 2, 1, 1)
    truth = truth_rollout(physical, calls, settings["stride"], settings["regime_F"][0],
                          settings["regime_k"][0], settings["dt"])
    emulator_v = predicted[:, 0, 1] * std[1] + mean[1]
    truth_v = truth[:, 0, 1]
    difference = emulator_v - truth_v
    span = float(np.abs(difference).max())
    span = span if span > 0 else 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    fig.set_dpi(DPI)
    images = [
        axes[0].imshow(truth_v[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX),
        axes[1].imshow(emulator_v[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX),
        axes[2].imshow(difference[0], cmap="coolwarm", vmin=-span, vmax=span),
    ]
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.04).set_label("v", fontsize=8)
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

    def draw(index):
        images[0].set_data(truth_v[index])
        images[1].set_data(emulator_v[index])
        images[2].set_data(difference[index])
        error = float(np.sqrt(np.mean(difference[index] ** 2)))
        axes[0].set_title("the true system", fontsize=10)
        axes[1].set_title(f"the emulator\nerror {error:.3f}", fontsize=10)
        axes[2].set_title("emulator \u2212 truth", fontsize=10)
        label.set_text(f"{index} calls; one call is {settings['stride']} steps, the main regime")
        return images

    animation = FuncAnimation(fig, draw, frames=calls + 1, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(animation, out_path, fps)
    plt.close(fig)


def stick_feature(model, sae, layer, settings, dataset, feature, out_path, start=0, count=40,
                  fps=8):
    """The selected feature on a stick-regime trajectory, beside ``v`` and the stick mask."""
    block = dataset["splits"]["test"]
    pool = block["inputs"][block["regime"] == 1]
    if len(pool) == 0:
        raise ValueError("the test split holds no stick-regime states")
    if start == 0:
        for index, candidate in enumerate(pool):
            field = candidate[-1, 1] * dataset["std"][1] + dataset["mean"][1]
            if stick_mask(field).any():
                start = index
                break
    states = pool[start:start + count]
    regime = np.ones(len(states), dtype=np.int64)
    tokens = token_activations(model, states, layer, regime)
    with torch.no_grad():
        activation = sae.encode(tokens).reshape(len(states), settings["N"], settings["N"],
                                                -1)[:, :, :, feature].numpy()
    v = states[:, -1, 1] * dataset["std"][1] + dataset["mean"][1]
    masks = np.stack([stick_mask(field) for field in v]).astype(np.float64)
    high = max(float(activation.max()), 1e-6)
    norm = PowerNorm(gamma=ACTIVATION_GAMMA, vmin=0.0, vmax=high)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    fig.set_dpi(DPI)
    images = [
        axes[0].imshow(v[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX),
        axes[1].imshow(masks[0], cmap="gray", vmin=0.0, vmax=1.0),
        axes[2].imshow(activation[0], cmap=ACTIVATION_CMAP, norm=norm),
    ]
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    activation_bar = fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.03)
    activation_bar.set_label(f"feature activation (power $\\gamma$ = {ACTIVATION_GAMMA:g})",
                             fontsize=8)
    activation_bar.ax.tick_params(labelsize=7)
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

    def draw(index):
        images[0].set_data(v[index])
        images[1].set_data(masks[index])
        images[2].set_data(activation[index])
        axes[0].set_title("the field $v$", fontsize=10)
        axes[1].set_title("white = cells with $E$ ≥ 3", fontsize=10)
        axes[2].set_title(f"SAE feature {feature}", fontsize=10)
        label.set_text(f"stick regime, state {start + index}; one state is "
                       f"{settings['stride']} model steps apart")
        return images

    animation = FuncAnimation(fig, draw, frames=len(v), interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(animation, out_path, fps)
    plt.close(fig)


def pair_animation(control_v, pushed_v, site_list, titles, note, out_path, calls, fps):
    """Three panels: the control field, the pushed field and their difference."""
    difference = pushed_v - control_v
    span = float(np.abs(difference).max())
    span = span if span > 0 else 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    fig.set_dpi(DPI)
    images = [
        axes[0].imshow(control_v[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX),
        axes[1].imshow(pushed_v[0], cmap="inferno", vmin=V_MIN, vmax=V_MAX),
        axes[2].imshow(difference[0], cmap="coolwarm", vmin=-span, vmax=span),
    ]
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for row, col in site_list:
            axis.plot([col], [row], marker="o", markersize=10, markerfacecolor="none",
                      markeredgecolor=MARKER_COLOUR, markeredgewidth=1.8)
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, label="$\\Delta v$")
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="#444444")

    def draw(index):
        for image, frames in zip(images, (control_v, pushed_v, difference)):
            image.set_data(frames[index])
        label.set_text(f"call {index + 1}; {note}; the marked cells are the sites")
        return images

    animation = FuncAnimation(fig, draw, frames=calls, interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(animation, out_path, fps)
    plt.close(fig)
    return out_path


def injection(model, sae, layer, settings, dataset, feature, out_path, calls=20, sites=3,
              seed=0, fps=8, candidates=12, push_strength=PUSH_STRENGTH):
    """The stick feature held on through the SAE at three sites, beside the same rollout with
    no added activation.

    The pair shown is the candidate with the largest final near-site stick share out of
    ``candidates`` states. The caption says which state it is, so the example is not silently
    chosen by eye.
    """
    block = dataset["splits"]["test"]
    main_states = block["inputs"][block["regime"] == 0]
    stick_states = block["inputs"][block["regime"] == 1]
    if len(main_states) == 0 or len(stick_states) == 0:
        raise ValueError("the test split must hold both regimes")
    rng = np.random.default_rng(seed)
    count = min(candidates, len(main_states))
    best = None
    best_gain = -np.inf
    for attempt in range(count):
        state = main_states[attempt]
        site_list = choose_sites(rng, settings["N"], sites)
        amplitude = np.zeros((settings["N"], settings["N"]), dtype=np.float64)
        for row, col in site_list:
            site_amplitude, _, _ = stick_figure(model, sae, layer, stick_states, 1, feature,
                                                row, col, rng, dataset["mean"], dataset["std"])
            amplitude = np.maximum(amplitude, site_amplitude)
        scale = float(token_activations(model, state[None], layer,
                                        np.array([0])).norm(dim=1).mean())
        pushed = sae_pushed_rollout(model, layer, sae, feature, state, 0,
                                    push_strength * scale, amplitude, calls)
        control = sae_pushed_rollout(model, layer, sae, feature, state, 0,
                                     0.0, amplitude, calls)
        pushed_v = pushed[:, 1] * dataset["std"][1] + dataset["mean"][1]
        control_v = control[:, 1] * dataset["std"][1] + dataset["mean"][1]
        near = near_cells((settings["N"], settings["N"]), site_list, 8)
        pushed_share, _, _ = field_sticks(pushed_v[-1], near)
        control_share, _, _ = field_sticks(control_v[-1], near)
        gain = pushed_share - control_share
        if gain > best_gain:
            best_gain = gain
            best = (pushed_v, control_v, site_list, attempt, pushed_share, control_share)
    pushed_v, control_v, site_list, attempt, pushed_share, control_share = best
    chosen_by = (f"the largest final near-site stick share out of {count} candidate states "
                 f"(state {attempt}: {pushed_share:.0%} with the push, "
                 f"{control_share:.0%} without)")

    pair_animation(control_v, pushed_v, site_list,
                   ("without the push", "with the push", "push \u2212 control"),
                   f"stick feature {feature} at {sites} sites", out_path, calls, fps)
    return {"gif": out_path.name, "chosen_by": chosen_by,
            "chosen_state": int(attempt), "sites": [list(site) for site in site_list],
            "calls": int(calls)}


def v_injection(model, sae, layer, settings, dataset, feature, out_path, control="random",
                calls=20, sites=3, seed=0, fps=8, candidates=12,
                push_strength=PUSH_STRENGTH):
    """A matched v addition at the same sites, beside the same SAE control rollout.

    The delta is matched to the RMS change in v the feature push makes on its first call. The
    pair shown is the candidate with the largest final near-site stick share out of
    ``candidates`` states.
    """
    block = dataset["splits"]["test"]
    main_states = block["inputs"][block["regime"] == 0]
    stick_states = block["inputs"][block["regime"] == 1]
    if len(main_states) == 0 or len(stick_states) == 0:
        raise ValueError("the test split must hold both regimes")
    rng = np.random.default_rng(seed)
    pattern_rng = np.random.default_rng(seed + 100)
    count = min(candidates, len(main_states))
    if control == "random":
        titles = ("without the v addition", "with random v", "v addition \u2212 control")
    else:
        titles = ("without the v addition", "with the footprint v", "v addition \u2212 control")
    best = None
    best_gain = -np.inf
    for attempt in range(count):
        state = main_states[attempt]
        site_list = choose_sites(rng, settings["N"], sites)
        amplitude = np.zeros((settings["N"], settings["N"]), dtype=np.float64)
        for row, col in site_list:
            site_amplitude, _, _ = stick_figure(model, sae, layer, stick_states, 1, feature,
                                                row, col, rng, dataset["mean"], dataset["std"])
            amplitude = np.maximum(amplitude, site_amplitude)
        scale = float(token_activations(model, state[None], layer,
                                        np.array([0])).norm(dim=1).mean())
        feature_pushed = sae_pushed_rollout(model, layer, sae, feature, state, 0,
                                            push_strength * scale, amplitude, 1)
        feature_control = sae_pushed_rollout(model, layer, sae, feature, state, 0,
                                             0.0, amplitude, 1)
        near = near_cells((settings["N"], settings["N"]), site_list, 8)
        target = float(np.sqrt(np.mean((feature_pushed[0, 1] - feature_control[0, 1])[near] ** 2)))
        support = amplitude > 0
        if control == "random":
            delta = random_pattern(pattern_rng, support, FOOTPRINT_SIGMA, near, target)
        else:
            delta = scaled_pattern(amplitude, near, target)
        pushed = sae_pushed_rollout(model, layer, sae, feature, state, 0,
                                    0.0, amplitude, calls, field_delta=delta)
        control_fields = sae_pushed_rollout(model, layer, sae, feature, state, 0,
                                            0.0, amplitude, calls)
        pushed_v = pushed[:, 1] * dataset["std"][1] + dataset["mean"][1]
        control_v = control_fields[:, 1] * dataset["std"][1] + dataset["mean"][1]
        pushed_share, _, _ = field_sticks(pushed_v[-1], near)
        control_share, _, _ = field_sticks(control_v[-1], near)
        gain = pushed_share - control_share
        if gain > best_gain:
            best_gain = gain
            best = (pushed_v, control_v, site_list, attempt, pushed_share, control_share)
    pushed_v, control_v, site_list, attempt, pushed_share, control_share = best
    chosen_by = (f"the largest final near-site stick share out of {count} candidate states "
                 f"(state {attempt}: {pushed_share:.0%} with the {control} v addition, "
                 f"{control_share:.0%} without)")
    note = (f"random v at {sites} sites" if control == "random"
            else f"v in the same footprint at {sites} sites")
    pair_animation(control_v, pushed_v, site_list, titles, note, out_path, calls, fps)
    return {"gif": out_path.name, "control": control, "chosen_by": chosen_by,
            "chosen_state": int(attempt), "sites": [list(site) for site in site_list],
            "calls": int(calls)}


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Draw the two-regime animations.")
    parser.add_argument("--data", default="data/gray_scott_mixed.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--injection-calls", type=int, default=20)
    parser.add_argument("--push-strength", type=float, default=PUSH_STRENGTH,
                        help="latent push per call, in mean token norms")
    parser.add_argument("--v-control", choices=("random", "footprint"), default="footprint",
                        help="pattern for the v-addition control animation")
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--only", default=None,
                        help="regimes, stick-perturb, disc, rollout, feature, injection, "
                             "v-injection (comma-separated)")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    figures = Path(args.figures)
    wanted = (set(args.only.split(",")) if args.only
              else {"regimes", "stick-perturb", "disc", "rollout", "feature", "injection",
                    "v-injection"})

    print(f"\nanimations at {settings['N']} x {settings['N']} cells, "
          f"main F = {settings['regime_F'][0]}, sticks F = {settings['regime_F'][1]}\n")
    if "regimes" in wanted:
        record = regimes_compare(settings, figures / "regimes_compare.gif", seed=args.seed,
                                 fps=args.fps)
        (figures / "regimes_compare.json").write_text(json.dumps(record, indent=2))
    if "stick-perturb" in wanted:
        record = stick_perturb(settings, figures / "stick_perturb.gif", seed=args.seed,
                               fps=min(args.fps, 6))
        (figures / "stick_perturb.json").write_text(json.dumps(record, indent=2))
    if "disc" in wanted:
        record = disc_threshold(settings, figures / "disc_threshold.gif", fps=args.fps)
        (figures / "disc_threshold.json").write_text(json.dumps(record, indent=2))

    needs_model = {"rollout", "feature", "injection", "v-injection"} & wanted
    if needs_model:
        model_path = Path(args.model_dir) / f"{args.model_name}.pt"
        sae_path = Path(args.model_dir) / "sae.pt"
        if not model_path.exists():
            raise FileNotFoundError("train_emulator.py has not been run, so there is no model "
                                    "to animate")
        model, _ = load_emulator(model_path)
        if ({"feature", "injection", "v-injection"} & wanted) and not sae_path.exists():
            raise FileNotFoundError("train_sae.py has not been run, so there is no feature "
                                    "to animate")
        sae = None
        layer = None
        if sae_path.exists():
            checkpoint = torch.load(sae_path, weights_only=False)
            config = checkpoint["config"]
            sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
            sae.load_state_dict(checkpoint["state_dict"])
            sae.eval()
            layer = config["layer"]
        if "rollout" in wanted:
            rollout_main(model, settings, dataset, mean, std,
                         figures / "rollout_main.gif", calls=args.calls, fps=args.fps)
        if {"feature", "injection", "v-injection"} & wanted:
            component_path = Path(args.model_dir) / "stick_components.json"
            stick_path = Path(args.model_dir) / "stick.json"
            if component_path.exists():
                selection = json.loads(component_path.read_text())["selected"]
                feature = int(selection["stick_feature"])
            elif stick_path.exists():
                feature = int(json.loads(stick_path.read_text())["select"]["stick_feature"])
            else:
                raise FileNotFoundError("run component_study.py or stick_study.py first")
            if "feature" in wanted:
                stick_feature(model, sae, layer, settings, dataset, feature,
                              figures / "stick_feature.gif", fps=args.fps)
            if "injection" in wanted:
                record = injection(model, sae, layer, settings, dataset, feature,
                                   figures / "injection.gif", calls=args.injection_calls,
                                   sites=args.sites, seed=args.seed, fps=args.fps,
                                   push_strength=args.push_strength)
                (figures / "injection.json").write_text(json.dumps(record, indent=2))
            if "v-injection" in wanted:
                record = v_injection(model, sae, layer, settings, dataset, feature,
                                     figures / "v_injection.gif", control=args.v_control,
                                     calls=args.injection_calls, sites=args.sites,
                                     seed=args.seed, fps=args.fps,
                                     push_strength=args.push_strength)
                (figures / "v_injection.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
