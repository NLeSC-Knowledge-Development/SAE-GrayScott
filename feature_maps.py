"""Show selected SAE feature maps alongside physical fields."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import laplacian, load_dataset
from train_emulator import load_emulator, token_activations
from train_sae import SparseAutoencoder


def feature_correlations(model, sae, layer, states, mean, std):
    """Correlate each feature with v and its Laplacian over the supplied states."""
    feature_sum = np.zeros(sae.n_latent)
    feature_sq = np.zeros(sae.n_latent)
    target_sum = np.zeros(2)
    target_sq = np.zeros(2)
    cross = np.zeros((2, sae.n_latent))
    count = 0
    for start in range(0, len(states), 4):
        batch = states[start:start + 4]
        tokens = token_activations(model, batch, layer)
        with torch.no_grad():
            features = sae.encode(tokens).numpy().astype(np.float64)
        v = batch[:, -1, 1] * std[1] + mean[1]
        targets = np.stack((v, np.stack([laplacian(field) for field in v])))
        targets = targets.reshape(2, -1).astype(np.float64)
        count += len(features)
        feature_sum += features.sum(axis=0)
        feature_sq += (features ** 2).sum(axis=0)
        target_sum += targets.sum(axis=1)
        target_sq += (targets ** 2).sum(axis=1)
        cross += targets @ features
    numerator = cross - target_sum[:, None] * feature_sum[None, :] / count
    feature_variance = np.maximum(feature_sq - feature_sum**2 / count, 0)
    target_variance = np.maximum(target_sq - target_sum**2 / count, 0)
    denominator = np.sqrt(target_variance[:, None] * feature_variance[None, :])
    result = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--fit-start", type=int, default=60)
    parser.add_argument("--fit-states", type=int, default=40)
    parser.add_argument("--eval-states", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.fit_start < 0 or min(args.fit_states, args.eval_states, args.threads) < 1:
        parser.error("state counts and threads must be positive; fit-start must be nonnegative")
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data)
    val = data["splits"]["val"]["inputs"]
    end_fit = args.fit_start + args.fit_states
    end_eval = end_fit + args.eval_states
    if end_eval > len(val):
        parser.error("requested states exceed the validation block")
    fit, evaluation = val[args.fit_start:end_fit], val[end_fit:end_eval]
    model_dir = Path(args.model_dir)
    model, _ = load_emulator(model_dir / "emulator.pt")
    checkpoint = torch.load(model_dir / "sae.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    layer = config["layer"]
    if layer not in model.sae_layer_names():
        raise ValueError(f"{layer!r} is not an emulator activation layer")
    sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()

    mean, std = data["mean"], data["std"]
    fit_corr = feature_correlations(model, sae, layer, fit, mean, std)
    eval_corr = feature_correlations(model, sae, layer, evaluation, mean, std)
    if not np.all(np.isfinite(fit_corr).any(axis=1)):
        raise ValueError("no defined correlations for at least one target")
    selected = np.nanargmax(np.abs(fit_corr), axis=1)
    if not np.all(np.isfinite(eval_corr[np.arange(2), selected])):
        raise ValueError("a selected feature has undefined evaluation correlation")
    sample = evaluation[:1]
    v = sample[0, -1, 1] * std[1] + mean[1]
    tokens = token_activations(model, sample, layer)
    with torch.no_grad():
        features = sae.encode(tokens).numpy().reshape(*v.shape, -1)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    targets = (v, laplacian(v))
    names = ("v", "Laplacian of v")
    for i, index in enumerate(selected):
        target_image = axes[i, 0].imshow(targets[i], cmap="inferno" if i == 0 else "coolwarm")
        fig.colorbar(target_image, ax=axes[i, 0], fraction=0.046)
        axes[i, 0].set_title(names[i])
        image = axes[i, 1].imshow(features[:, :, index], cmap="viridis")
        fig.colorbar(image, ax=axes[i, 1], fraction=0.046)
        axes[i, 1].set_title(f"feature {index}; held-out r = {eval_corr[i, index]:.3f}")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"{layer}: validation state {end_fit}; feature selection used earlier states")
    fig.tight_layout()
    output = Path(args.figures) / "sae_feature_maps.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Selected features: {selected.tolist()}; evaluation correlations: "
          f"{eval_corr[np.arange(2), selected].tolist()}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
