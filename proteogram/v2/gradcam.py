"""Grad-CAM residue-pair importance maps for proteogram similarity search.

This module is fully self-contained and has no dependency on Img2Vec internals.
It accepts a plain ``nn.Module`` embedding network, two preprocessed image
tensors, and produces a spatial heatmap whose (i, j) value reflects how much
the interaction between residues i and j influenced the cosine similarity
between query and target.

Because proteogram pixels directly encode pairwise residue interactions
(row i, column j → residue-i vs. residue-j), the Grad-CAM output is directly
interpretable as a residue-pair attribution map — a property unique to the
proteogram representation.

Typical usage
-------------
>>> from proteogram.v2.gradcam import GradCAM
>>> gcam = GradCAM(embed_net)               # stripped embedding network
>>> heatmap = gcam.compute(query_tensor, target_tensor)   # (H, W) float [0,1]
>>> gcam.save_figure(heatmap, query_img_np, cos_sim, "query", "target", "/out")
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_last_conv_sequential(net: nn.Module) -> Optional[nn.Module]:
    """Return the last ``nn.Sequential`` submodule containing a ``Conv2d``.

    This heuristic correctly identifies:
    - ``layer4`` for ResNet18 (last residual block, 512 output channels).
    - ``block4`` for the custom 4-block ConvNet (256 output channels).

    If no such block is found, falls back to the last ``Conv2d`` layer itself.

    Args:
        net: The embedding network (fc / classifier head already stripped).

    Returns:
        The target layer module, or ``None`` if the network has no Conv2d.
    """
    target = None

    # Walk direct children looking for Sequential blocks that contain Conv2d
    for child in net.children():
        if isinstance(child, nn.Sequential):
            has_conv = any(isinstance(m, nn.Conv2d) for m in child.modules())
            if has_conv:
                target = child

    if target is not None:
        return target

    # Fallback: last Conv2d anywhere in the network
    last_conv = None
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _pad_to_size(img: Image.Image, target: int = 200, fill: int = 128) -> np.ndarray:
    """Pad/crop a PIL image to ``target × target`` with ``fill`` colour.

    Matches the preprocessing applied in ``train_multiple_models.py`` and
    ``measure_similarity_v2.py`` so GradCAM inputs are on the same distribution
    as training inputs.

    Args:
        img:    Input PIL image (any mode; converted to RGB internally).
        target: Target side length in pixels.
        fill:   Constant fill value for padding (default 128 = mid-grey).

    Returns:
        ``uint8`` numpy array of shape ``(target, target, 3)``.
    """
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]

    def _pad(curr: int, tgt: int):
        d = tgt - curr
        if d <= 0:
            return (0, 0)
        p1 = d // 2
        return (p1, d - p1)

    padding = (_pad(H, target), _pad(W, target), (0, 0))
    arr = np.pad(arr, padding, constant_values=fill)
    return arr[:target, :target, :]


def _preprocess_image(image_path: str) -> torch.Tensor:
    """Load a proteogram JPG and return a normalised (1, 3, 200, 200) tensor.

    Uses the same pad-to-200 + ImageNet normalisation as the training pipeline.

    Args:
        image_path: Absolute or relative path to a proteogram JPG.

    Returns:
        Float tensor of shape ``(1, 3, 200, 200)`` ready for model inference.
    """
    from torchvision import transforms as T

    img = Image.open(image_path)
    arr = _pad_to_size(img, target=200)
    pil = Image.fromarray(arr)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil).unsqueeze(0)          # (1, 3, H, W)


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Cosine-similarity Grad-CAM for proteogram embedding networks.

    Differentiates the cosine similarity between a query and target embedding
    with respect to the activations of the last convolutional block.  The
    resulting heatmap identifies which spatial regions of the *query* proteogram
    most influenced the similarity score.

    Args:
        embed_net: The embedding sub-network (classification head removed).
                   Must be in eval mode and on the correct device.
        device:    Torch device string or object.  Defaults to ``"cpu"``.
    """

    def __init__(self, embed_net: nn.Module, device: str = "cpu") -> None:
        self.embed_net = embed_net
        self.device = torch.device(device)
        self._target_layer = _find_last_conv_sequential(embed_net)
        if self._target_layer is None:
            raise RuntimeError(
                "GradCAM: could not find a convolutional layer in embed_net.  "
                "Ensure the network contains at least one nn.Conv2d."
            )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        query_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap for one query→target pair.

        Args:
            query_tensor:  Preprocessed query image, shape ``(1, 3, H, W)``.
            target_tensor: Preprocessed target image, shape ``(1, 3, H, W)``.

        Returns:
            Float32 numpy array of shape ``(H, W)``, values in ``[0, 1]``.
            High values indicate residue pairs that most increased the cosine
            similarity between query and target.
        """
        activations: dict = {}
        gradients: dict = {}

        # Register hooks
        fwd_hook = self._target_layer.register_forward_hook(
            lambda m, inp, out: activations.update({"value": out.detach()})
        )
        bwd_hook = self._target_layer.register_full_backward_hook(
            lambda m, grad_inp, grad_out: gradients.update({"value": grad_out[0].detach()})
        )

        try:
            query  = query_tensor.to(self.device).float().requires_grad_(True)
            target = target_tensor.to(self.device).float()

            self.embed_net.eval()

            # Forward pass — activations hook fires here
            q_feat = self.embed_net(query)                       # (1, d)
            with torch.no_grad():
                t_feat = self.embed_net(target)                  # (1, d)

            # Cosine similarity as a scalar (not through nn.CosineSimilarity
            # so we can call .backward() directly)
            q_norm  = F.normalize(q_feat, dim=1)
            t_norm  = F.normalize(t_feat.detach(), dim=1)
            cos_sim = (q_norm * t_norm).sum()                    # scalar

            self.embed_net.zero_grad()
            cos_sim.backward()                                   # gradients hook fires here

        finally:
            fwd_hook.remove()
            bwd_hook.remove()

        # Grad-CAM: global-average-pool gradients → weight activation maps
        grads   = gradients["value"]                             # (1, C, h, w)
        acts    = activations["value"]                           # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)           # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True)          # (1, 1, h, w)
        cam = F.relu(cam)                                        # keep positive

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Upsample to query input size
        H, W = query_tensor.shape[2], query_tensor.shape[3]
        cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        return cam_up.squeeze().cpu().detach().numpy().astype(np.float32)

    def compute_from_paths(
        self,
        query_path: str,
        target_path: str,
    ) -> tuple[np.ndarray, float]:
        """Convenience wrapper: load images from disk, compute heatmap.

        Args:
            query_path:  Path to query proteogram JPG.
            target_path: Path to target proteogram JPG.

        Returns:
            Tuple of:
            - heatmap array (H, W), float32, values in [0, 1].
            - cosine similarity score (float).
        """
        q_tensor = _preprocess_image(query_path)
        t_tensor = _preprocess_image(target_path)

        # Compute cosine score separately (no grad) for the return value
        with torch.no_grad():
            q_emb = self.embed_net(q_tensor.to(self.device).float())
            t_emb = self.embed_net(t_tensor.to(self.device).float())
            cos_sim = float(
                F.cosine_similarity(q_emb, t_emb, dim=1).item()
            )

        heatmap = self.compute(q_tensor, t_tensor)
        return heatmap, cos_sim

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def save_figure(
        self,
        heatmap: np.ndarray,
        query_img: np.ndarray,
        cos_sim: float,
        query_name: str,
        target_name: str,
        output_dir: str,
        query_sequence: Optional[str] = None,
    ) -> str:
        """Save a 3-panel Grad-CAM figure (original | heatmap | overlay).

        Args:
            heatmap:        Float32 array (H, W), values in [0, 1].
            query_img:      ``uint8`` RGB array of the query proteogram.
            cos_sim:        Cosine similarity score to display in the title.
            query_name:     Protein ID of the query (used in file name & title).
            target_name:    Protein ID of the target.
            output_dir:     Directory where the PNG is written.
            query_sequence: Optional 1-letter amino acid sequence.  When
                            provided and ≤ 200 residues, residue tick labels are
                            added to the axes.

        Returns:
            Absolute path of the saved PNG file.
        """
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Grad-CAM:  {query_name}  →  {target_name}"
            f"   (cosine similarity = {cos_sim:.4f})",
            fontsize=13,
            y=1.01,
        )

        # Panel 1 — original query proteogram
        axes[0].imshow(query_img)
        axes[0].set_title("Query proteogram", fontsize=11)
        axes[0].axis("off")

        # Panel 2 — Grad-CAM heatmap
        im = axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title(
            "Grad-CAM heatmap\n(high = important residue pairs)", fontsize=11
        )
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3 — semi-transparent overlay
        axes[2].imshow(query_img)
        overlay = axes[2].imshow(heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")
        plt.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)

        # Optional residue-index tick labels
        if query_sequence and 0 < len(query_sequence) <= 200:
            step = max(1, len(query_sequence) // 20)
            ticks = list(range(0, len(query_sequence), step))
            labels = [f"{i}\n{query_sequence[i]}" for i in ticks]
            for ax in axes:
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, fontsize=6)
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels, fontsize=6)
                ax.tick_params(axis="both", length=2)

        plt.tight_layout()
        stem = f"{query_name}_vs_{target_name}"
        out_path = os.path.join(output_dir, f"{stem}_gradcam.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def save_npy(
        self,
        heatmap: np.ndarray,
        query_name: str,
        target_name: str,
        output_dir: str,
    ) -> str:
        """Save the raw heatmap array as a ``.npy`` file.

        Args:
            heatmap:     Float32 array (H, W).
            query_name:  Protein ID of the query.
            target_name: Protein ID of the target.
            output_dir:  Destination directory.

        Returns:
            Absolute path of the saved ``.npy`` file.
        """
        os.makedirs(output_dir, exist_ok=True)
        stem = f"{query_name}_vs_{target_name}"
        out_path = os.path.join(output_dir, f"{stem}_gradcam.npy")
        np.save(out_path, heatmap)
        return out_path
