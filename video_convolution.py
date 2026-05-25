"""
video_convolution.py
A module: image/video-frame convolution effects.

This file provides one clear function for C/D modules:
    apply_video_effect(frame, effect_type, kernel_size, strength, iterations)

Supported effects:
    - "none"
    - "mean_blur"
    - "gaussian_blur"
    - "sharpen"

Input frame format:
    OpenCV BGR image, shape=(H, W, 3)
Output:
    Processed OpenCV BGR image, uint8
"""

from __future__ import annotations

import cv2
import numpy as np


# =========================================================
# 1. Core 2D convolution function
# =========================================================
def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a 2D convolution kernel to an image.

    This is the image-processing core of the project.
    cv2.filter2D applies the sliding-window weighted-sum operation.
    """
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


# =========================================================
# 2. Kernel generators
# =========================================================
def make_mean_blur_kernel(kernel_size: int = 5) -> np.ndarray:
    """Create a mean blur kernel."""
    kernel_size = _make_valid_odd_kernel_size(kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= kernel.sum()
    return kernel


def make_gaussian_blur_kernel(kernel_size: int = 5, sigma: float | None = None) -> np.ndarray:
    """Create a 2D Gaussian blur kernel."""
    kernel_size = _make_valid_odd_kernel_size(kernel_size)

    if sigma is None:
        # Common OpenCV-style heuristic
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    one_d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = one_d @ one_d.T
    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def make_sharpen_kernel(strength: float = 1.0) -> np.ndarray:
    """
    Create a sharpening kernel.

    strength controls how aggressive the sharpening is:
        0.0 -> almost original
        1.0 -> standard sharpening
        2.0 -> stronger sharpening
    """
    strength = max(0.0, float(strength))

    # Base Laplacian-enhancement sharpening:
    # center is positive, neighbors are negative.
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    identity = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )

    return identity + strength * kernel


def _make_valid_odd_kernel_size(kernel_size: int) -> int:
    """Ensure kernel size is an odd integer >= 3."""
    kernel_size = int(kernel_size)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


# =========================================================
# 3. Main frame-level interface for C/D modules
# =========================================================
def apply_video_effect(
    frame: np.ndarray,
    effect_type: str = "gaussian_blur",
    kernel_size: int = 5,
    strength: float = 1.0,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply a selected convolution effect to one video frame.

    Parameters
    ----------
    frame:
        One OpenCV video frame in BGR format.
    effect_type:
        "none", "mean_blur", "gaussian_blur", or "sharpen".
        For convenience, "blur" is treated as "gaussian_blur".
    kernel_size:
        Blur kernel size. Larger values produce stronger blur.
        Used by mean_blur and gaussian_blur.
    strength:
        Sharpen strength. Used mainly by sharpen.
    iterations:
        How many times to apply the same convolution effect.
        Useful for stronger blur/sharpening without changing the kernel.

    Returns
    -------
    np.ndarray:
        Processed frame, uint8 BGR.
    """
    if frame is None:
        raise ValueError("frame cannot be None")

    effect = (effect_type or "none").lower().strip()
    iterations = max(1, int(iterations))

    if effect in ["none", "original", "no_effect"]:
        return frame.copy()

    if effect in ["blur", "gaussian", "gaussian_blur"]:
        kernel = make_gaussian_blur_kernel(kernel_size)
    elif effect in ["mean", "mean_blur", "average", "average_blur"]:
        kernel = make_mean_blur_kernel(kernel_size)
    elif effect in ["sharpen", "sharp"]:
        kernel = make_sharpen_kernel(strength)
    else:
        raise ValueError(
            f"Unsupported effect_type: {effect_type}. "
            "Use 'none', 'mean_blur', 'gaussian_blur', or 'sharpen'."
        )

    processed = frame.copy()
    for _ in range(iterations):
        processed = apply_convolution(processed, kernel)

    return np.clip(processed, 0, 255).astype(np.uint8)


# =========================================================
# 4. Optional standalone test
# =========================================================
if __name__ == "__main__":
    input_image = "test_frame.jpg"
    img = cv2.imread(input_image)

    if img is None:
        print(f"Could not read {input_image}. Put a test image in this folder first.")
    else:
        blurred = apply_video_effect(img, effect_type="gaussian_blur", kernel_size=9)
        sharpened = apply_video_effect(img, effect_type="sharpen", strength=1.5)
        cv2.imwrite("test_blur.jpg", blurred)
        cv2.imwrite("test_sharpen.jpg", sharpened)
        print("Saved test_blur.jpg and test_sharpen.jpg")
