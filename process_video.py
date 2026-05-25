"""
process_video.py
C module: system integration pipeline.

This script connects:
    A: video_convolution.py             -> frame convolution blur/sharpen
    B: audio_convolution_advanced.py    -> 1D convolution audio denoising
    D: Gradio/Streamlit UI              -> process_video_for_ui(...)

Main capability:
    Input video -> video blur/sharpen + audio denoise -> output video
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import librosa
import numpy as np
import soundfile as sf

# MoviePy import is different across versions, so keep this fallback.
try:
    from moviepy import VideoFileClip, AudioFileClip
except ImportError:  # MoviePy v1 style
    from moviepy.editor import VideoFileClip, AudioFileClip

from video_convolution import apply_video_effect
from audio_convolution_advanced import denoise_audio_by_convolution


# =========================================================
# 0. Utilities
# =========================================================
def ensure_parent_dir(file_path: str | os.PathLike) -> None:
    """Create parent folder if needed."""
    parent = Path(file_path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _with_audio(video_clip: VideoFileClip, audio_clip: AudioFileClip):
    """Compatibility wrapper for MoviePy v1/v2."""
    if hasattr(video_clip, "with_audio"):
        return video_clip.with_audio(audio_clip)
    return video_clip.set_audio(audio_clip)


# =========================================================
# 1. Extract audio
# =========================================================
def extract_audio_from_video(
    input_video_path: str,
    output_audio_path: str,
    audio_fps: int = 44100,
) -> bool:
    """
    Extract audio from video.

    Returns True if the input video has audio, otherwise False.
    """
    ensure_parent_dir(output_audio_path)
    clip = VideoFileClip(input_video_path)

    try:
        if clip.audio is None:
            return False

        clip.audio.write_audiofile(
            output_audio_path,
            fps=audio_fps,
            codec="pcm_s16le",
            logger=None,
        )
        return True
    finally:
        clip.close()


# =========================================================
# 2. Process video frames using A module
# =========================================================
def process_video_frames(
    input_video_path: str,
    output_video_path: str,
    effect_type: str = "gaussian_blur",
    kernel_size: int = 5,
    visual_strength: float = 1.0,
    iterations: int = 1,
    grayscale: bool = False,
    progress: bool = True,
) -> None:
    """
    Read video frame by frame, apply convolution effect, and save silent video.
    """
    ensure_parent_dir(output_video_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_video_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = apply_video_effect(
                frame=frame,
                effect_type=effect_type,
                kernel_size=kernel_size,
                strength=visual_strength,
                iterations=iterations,
            )

            if grayscale:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            writer.write(processed)
            frame_count += 1

            if progress and frame_count % 60 == 0:
                print(f"Processed {frame_count} frames...")
    finally:
        cap.release()
        writer.release()

    if progress:
        print(f"Video frame processing complete. Total frames: {frame_count}")


# =========================================================
# 3. Process audio using B module
# =========================================================
def process_audio_track(
    input_audio_path: str,
    output_audio_path: str,
    enable_denoise: bool = True,
    cutoff_hz: float = 3000.0,
    num_taps: int = 101,
    denoise_strength: float = 1.0,
    pitch_shift_steps: float = 0.0,
    target_sr: int | None = None,
) -> None:
    """
    Load audio, optionally denoise it by 1D convolution, optionally pitch-shift,
    and write the processed audio file.
    """
    ensure_parent_dir(output_audio_path)

    # mono=False preserves stereo/multichannel as shape=(channels, samples).
    y, sr = librosa.load(input_audio_path, sr=target_sr, mono=False)

    def process_one_channel(channel: np.ndarray) -> np.ndarray:
        out = channel.astype(np.float32)

        if enable_denoise:
            out = denoise_audio_by_convolution(
                out,
                sample_rate=sr,
                cutoff_hz=cutoff_hz,
                num_taps=num_taps,
                strength=denoise_strength,
                use_manual=False,
                normalize_output=False,
            )

        if abs(float(pitch_shift_steps)) > 1e-8:
            out = librosa.effects.pitch_shift(
                y=out,
                sr=sr,
                n_steps=float(pitch_shift_steps),
            ).astype(np.float32)

        return np.clip(out, -1.0, 1.0).astype(np.float32)

    if y.ndim == 1:
        processed = process_one_channel(y)
        sf.write(output_audio_path, processed, sr)
    else:
        channels = [process_one_channel(ch) for ch in y]
        min_len = min(len(ch) for ch in channels)
        channels = [ch[:min_len] for ch in channels]
        processed = np.stack(channels, axis=1)  # soundfile expects samples x channels
        sf.write(output_audio_path, processed, sr)


# =========================================================
# 4. Merge processed audio and video
# =========================================================
def merge_audio_and_video(
    input_video_path: str,
    input_audio_path: str | None,
    output_video_path: str,
) -> None:
    """Combine processed video and processed audio into final MP4."""
    ensure_parent_dir(output_video_path)

    video_clip = VideoFileClip(input_video_path)
    audio_clip = None
    final_clip = None

    try:
        if input_audio_path is None:
            # No audio track: just export the processed video.
            video_clip.write_videofile(
                output_video_path,
                codec="libx264",
                audio=False,
                logger=None,
            )
        else:
            audio_clip = AudioFileClip(input_audio_path)
            final_clip = _with_audio(video_clip, audio_clip)
            final_clip.write_videofile(
                output_video_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
    finally:
        if final_clip is not None:
            final_clip.close()
        if audio_clip is not None:
            audio_clip.close()
        video_clip.close()


# =========================================================
# 5. Main project pipeline
# =========================================================
def process_video_pipeline(
    input_video_path: str,
    output_video_path: str = "outputs/final_output.mp4",
    effect_type: str = "gaussian_blur",
    kernel_size: int = 5,
    visual_strength: float = 1.0,
    iterations: int = 1,
    enable_denoise: bool = True,
    cutoff_hz: float = 3000.0,
    num_taps: int = 101,
    denoise_strength: float = 1.0,
    pitch_shift_steps: float = 0.0,
    grayscale: bool = False,
    keep_temp_files: bool = False,
) -> str:
    """
    Complete system-integration function.

    Parameters are intentionally UI-friendly so that D can directly connect them
    to sliders/dropdowns/checkboxes in Gradio or Streamlit.

    Returns
    -------
    str:
        Path to final processed video.
    """
    input_video_path = str(input_video_path)
    output_video_path = str(output_video_path)

    if not Path(input_video_path).exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    ensure_parent_dir(output_video_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        temp_video = str(tmp_dir / "processed_video_no_audio.mp4")
        temp_audio = str(tmp_dir / "original_audio.wav")
        temp_processed_audio = str(tmp_dir / "processed_audio.wav")

        # A module: frame-by-frame visual convolution.
        process_video_frames(
            input_video_path=input_video_path,
            output_video_path=temp_video,
            effect_type=effect_type,
            kernel_size=kernel_size,
            visual_strength=visual_strength,
            iterations=iterations,
            grayscale=grayscale,
            progress=True,
        )

        # Extract and process audio only if the video has an audio track.
        has_audio = extract_audio_from_video(input_video_path, temp_audio)
        if has_audio:
            process_audio_track(
                input_audio_path=temp_audio,
                output_audio_path=temp_processed_audio,
                enable_denoise=enable_denoise,
                cutoff_hz=cutoff_hz,
                num_taps=num_taps,
                denoise_strength=denoise_strength,
                pitch_shift_steps=pitch_shift_steps,
            )
            audio_for_merge = temp_processed_audio
        else:
            print("Input video has no audio track. Output video will be silent.")
            audio_for_merge = None

        # C module: final recombination.
        merge_audio_and_video(
            input_video_path=temp_video,
            input_audio_path=audio_for_merge,
            output_video_path=output_video_path,
        )

        if keep_temp_files:
            debug_dir = Path("debug_outputs")
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(temp_video, debug_dir / "processed_video_no_audio.mp4")
            if has_audio:
                shutil.copy(temp_audio, debug_dir / "original_audio.wav")
                shutil.copy(temp_processed_audio, debug_dir / "processed_audio.wav")

    return output_video_path


# =========================================================
# 6. UI interface for D module
# =========================================================
def process_video_for_ui(
    input_video_path: str,
    effect_type: str = "gaussian_blur",
    kernel_size: int = 5,
    visual_strength: float = 1.0,
    enable_denoise: bool = True,
    denoise_strength: float = 1.0,
    cutoff_hz: float = 3000.0,
    pitch_shift_steps: float = 0.0,
) -> str:
    """
    Simple wrapper for Gradio/Streamlit.

    Suggested UI controls:
        - file upload -> input_video_path
        - dropdown -> effect_type: none / gaussian_blur / mean_blur / sharpen
        - slider -> kernel_size: 3,5,7,9,11,15
        - slider -> visual_strength: 0.2 to 2.0
        - checkbox -> enable_denoise
        - slider -> denoise_strength: 0.0 to 1.0
        - slider -> cutoff_hz: 1000 to 8000
        - slider -> pitch_shift_steps: -6 to +6, optional extension
    """
    unique_name = f"processed_{uuid4().hex[:8]}.mp4"
    output_path = str(Path("outputs") / unique_name)

    return process_video_pipeline(
        input_video_path=input_video_path,
        output_video_path=output_path,
        effect_type=effect_type,
        kernel_size=kernel_size,
        visual_strength=visual_strength,
        iterations=1,
        enable_denoise=enable_denoise,
        cutoff_hz=cutoff_hz,
        num_taps=101,
        denoise_strength=denoise_strength,
        pitch_shift_steps=pitch_shift_steps,
        grayscale=False,
        keep_temp_files=False,
    )


# =========================================================
# 7. Command-line test entry
# =========================================================
if __name__ == "__main__":
    # Change this to your test video path.
    input_video = "input.mp4"

    result = process_video_pipeline(
        input_video_path=input_video,
        output_video_path="output/result.mp4",
        effect_type="sharpen",  # none / gaussian_blur / sharpen / mean_blur
        kernel_size=7,
        visual_strength=1.0,
        enable_denoise=True,
        cutoff_hz=3000.0,
        denoise_strength=1.0,
        pitch_shift_steps=0.0, # no pitch shift by default, could be added and selected.
    )

    print(f"Done: {result}")
