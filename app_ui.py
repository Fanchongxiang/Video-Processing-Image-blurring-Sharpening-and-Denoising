import gradio as gr
import os
import shutil
from pathlib import Path
from uuid import uuid4

from process_video import process_video_for_ui


# Create output folder
os.makedirs("outputs", exist_ok=True)


def get_video_path(video_file):
    """
    Compatible with different Gradio versions.

    Some versions return a string path.
    Some versions return a dictionary with "path" or "name".
    """
    if video_file is None:
        raise gr.Error("Please upload a video first.")

    if isinstance(video_file, str):
        return video_file

    if isinstance(video_file, dict):
        if "path" in video_file:
            return video_file["path"]
        if "name" in video_file:
            return video_file["name"]

    raise gr.Error(f"Unsupported video input format: {type(video_file)}")


def process_video_ui(
        video_file,
        effect_type,
        kernel_size,
        visual_strength,
        enable_denoise,
        denoise_strength,
        cutoff_hz,
        pitch_shift_steps
):
    """
    UI wrapper.

    Important:
    This function must return the real processed video path.
    Otherwise Gradio may display an old or mismatched video.
    """

    input_video_path = get_video_path(video_file)

    # Call backend processing function
    output_path = process_video_for_ui(
        input_video_path=input_video_path,
        effect_type=effect_type,
        kernel_size=kernel_size,
        visual_strength=visual_strength,
        enable_denoise=enable_denoise,
        denoise_strength=denoise_strength,
        cutoff_hz=cutoff_hz,
        pitch_shift_steps=pitch_shift_steps,
    )

    # Build a clear and unique output filename
    original_name = Path(input_video_path).stem
    unique_id = uuid4().hex[:8]

    effect = effect_type
    kernel = f"k{int(kernel_size)}"
    strength = f"vs{float(visual_strength):.1f}"
    denoise = "denoiseT" if enable_denoise else "denoiseF"
    ds = f"ds{float(denoise_strength):.1f}"
    cut = f"cut{int(cutoff_hz)}Hz"
    pitch = f"pitch{float(pitch_shift_steps):.1f}"

    new_filename = (
        f"{original_name}_{unique_id}_{effect}_{kernel}_"
        f"{strength}_{denoise}_{ds}_{cut}_{pitch}.mp4"
    )

    new_path = Path("outputs") / new_filename

    # Safely rename/move the generated video.
    # If renaming fails, return the original generated path instead of a wrong path.
    try:
        shutil.move(output_path, new_path)
        return str(new_path)
    except Exception as e:
        print(f"Warning: failed to rename output video. Returning original path. Error: {e}")
        return output_path


with gr.Blocks(title="Convolution-based Video-Audio Processing Tool") as demo:
    gr.Markdown("# Convolution-based Video-Audio Processing Tool")
    gr.Markdown(
        "Upload a video, choose a convolution-based visual effect, "
        "optionally apply audio denoising, and export the processed video."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Upload Video",
                format="mp4"
            )

            effect_type = gr.Dropdown(
                label="Video Effect Type",
                choices=["none", "gaussian_blur", "mean_blur", "sharpen"],
                value="gaussian_blur"
            )

            kernel_size = gr.Slider(
                label="Kernel Size / Blur Level",
                minimum=1,
                maximum=10,
                step=1,
                value=3
            )

            visual_strength = gr.Slider(
                label="Visual Strength",
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0
            )

            enable_denoise = gr.Checkbox(
                label="Enable Audio Denoising",
                value=True
            )

            denoise_strength = gr.Slider(
                label="Denoising Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=1.0
            )

            cutoff_hz = gr.Slider(
                label="Low-pass Cutoff Frequency (Hz)",
                minimum=1000,
                maximum=8000,
                step=100,
                value=3000
            )

            pitch_shift_steps = gr.Slider(
                label="Pitch Shift",
                minimum=-6,
                maximum=6,
                step=1,
                value=0
            )

            submit_btn = gr.Button(
                "Start Processing",
                variant="primary"
            )

        with gr.Column():
            video_output = gr.Video(
                label="Processed Video"
            )

    submit_btn.click(
        fn=process_video_ui,
        inputs=[
            video_input,
            effect_type,
            kernel_size,
            visual_strength,
            enable_denoise,
            denoise_strength,
            cutoff_hz,
            pitch_shift_steps
        ],
        outputs=video_output
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8888,
        inbrowser=True,
        share=False
    )
