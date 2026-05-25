# Video-Processing-Image-blurring-Sharpening-and-Denoising
The goal of our project is to create a website based on Gradio for the users that input a video and choose the type they want. In addtion, the strength is adjustable. Users could download the processed video after processing.


1.audio_convolution_advanced.py  声音降噪处理文件
	video_convolution.py  视频锐化/模糊处理文件
	process_video.py 是集成了A和B部分的代码，并为D提供了UI设计的接口



前端调用函数：
process_video_for_ui(
    input_video_path,
    effect_type,
    kernel_size,
    visual_strength,
    enable_denoise,
    denoise_strength,
    cutoff_hz,
    pitch_shift_steps
)

实现代码：
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




参数说明：

1. input_video_path
   上传视频后的本地路径。

2. effect_type
   画面处理方式，可选：
   - "none"：不处理画面
   - "gaussian_blur"：高斯模糊
   - "mean_blur"：均值模糊
   - "sharpen"：锐化

3. kernel_size
   模糊程度，建议只给奇数：
   3, 5, 7, 9, 11, 15        （UI设计的时候建议用连续的整数，通过代码设计映射，比如用户前端上显示的是1-10，内部操作的是3开始的奇数）
   数值越大，模糊越明显。
   对 sharpen 不起主要作用。

4. visual_strength
   锐化强度，建议范围：
   0.5 到 2.0
   默认 1.0。 （UI设计道理同上）
   对 sharpen 作用明显，对 blur 作用不大。

5. enable_denoise
   是否开启音频降噪：
   True / False

6. denoise_strength
   降噪强度，建议范围：
   0.0 到 1.0
   0.0 表示不降噪，1.0 表示完整降噪。

7. cutoff_hz
   降噪低通截止频率，建议范围：
   1000 到 8000
   默认 3000。
   越低，去高频噪声越强，但声音可能更闷。
