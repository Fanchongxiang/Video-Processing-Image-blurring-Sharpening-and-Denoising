# Video-Processing-Image-blurring-Sharpening-and-Denoising
The goal of our project is to create a website based on Gradio for the users that input a video and choose the type they want. In addtion, the strength is adjustable. Users could download the processed video after processing.


1.audio_convolution_advanced.py  声音降噪处理文件
	video_convolution.py  视频锐化/模糊处理文件
	process_video.py 是集成了A和B部分的代码，并为D提供了UI设计的接口

2.前端调用函数：
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

3. 参数说明：

3.1 input_video_path
   上传视频后的本地路径。

3.2 effect_type
   画面处理方式，可选：
   - "none"：不处理画面
   - "gaussian_blur"：高斯模糊
   - "mean_blur"：均值模糊
   - "sharpen"：锐化

3.3 kernel_size
   模糊程度，建议只给奇数：
   3, 5, 7, 9, 11, 15  （UI设计的时候建议用连续的整数，通过代码设计映射，比如用户前端上显示的是1-10，内部操作的是3开始的奇数）
   对 sharpen 不起主要作用。

3.4 visual_strength
   锐化强度，建议范围：
   0.5 到 2.0
   默认 1.0。 （UI设计道理同上）
   对 sharpen 作用明显，对 blur 作用不大。

3.5 enable_denoise
   是否开启音频降噪：
   True / False

3.6 denoise_strength
   降噪强度，建议范围：
   0.0 到 1.0
   0.0 表示不降噪，1.0 表示完整降噪。

3.7 lowcut_hz/highcut_hz
   带通截止频率。
