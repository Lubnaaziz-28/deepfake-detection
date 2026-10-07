"""Gradio live demo: paste/upload a video, get a Real/Fake verdict,
per-clip predictions, a probability bar, and a Grad-CAM heatmap strip.

This file doubles as the Hugging Face Spaces entrypoint (app_file: app.py).
"""

import os
import tempfile
import urllib.request

import gradio as gr

import pipeline

MAX_VIDEO_MB = 200
SPACE_URL = "https://huggingface.co/spaces/Lubnaaziz-28/deepfake-detection-demo"

pipe = pipeline.DeepfakePipeline(weights_path=pipeline.resolve_weights())


def download_video(url):
    url = url.strip()
    if not url:
        raise gr.Error("No URL provided. Upload a video file or paste a direct video URL.")
    if not url.lower().startswith(("http://", "https://")):
        raise gr.Error("URL must start with http:// or https://")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        size = int(response.headers.get("Content-Length", 0))
        if size > MAX_VIDEO_MB * 1024 * 1024:
            raise gr.Error(f"Video too large ({size / 1024 / 1024:.0f} MB > {MAX_VIDEO_MB} MB).")
        data = response.read()
    if not data:
        raise gr.Error("Downloaded file is empty.")
    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp4"
    handle, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(handle, "wb") as f:
        f.write(data)
    return path


def analyze(video, url, clip_size, max_clips):
    video_path = video
    source = "upload"
    if url and url.strip():
        video_path = download_video(url)
        source = f"URL: {url.strip()}"
    if not video_path:
        raise gr.Error("Please upload a video or paste a video URL.")

    yield (
        "Sampling frames...",
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
    )
    frames = pipeline.sample_frames(video_path)

    yield (
        f"Running {len(frames)} frames through EfficientNet-B4 + LSTM...",
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
    )
    result = pipe.analyze(
        frames, clip_size=int(clip_size), max_clips=int(max_clips)
    )

    confidence = max(result["p_real"], result["p_fake"]) * 100
    color = "#C0392B" if result["verdict"] == "FAKE" else "#1E8449"
    mode_note = (
        "\n\n⚠️ **Demo mode** — no trained checkpoint loaded (random weights). "
        "Upload `weights/best.pth` or set `HF_WEIGHTS` for real predictions."
        if pipe.demo_mode
        else ""
    )
    banner = (
        f"<div style='text-align:center; padding:16px; "
        f"background:{color}1a; border:2px solid {color}; border-radius:12px;'>"
        f"<span style='font-size:36px; font-weight:bold; color:{color};'>"
        f"VERDICT: {result['verdict']}</span><br/>"
        f"<span style='font-size:18px;'>Confidence {confidence:.1f}% "
        f"· {len(result['clips'])} clip(s) · Grad-CAM on clip {result['keyframe_clip']}"
        f"</span></div>{mode_note}"
    )

    plot = [["Real", round(result["p_real"], 4)], ["Fake", round(result["p_fake"], 4)]]

    table = [
        [c["clip"], c["frames"], round(c["p_real"], 4), round(c["p_fake"], 4), c["verdict"]]
        for c in result["clips"]
    ]

    strip_caption = (
        f"Source: {source} — side-by-side: original frame | temporal Grad-CAM overlay"
        f" (red = fake-driving regions)"
    )

    yield (
        f"Done — verdict: {result['verdict']} ({confidence:.1f}% confidence)",
        banner, plot, table,
        (result["strip"], strip_caption),
        [(pair, f"frame pair {i + 1}") for i, pair in enumerate(result["gallery"])],
    )


with gr.Blocks(title="Deepfake Detection — Live Demo") as demo:
    gr.Markdown(
        "# Deepfake Detection — Live Demo\n"
        "**EfficientNet-B4 + LSTM** temporal model with **Grad-CAM** explainability. "
        "Upload a video or paste a direct video URL (mp4/mov/webm)."
    )
    if pipe.demo_mode:
        gr.HTML(
            "<div style='background:#FFF3CD; border:1px solid #FFD21E; "
            "padding:10px; border-radius:8px; margin-bottom:12px;'>"
            "⚠️ <b>Demo mode:</b> no trained checkpoint found — predictions come from "
            "random weights and are meaningless. Drop a trained <code>weights/best.pth</code> "
            "into this Space or set the <code>HF_WEIGHTS</code> env variable "
            "(format: <code>org/repo/filename</code>) to enable real detection.</div>"
        )

    with gr.Row():
        video_input = gr.Video(label="Upload video", sources=["upload"])
        with gr.Column():
            url_input = gr.Textbox(label="...or paste a direct video URL", placeholder="https://example.com/video.mp4")
            with gr.Row():
                clip_size = gr.Slider(4, 32, value=16, step=4, label="Frames per clip")
                max_clips = gr.Slider(1, 4, value=2, step=1, label="Max clips analyzed")
            analyze_btn = gr.Button("Analyze", variant="primary")

    status = gr.Textbox(label="Status", interactive=False)
    verdict = gr.HTML()
    prob_plot = gr.BarPlot(
        x="class",
        y="probability",
        label="Probability bar",
        color=["#1E8449", "#C0392B"],
        title="P(Real) vs P(Fake)",
        y_lim=[0, 1],
    )
    clip_table = gr.Dataframe(
        headers=["Clip", "Frames", "P(Real)", "P(Fake)", "Verdict"],
        datatype=["number", "str", "number", "number", "str"],
        label="Per-clip predictions",
        interactive=False,
    )
    strip_image = gr.Image(label="Grad-CAM heatmap strip (original | overlay)", type="numpy")
    pair_gallery = gr.Gallery(label="Frame-by-frame pairs", columns=2, height="auto")

    analyze_btn.click(
        fn=analyze,
        inputs=[video_input, url_input, clip_size, max_clips],
        outputs=[status, verdict, prob_plot, clip_table, strip_image, pair_gallery],
        api_name="analyze",
    )

    gr.Markdown(
        "---\n"
        "**How it works:** the video is uniformly sampled (default 16 frames), "
        "each frame is encoded by a shared EfficientNet-B4 backbone, an LSTM models "
        "temporal consistency across the sequence, and a classifier head outputs "
        "Real/Fake probabilities per clip. For explainability, the fake-class score "
        "is backpropagated through the LSTM to every frame's final convolutional "
        "feature map (temporal Grad-CAM); red regions in the overlay are the "
        "strongest fake-driving artifacts.\n\n"
        "This is a **defensive detection tool** only. No deepfake generation code "
        "is included."
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="purple"),
    )
