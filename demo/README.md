---
title: Deepfake Detection — Live Demo
emoji: 🎥
colorFrom: rose
colorTo: purple
sdk: gradio
app_file: app.py
license: mit
---

# Deepfake Detection — Live Demo

Paste a video URL or upload a video and get:

- **Real/Fake verdict** with a confidence probability bar
- **Per-clip predictions** (temporal LSTM consistency across frame sequences)
- **Grad-CAM heatmap strip** — each sampled frame shown side-by-side with its
  temporal Grad-CAM overlay (red = regions driving the fake verdict)

Architecture: **EfficientNet-B4** spatial backbone (shared weights) + **LSTM**
temporal modeling + classifier head, with temporal Grad-CAM backpropagating the
fake-class score through the LSTM to every frame's final convolutional map.

## Deploy your own

This folder is a complete, self-contained Hugging Face Space:

```bash
# 1. Create the space (or via https://huggingface.co/new-space, SDK: Gradio)
curl -X POST "https://huggingface.co/api/spaces/<org>/deepfake-detection-demo" \
  -H "Authorization: Bearer $HF_TOKEN" -d '{"sdk":"gradio"}'

# 2. Push this folder as the space content
git clone https://huggingface.co/spaces/<org>/deepfake-detection-demo
cp -r demo/* <space-clone>/ && cd <space-clone>
git add . && git commit -m "Add demo" && git push
```

To run real (non-random-weight) predictions, upload a trained checkpoint as
`weights/best.pth` in the Space files, or set Space secret/env variable
`HF_WEIGHTS` to `org/repo/filename` on the Hub.

## Local run

```bash
pip install -r demo/requirements.txt
python demo/app.py   # http://localhost:7860
```
