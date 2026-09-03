<div align="center">

# Deepfake Detection

### Temporal Analysis Meets Interpretability

[![Tech](https://img.shields.io/badge/Tech-EfficientNet_%2B_LSTM-1ABC9C)]()
[![Dataset](https://img.shields.io/badge/Dataset-FaceForensics%2B%2B-E67E22)]()
[![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch&logoColor=white)]()

*Hybrid spatial-temporal architecture for deepfake detection with explainable outputs.*

</div>

---

## The Problem

Single-frame deepfake detectors miss temporal inconsistencies. A deepfake may look perfect in one frame but the video's temporal coherence reveals manipulation.

## The Solution

**Spatial features** (EfficientNet-B4) + **temporal consistency** (LSTM) + **interpretability** (Grad-CAM) to detect deepfakes across video sequences.

```
Video Input
    │
    ▼
┌─────────────────────────────────────────┐
│           Frame Sampler                 │
│     (16 frames per video)               │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Frame 1 │  │ Frame 2 │  │ Frame N │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     ▼            ▼            ▼
┌─────────────────────────────────────────┐
│      EfficientNet-B4 (shared weights)   │
│      Extracts spatial features          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  LSTM Temporal │
         │  Consistency   │
         │  Modeling      │
         └────────┬───────┘
                  │
         ┌────────▼───────┐
         │   Classifier   │
         │   (Real/Fake)  │
         └────────┬───────┘
                  │
         ┌────────▼───────┐
         │   Grad-CAM     │
         │   Overlays     │
         └────────────────┘
```

## Results (FaceForensics++ c23)

| Model | AUC | Accuracy |
|---|---|---|
| Frame-only (EfficientNet) | TBD | TBD |
| **+ Temporal LSTM** | **TBD** | **TBD** |

## Ethics

This is a **defensive detection tool** only. No deepfake generation or manipulation code is included in this repository.

## Quickstart

```bash
pip install -r requirements.txt

# Prepare dataset
python scripts/prepare_faceforensics.py --data-root ./data/faceforensics

# Train
python train.py --config configs/efficientnet_lstm.yaml --data-root ./data/faceforensics

# Inference
python detect.py --model weights/best.pth --video suspect_video.mp4
```

## Citation

```bibtex
@software{aziz2023deepfake,
  title={Deepfake Detection with Temporal Analysis},
  author={Aziz, Lubna},
  year={2023}
}
```

## Contact

Dr. Lubna Aziz — engr.lubnaaziz@gmail.com — [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
