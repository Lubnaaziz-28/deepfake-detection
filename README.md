<div align="center">

# Deepfake Detection

### Temporal Analysis Meets Interpretability

[![CI](https://img.shields.io/github/actions/workflow/status/Lubnaaziz-28/deepfake-detection/ci.yml?logo=github&style=flat-square)]()
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
pip install -e .[dev]            # or: pip install deepfake-scan

# Prepare dataset
python scripts/prepare_faceforensics.py --data-root ./data/faceforensics

# Train
python train.py --config configs/efficientnet_lstm.yaml --data-root ./data/faceforensics

# Inference
python detect.py --model weights/best.pth --video suspect_video.mp4
```

## CI Integration

Scan every video in a pull request with a single step:

```yaml
- uses: Lubnaaziz-28/deepfake-detection/action@main
  with:
    input-dir: ./videos
    threshold: '0.5'
```

The action runs `deepfake-scan` on PRs and pushes, writes `deepfake-report.json` (per-file **verdict**, **confidence**, and **heatmap paths**) plus heatmap overlays, and fails the job when any video's confidence meets `--threshold` (disable with `fail-on-deepfake: 'false'`). A full example with artifact uploads lives in [`.github/workflows/deepfake-scan.yml`](.github/workflows/deepfake-scan.yml).

| Input | Default | Purpose |
|---|---|---|
| `input-dir` | `./videos` | Directory scanned recursively for video files |
| `threshold` | `0.5` | Confidence cutoff for a deepfake verdict (precision/recall tuning) |
| `report` | `deepfake-report.json` | JSON report output path |
| `heatmaps` | `deepfake-heatmaps` | Heatmap overlay output directory |
| `frames` | `16` | Frames sampled per video |
| `model` | – | Trained weights (`weights/best.pth`); without them the temporal-coherence baseline is used |
| `fail-on-deepfake` | `true` | Fail the workflow when a deepfake is flagged |

**Outputs:** `flagged` (number of flagged videos), `report-path`.

### CLI

```bash
pip install deepfake-scan
deepfake-scan ./videos --threshold 0.5 --report report.json
```

Exit code `1` with `--fail-on-deepfake` when any video is flagged, so CI can gate on the verdict.

### Marketplace

`action/action.yml` is Marketplace-ready. To publish: create a release tag, then enable the listing under **Settings → General → Marketplace**. The action also works without publishing — reference it by path (`uses: ./action`) in private repos or on self-hosted runners.

## Citation

```bibtex
@software{aziz2023deepfake,
  title={Deepfake Detection with Temporal Analysis},
  author={Aziz, Lubna},
  year={2023}
}
```

## Contact

Dr. Lubna Aziz, engr.lubnaaziz@gmail.com, [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
