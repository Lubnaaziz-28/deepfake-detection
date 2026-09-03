# deepfake-detection

Deepfake detection combining frame-level CNN features (EfficientNet) with temporal-consistency modeling (LSTM) over video sequences. Interpretable via Grad-CAM.

## Architecture
```
video -> frame sampler -> EfficientNet-B4 (per frame) -> LSTM (temporal) -> classifier
                                        \-> Grad-CAM overlays
```

## Results (FaceForensics++ c23)
| Model | AUC | Accuracy |
|---|---|---|
| Frame-only | TBD | TBD |
| + Temporal LSTM | **TBD** | **TBD** |

## Ethics
Defensive detection tooling only. No generation/manipulation code in this repo.

## Contact
Dr. Lubna Aziz | engr.lubnaaziz@gmail.com | [Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
