# Fingerprint Classification & Analysis

End-to-end pipeline for fingerprint pattern classification (8 classes) and core/delta detection with FastAPI endpoints.

## Setup
- Python 3.9+ recommended.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Dataset Preparation
Expected raw layout (grayscale images):
```
dataset_raw/
  wpe/ ws/ wd/ we/ lu/ au/ at/ as/
```
Stratify into train/val/test (70/15/15, seed=42):
```bash
python splitDataset.py
```
This creates `dataset_split/train|val|test/<class>/`.

## Training
- Single run (saves `classification.pth`, `confusion_matrix.png`):
  ```bash
  python trainDataset.py
  ```
- Multi-seed experiments + fine-tune + two-stage routing (outputs under `results_baseline`, `results_finetune`, `results_two_stage`):
  ```bash
  python runAllExperiments.py
  ```
  Fine-tuned checkpoint used by the APIs: `results_finetune/seed_2025/finetune_best_model.pth`.

## APIs
### Classification + Poincaré (full analysis)
`api_server.py` loads the fine-tuned ResNet-18 and exposes:
- `GET /health`
- `POST /predict` (classification)
- `POST /poincare` (core/delta)
- `POST /analyze` (classification + core/delta)

Run:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Detector demo (classification + overlay)
`api.py` classifies first, then runs core/delta detection, and serves a simple UI. Upload validity is decided only by structure + quality gates; Poincaré/feasibility disagreements become warnings (not rejects):
- `GET /` , `GET /health`
- `GET /demo` (HTML demo)
- `POST /detect` (returns classification + detection; overlay is returned inline as base64; includes rule consistency, warnings, and decision block)

Run:
```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/demo` to test in the browser.

#### Upload gates in `api.py` (summary)
- Structure rejects: foreground too small, span too small, elongated aspect, extreme edge density low, heavy fragmentation with noise, missing core region when fragmented. High foreground by itself is only a warning; smudge reject needs high FG + extreme edges + ≥1 more strong evidence (speckle/fragmented/ridges_bad).
- Quality rejects: `mean_quality < 0.55` or `mean_coherence < 0.60`. Coherence 0.60–0.65 is a warning. Missing quality metrics adds a warning, not a reject.
- Warnings include: high FG, dense edges (non-extreme), low center coverage, fragmentation suspect, speckle noise, low coherence, quality missing, CNN/rule disagreement, feasibility mismatch.

## Other Utilities
- `summarize_experiments.py`: aggregate experiment outputs.

## Included Artifacts (tracked)
- `results_finetune/seed_2025/finetune_best_model.pth` — fine-tuned ResNet-18 checkpoint used by the APIs.
- `results_finetune/seed_2025/finetune_training_log.csv` — per-epoch train loss, val loss, and val macro-F1.
- `results_finetune/seed_2025/finetune_training_curve.png` — plot of train/val loss and val macro-F1.
- `results_finetune/seed_2025/train_metrics.png` & `train_confusion_matrix.png` — training-set precision/recall/F1 and confusion matrix.
- `results_finetune/seed_2025/test_metrics.json` & `test_confusion_matrix.png` — test-set metrics and confusion matrix.

## Deployment (Docker / Render)
Build and run locally:
```bash
docker build -t fingerprint-api .
docker run -p 8000:8000 fingerprint-api
```
Then open `http://localhost:8000/docs` (FastAPI Swagger UI).

Deploy to Render (Docker service):
- Choose “New > Web Service” → “Build & Deploy from a Git repo”.
- Runtime: Docker; Dockerfile path: `Dockerfile`; region: closest to users.
- Render sets `$PORT` automatically; the container command `uvicorn api:app --host 0.0.0.0 --port $PORT` is already in the Dockerfile.
- Ensure `results_finetune/seed_2025/finetune_best_model.pth` stays in the repo so the API loads the checkpoint.
"# luminousdemo" 
