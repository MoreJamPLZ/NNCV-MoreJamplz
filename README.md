# Final Assignment Overview

This folder contains both the experimentation workflow and the final submission packaging workflow.

## What each part is for

- `OOD_experiments.ipynb`:
  main notebook where the OOD-related experiments, sweeps, and score computations were run.
- `Exp1_UnetFolder` ... `Exp9_GSVDFolderThres5.5`:
  submission-preparation folders with model/predict scripts and helper scripts for creating `model.pt` variants.
- `README_2_DatasetModel.md`:
  setup guide for datasets and pretrained weights.
- `README-Submission.md`:
  submission guide for creating Docker images and exporting `.tar` files.

## Recommended workflow

1. Do analysis and metric computation in `OOD_experiments.ipynb`.
2. Pick one `Exp*` folder as the submission variant.
3. Copy its scripts to `Final assignment/` (one level up).
4. Run `make weights.py` (or equivalent) to create `model.pt` in `Final assignment/`.
5. Build Docker with `Final assignment/Dockerfile` and export the image as `.tar`.

Use the detailed docs above for exact commands.
