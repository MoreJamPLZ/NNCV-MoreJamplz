# GSVD-Based Outlier Detection (and a bit of Semantic Segmentation)

Code for the final assignment of **5LSM0 – Neural Networks for Computer Vision** at TU/e.

This repo contains the code behind the paper *"GSVD-Based Outlier Detection Score for Image Detection"* by Mirjam Hochstenbach (me). The idea: compare the feature maps of a pretrained DINOv3 expert and a randomly initialized novice using the Generalized Singular Value Decomposition (GSVD), and use the size of the disagreement as an OOD score.

---

## Submission Details
**Best performing model(s):**
* **For peak performance:** `MoreJamPLZSegFinal`
* **For out of distribution:** `MoreJamPLZ_entropy_0.1585`

---

## What's in this folder

- `OOD_experiments.ipynb`:
  main notebook where the OOD experiments, sweeps, and score computations were run.
- `Exp1_UnetFolder` … `Exp9_GSVDFolderThres5.5`:
  submission-preparation folders with model/predict scripts and helper scripts for creating `model.pt` variants.
- `README_1_DatasetModel.md`:
  **start here**, setup guide for datasets and pretrained weights.
- `README-Submission.md`:
  submission guide for creating Docker images and exporting `.tar` files.
- `Final assignment/`:
  the folder that gets packaged up for the actual challenge submission (Dockerfile lives here).

## Recommended workflow

0. Set up datasets and weights by following `README_1_DatasetModel.md` first.
1. Do analysis and metric computation in `OOD_experiments.ipynb`.
2. Pick one `Exp*` folder as the submission variant.
3. Copy its scripts to `Final assignment/` (one level up).
4. Run `make weights.py` (or equivalent) to create `model.pt` in `Final assignment/`.
5. Build Docker with `Final assignment/Dockerfile` and export the image as `.tar`.

Use the detailed docs above for exact commands.

## Acknowledgments

Thanks to the 5LSM0 course supervisors at TU/e, and to Michiel Hochstenbach (TU/e) for the GSVD code for rectangular matrices.
