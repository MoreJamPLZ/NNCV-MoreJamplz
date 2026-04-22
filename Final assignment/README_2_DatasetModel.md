# Dataset + pretrained weights downloader

## 1. Setting up the environment

This project uses two environments:

1. A local Python environment for downloading datasets and pretrained weights (see sections 2 and 3).
2. A Docker environment for the final challenge submission/inference (`Final assignment/Dockerfile`).

For this README you only need the local Python environment. Downloading datasets and model weights doesn't need much compute since only the U-Net is trained here.

Run everything from the repo root (`NNCV-MoreJamplz/`), which is where `requirements.txt` lives.

### Conda environment

```bash
conda create -n nncv_env python=3.10 -y
conda activate nncv_env
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "huggingface_hub[cli]" timm transformers
```

### Quick check to see if it worked

```bash
python -c "import torch, timm, transformers; print('Environment OK')"
huggingface-cli --version
```

---

## 2. Getting the data

We use both In-Distribution (ID) and Out-of-Distribution (OOD) datasets. Because of file size limits and licensing stuff, we had to split the download into two steps.

### Step 2a: In-Distribution data (Cityscapes)

Cityscapes is hosted on the course Hugging Face repo. With your env active, just run:

```bash
huggingface-cli download TimJaspersTue/5LSM0 --local-dir ./datasets/cityscapes --repo-type dataset
```

> **Note:** If it ends up not working i grabbed Cityscapes manually from the [official website](https://www.cityscapes-dataset.com) and drop it into `datasets/cityscapes/`.

### Step 2b: Out-of-Distribution data (Fishyscapes & SMIYC)

We pre-configured the OOD datasets (Fishyscapes and SegmentMeIfYouCan AnomalyTrack) so you don't have to run the annoying mapping scripts yourself.

> Access is restricted and needs a TU/e account, so make sure you're logged into your TU/e Microsoft/Google account first.

1. Grab the zip here: [**OneDrive link**](https://tuenl-my.sharepoint.com/:f:/g/personal/m_c_s_hochstenbach_student_tue_nl/IgC_07q9F-LsRas1tgzKZR8fAXYRgsksL-oyM1SGnbbTu6Y?e=czeOEH)
2. Unzip it straight into the `datasets/` folder.

---

## 3. Model weights

You'll need to download two pretrained models locally for the different tests.

### Step 3a: DINOv3 ViT-B/16

You can pull DINOv3 through timm by running this in the terminal:

```bash
python -c"
import timm, torch
import os
os.makedirs('weights', exist_ok=True)  # make the folder if it isn't there yet
m = timm.create_model('vit_base_patch16_dinov3', pretrained=True, num_classes=0)
torch.save(m.state_dict(), 'weights/dinov3_vitb16_timm.pth')
print('DINOv3 weights saved successfully!')
"
```

### Step 3b: SegFormer-B5 (Cityscapes)

SegFormer-B5 pretrained on Cityscapes is on Hugging Face. Download the weights with:

```bash
python -c "
from transformers import SegformerForSemanticSegmentation
import torch

# Grab the B5 model fine-tuned on Cityscapes
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

# Save just the state_dict so it matches what the script expects
torch.save(model.state_dict(), 'weights/segformer.pt')
print('Successfully saved SegFormer weights to segformer.pt!')
"
```

---

## 4. What the folder should look like in the end

```plaintext
NNCV-MOREJAMPLZ/
├── Other python files etc from the repo
├── requirements.txt
├── weights/
│   ├── dinov3_vitb16_timm.pth            # from Step 3a
│   └── segformer.pth                     # from Step 3b
└── datasets/
    ├── ID data/cityscapes/               # from Hugging Face (Step 2a)
    ├    ├── bdd100k_500/
    │    └── dataset_AnomalyTrack/
    └── OOD data/
        ├── datafishyscapes/              # from the OneDrive zip (Step 2b)
        └── dataset_AnomalyTrack/         # from the OneDrive zip (Step 2b)
```