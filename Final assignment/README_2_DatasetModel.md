# Dataset + pretrained weights downloader
---
## 1. Make Environment 

---


## 2. Data Preparation

This project uses both In-Distribution (ID) and Out-of-Distribution (OOD) datasets. Due to file size limits and licensing, data retrieval is split into two steps.


### Step 2a: In-Distribution Data (Cityscapes)

The Cityscapes dataset is provided via the official course Hugging Face repository. With your virtual environment active, run:

```bash
huggingface-cli download TimJaspersTue/5LSM0 --local-dir ./datasets/cityscapes --repo-type dataset
```

> **Note:** If the Hugging Face CLI throws an authentication or connection error, download the Cityscapes dataset manually from the [official website](https://www.cityscapes-dataset.com) and place it in `datasets/cityscapes/`.


### Step 2b: Out-of-Distribution Data (Fishyscapes & SMIYC)

The OOD datasets (Fishyscapes and SegmentMeIfYouCan AnomalyTrack) have been pre-configured to avoid running complex mapping scripts.

> The access is restricted and requires a TU/e account. Ensure you are logged into your TU/e Microsoft/Google account before proceeding.

1. Download the pre-configured dataset zip file here: [**OneDrive link**](https://tuenl-my.sharepoint.com/:f:/g/personal/m_c_s_hochstenbach_student_tue_nl/IgC_07q9F-LsRas1tgzKZR8fAXYRgsksL-oyM1SGnbbTu6Y?e=czeOEH)
2. Extract the contents directly into the `datasets/` folder.

---

## 3. Model Weights
The weights of two pretrained models should be downloaded locally for the different tests. 

### Step 3a: DINOv3 ViT-B/16
The DINOv3 can be downloaded via timm using the following code in the terminal:
```bash
python -c"
import timm, torch
import os
os.makedirs('weights', exist_ok=True) # Makes sure the folder exists!
m = timm.create_model('vit_base_patch16_dinov3', pretrained=True, num_classes=0)
torch.save(m.state_dict(), 'weights/dinov3_vitb16_timm.pth')
print('DINOv3 weights saved successfully!')
"
```


### Step 3b: SegFormer-B5 (Cityscapes)

The SegFormer-B5 pretrained on Cityscapes is available on Hugging Face, the weights can be locally downloaded using:
```bash
python -c "
from transformers import SegformerForSemanticSegmentation
import torch

# Download the B5 model fine-tuned on Cityscapes
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

# Save just the state_dict (weights) to match your script's expectations
torch.save(model.state_dict(), 'weights/segformer.pt')
print('Successfully saved SegFormer weights to segformer.pt!')
"
```
---

## 4. Final Folder Structure

```plaintext
NNCV-MOREJAMPLZ/
├── Other python files etc from respiratory. 
├── requirements.txt
├── weights/
│   ├── dinov3_vitb16_timm.pth            # Downloaded via Step 3a
│   └── segformer.pth                     # Downloaded via Step 3b
└── datasets/
    ├── ID data/cityscapes/               # Downloaded via Hugging Face (Step 2a)
    ├    ├── bdd100k_500/
    │    └── dataset_AnomalyTrack/ 
    └── OOD data/   
        ├── datafishyscapes/              # Extracted from OneDrive zip (Step 2b)     
        └── dataset_AnomalyTrack/         # Extracted from OneDrive zip (Step 2b)                      
```

---


