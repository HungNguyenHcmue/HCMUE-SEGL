<div align="center">

<h1>🎓 HCMUE-SEGL</h1>
<h2>Group Learning Integrating Behaviors and Emotions Recognition to Assess Student Engagement in Classroom Context</h2>
<h3><em>The 17th International Conference on Knowledge and Systems Engineering (KSE 2025)</em></h3>

<p>
  Hung Q. Nguyen<sup>1</sup>, Dat Ly<sup>1</sup>, Nha Tran<sup>2,3,1,*</sup>, Tri Le<sup>1</sup>, Hung Nguyen<sup>1</sup>, and Hien D. Nguyen<sup>2,3</sup>
</p>
<p>
  <sup>1</sup>Faculty of Information Technology, Ho Chi Minh City University of Education, Ho Chi Minh City, Vietnam
  <br>
  <sup>2</sup>Faculty of Computer Science, University of Information Technology, Ho Chi Minh City, Vietnam
  <br>
  <sup>3</sup>Vietnam National University, Ho Chi Minh City, Vietnam
</p>

<p>
  <a href="https://doi.org/10.1109/KSE65613.2025.10981613">
    <img src="https://img.shields.io/badge/Paper-IEEE%20KSE%202025-blue?style=flat-square&logo=ieee" alt="Paper">
  </a>
  <a href="https://drive.google.com/drive/folders/14M_DLRPvoNvquJCZN4482E0KOsjjoWRq?usp=sharing">
    <img src="https://img.shields.io/badge/Dataset-Google%20Drive-green?style=flat-square&logo=googledrive" alt="Dataset">
  </a>
  <a href="https://hungnguyenhcmue.github.io/HCMUE-SEGL/">
    <img src="https://img.shields.io/badge/Project-Page-orange?style=flat-square&logo=github" alt="Project Page">
  </a>
</p>

<p><b>HCMUE Student Engagement in Group Learning Dataset</b></p>

</div>

---

## 📋 Overview

**HCMUE-SEGL** is a real-world multimodal dataset designed for assessing **student engagement in small-group learning** environments. It captures behavioral, emotional, and group-context data from multiple camera angles, providing diverse perspectives for group learning engagement analysis.

| Property | Details |
|---|---|
| # Participants | 61 students (avg. age 21) |
| # Video segments | 1,663 annotated samples |
| # Engagement levels | 4 (Very Engaged / Engaged / Normal / Disengaged) |
| # Camera angles | 4 (Angle1 – Angle4) |
| Frames per sample | 15 sampled time steps: 15 RGB body frames, 15 face crops, and 1 group-context frame |
| Frame resolution | 720p @ 30fps |
| Modalities | RGB body frames, face crops, skeleton CSV, group-context (middle frame) |

<p align="center">
  <img src="docs/method.png" width="850" alt="MoGLE-Net architecture">
  <br>
  <em>Fig. 1 — Overview of the multimodal pipeline used to process HCMUE-SEGL.</em>
</p>

---

## 📦 Dataset Structure

The dataset is organized hierarchically by **camera angle → engagement label → modality**. Below is the full directory layout:

```
HCMUE-SEGL/
├── Angle1/                        # Camera angle 1
│   └── Table1/
│       ├── Context/                # Group-context frame (shared across all labels)
│       │   ├── X0_A1/              # Context frame for Disengaged samples
│       │   │   └── Video18/
│       │   │       └── id_2/
│       │   │           └── context_frame.jpg     # middle frame of the 15-frame sequence
│       │   ├── X1_A1/              # Context frame for Normal samples
│       │   ├── X2_A1/              # Context frame for Engaged samples
│       │   └── X3_A1/              # Context frame for Very Engaged samples
│       ├── Disengaged/          # Engagement label folder
│       │   ├── Face/                # Face crops (15 frames per sample)
│       │   │   └── Video18/
│       │   │       └── id_2/
│       │   │           ├── frame_001.png
│       │   │           ├── frame_002.png
│       │   │           └── ... (frame_001 to frame_015)
│       │   └── X0_A1/               # RGB body frames + skeleton data
│       │       └── Video18/
│       │           └── id_2/
│       │               ├── frame_001.png
│       │               ├── frame_002.png
│       │               ├── ... (15 RGB frames)
│       │               └── skeletons.csv
│       ├── Normal/
│       │   ├── Face/
│       │   └── X1_A1/
│       ├── Engaged/
│       │   ├── Face/
│       │   └── X2_A1/
│       └── Very Engaged/
│           ├── Face/
│           └── X3_A1/
├── Angle2/                        # Camera angle 2 (same structure)
│   └── Table1/
├── Angle3/                        # Camera angle 3 (same structure)
│   └── Table1/
└── Angle4/                        # Camera angle 4 (same structure)
    └── Table1/
```

### Folder Naming Convention

| Pattern | Meaning |
|---|---|
| `Angle1` – `Angle4` | Four camera corners around the table |
| `X0_A<N>` | Disengaged label, angle N |
| `X1_A<N>` | Normal label, angle N |
| `X2_A<N>` | Engaged label, angle N |
| `X3_A<N>` | Very Engaged label, angle N |
| `Face/` | Face crop sequences |
| `Context/` | Group-context frame, one middle frame per sample |
| `skeletons.csv` | 13-keypoint skeleton coordinates for all 15 frames |

> **Note on `Context/`:** unlike the other branches, `Context/` sits one level above the four engagement-label folders and holds its own `X0_A<N>` … `X3_A<N>` subfolders (mirroring the same `Video/id` layout as the other branches). Each `Video/id` folder contains a **single RGB frame of the whole group** — the middle frame of that sample's 15-frame sequence. This frame is not cropped to one person; it captures the full group at that moment, consistent with the context branch described in the paper.

### Per-Sample Contents

Each sample (one student, one video clip, one camera angle) contains:

```
VideoXXX/id_Y/
├── [inside <Label>/X<N>_A<M>/]   ← RGB body branch
│   ├── frame_001.png
│   ├── frame_002.png
│   ├── ...
│   ├── frame_015.png   
│   └── skeletons.csv               
├── [inside <Label>/Face/]         ← Face emotion branch
│   ├── frame_001.png
│   ├── frame_002.png
│   ├── ...
│   └── frame_015.png             
└── [inside Context/X<N>_A<M>/]    ← Group context branch
    └── context_frame.jpg           (group RGB, middle frame of the 15-frame seq)
```

> **Note:** Both the RGB body branch and the face branch use sequential frame names from `frame_001.png` to `frame_015.png`.

## 🚀 Quick Start

### Requirements

```bash
pip install torch torchvision torch_geometric pillow pandas scikit-learn tqdm
```

### Data Loading

Below is a minimal example to load one sample from the dataset:

```python
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision import transforms

IMG_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- Paths (adjust to your local mount) ----
LABEL_MAP = {"Disengaged": 0, "Normal": 1, "Engaged": 2, "Very Engaged": 3}
X_TAG     = {"Disengaged": "X0", "Normal": "X1", "Engaged": "X2", "Very Engaged": "X3"}

def load_sample(root: Path, angle: str, label_name: str, video: str, id_name: str):
    """
    root       : path to HCMUE-SEGL/
    angle      : e.g. "Angle1"
    label_name : one of the keys in LABEL_MAP
    video      : e.g. "Video18"
    id_name    : e.g. "id_2"
    """
    table = root / angle / "Table1"
    x_tag = f"{X_TAG[label_name]}_{angle.replace('Angle', 'A')}"   # e.g. X0_A1

    # 1. RGB body frames (15 frames + skeletons.csv)
    rgb_dir   = table / label_name / x_tag / video / id_name
    rgb_imgs  = sorted(rgb_dir.glob("*.png"))[:15]
    beh_tensor = torch.stack([IMG_TF(Image.open(p).convert("RGB")) for p in rgb_imgs])

    # 2. Skeleton CSV  →  shape (15, n_cols)
    skel_df  = pd.read_csv(rgb_dir / "skeletons.csv")
    skel_arr = skel_df.select_dtypes(include=[np.number]).values.astype(np.float32)[:15]
    skel_tensor = torch.from_numpy(skel_arr)  # (15, 26) for 13 joints × 2 coords

    # 3. Face emotion frames (15 frames)
    face_dir   = table / label_name / "Face" / video / id_name
    face_imgs  = sorted(face_dir.glob("*.png"))
    face_tensor = torch.stack([IMG_TF(Image.open(p).convert("RGB")) for p in face_imgs])

    # 4. Group-context frame (middle frame of the 15-frame sequence)
    ctx_dir  = table / "Context" / x_tag / video / id_name
    ctx_path = ctx_dir / "context_frame.jpg"
    ctx_tensor = IMG_TF(Image.open(ctx_path).convert("RGB")) if ctx_path.exists() else None

    label = LABEL_MAP[label_name]
    return beh_tensor, skel_tensor, face_tensor, ctx_tensor, label


# Example usage
root = Path("/path/to/HCMUE-SEGL")
beh, skel, face, ctx, label = load_sample(root, "Angle1", "Disengaged", "Video18", "id_2")
print(f"Body    : {beh.shape}")    # → torch.Size([15, 3, 224, 224])
print(f"Skel    : {skel.shape}")   # → torch.Size([15, 26])
print(f"Face    : {face.shape}")   # → torch.Size([15, 3, 224, 224])
print(f"Context : {ctx.shape if ctx is not None else None}")  # → torch.Size([3, 224, 224])
print(f"Label   : {label}")        # → 0
```

### Full Dataset Class

For training, use the `MultiBranchDataset` provided in [`dataloader.py`](dataloader.py), which handles all four camera angles, skeleton graph construction, and the group-context frame.

```python
from pathlib import Path
from dataloader import MultiBranchDataset, mb_collate
from torch.utils.data import DataLoader

ROOT = Path("/path/to/HCMUE-SEGL")
ANGLE_ROOTS = [ROOT / f"Angle{i}" / "Table1" for i in range(1, 5)]

ds = MultiBranchDataset(ANGLE_ROOTS)
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=mb_collate)
```

---

## 📊 Results

| Configuration | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| RGB Stream only | 83.00% | 84.00% | 83.00% | 86.00% |
| Facial Stream only | 72.00% | 75.00% | 72.00% | 73.00% |
| Pose Stream only | 58.92% | 60.97% | 58.43% | 60.64% |
| RGB + Facial | 87.00% | 87.00% | 87.13% | 87.13% |
| Pose + Facial | 73.59% | 73.73% | 74.61% | 76.74% |
| **Full model** | **90.44%** | **90.83%** | **90.70%** | **90.70%** |

---

## 📥 Download

The dataset is available on Google Drive:

> 🔗 **[Download HCMUE-SEGL Dataset](https://drive.google.com/drive/folders/14M_DLRPvoNvquJCZN4482E0KOsjjoWRq?usp=sharing)**

After downloading, extract and place the `HCMUE-SEGL/` folder at your preferred root path, then update the `ANGLE_ROOTS` path in the training scripts accordingly.

---

## 📄 Citation

If you use HCMUE-SEGL in your research, please cite:

```bibtex
@inproceedings{nguyen2025group,
  title={Group Learning Integrating Behaviors and Emotions Recognition to Assess Student Engagement in Classroom Context},
  author={Nguyen, Hung Q and Ly, Dat and Tran, Nha and Le, Tri and Nguyen, Hung and Nguyen, Hien D},
  booktitle={2025 17th International Conference on Knowledge and Systems Engineering (KSE)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

---

## 🙏 Acknowledgment

This research was supported by **The Ho Chi Minh University of Education's Scientific Research Support Fund**.

---

## 📬 Contact

For questions about the dataset or code, please open a GitHub issue or contact the corresponding author:

**Nha Tran** — nhatt@hcmue.edu.vn  
Faculty of Information Technology, Ho Chi Minh City University of Education

---

## License and Terms of Use

The dataset is released for non-commercial academic research purposes only. Redistribution, commercial use, and attempts to re-identify participants are prohibited.

<div align="center">
<sub>© 2025 HCMUE-SEGL.</sub>
</div>
