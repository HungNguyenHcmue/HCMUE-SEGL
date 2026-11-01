"""
dataloader.py — HCMUE-SEGL Dataset Loader
=============================================================

Updated loader for the NEW HCMUE-SEGL folder structure.
Only the 4-modality dataset is kept:
  1) RGB body frames        : 15 x 3 x 224 x 224
  2) Skeleton graph sequence: 15 PyG Data objects
  3) Face crops             : 15 x 3 x 224 x 224
  4) Group-context frame    : 3 x 224 x 224

Expected structure:

HCMUE-SEGL/
├── Angle1/
│   └── Table1/
│       ├── Context/
│       │   ├── X0_A1/Video18/id_2/context_frame.jpg
│       │   ├── X1_A1/...
│       │   ├── X2_A1/...
│       │   └── X3_A1/...
│       ├── Disengaged/
│       │   ├── Face/Video18/id_2/frame_001.png ... frame_015.png
│       │   └── X0_A1/Video18/id_2/frame_001.png ... frame_015.png + skeletons.csv
│       ├── Normal/
│       │   ├── Face/
│       │   └── X1_A1/
│       ├── Engaged/
│       │   ├── Face/
│       │   └── X2_A1/
│       └── Very Engaged/
│           ├── Face/
│           └── X3_A1/
├── Angle2/...
├── Angle3/...
└── Angle4/...

Usage:
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset
    from dataloader import MultiBranchDataset, mb_collate, split_by_angle

    ROOT = Path('/content/drive/MyDrive/HCMUE-SEGL')
    ANGLE_ROOTS = [ROOT / f"Angle{i}" / "Table1" for i in range(1, 5)]
    ds = MultiBranchDataset(ANGLE_ROOTS)

    train_ids, test_ids = split_by_angle(ds)
    train_loader = DataLoader(Subset(ds, train_ids), batch_size=4, shuffle=True,
                              collate_fn=mb_collate, num_workers=2)
    test_loader = DataLoader(Subset(ds, test_ids), batch_size=4, shuffle=False,
                            collate_fn=mb_collate, num_workers=2)
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_geometric.data import Data

# ──────────────────────────────────────────────
# 1. Constants & Default Transforms
# ──────────────────────────────────────────────

LABEL_MAP = {
    "Disengaged":   0,
    "Normal":       1,
    "Engaged":      2,
    "Very Engaged": 3,
}

# X-tag prefix per label, combined with angle suffix: X0 + _A1 -> X0_A1
X_PREFIX = {
    "Disengaged":   "X0",
    "Normal":       "X1",
    "Engaged":      "X2",
    "Very Engaged": "X3",
}

# 13-joint skeleton edge index (source -> target)
EDGE_INDEX = torch.tensor([
    [0,  0,  1,  1,  2,  2,  3,  3,  4, 5,  5,  5,  5, 6,  6,  6,  7,  7,  8,  8, 9, 10, 11, 11, 12, 12],
    [1,  2,  0,  3,  0,  4,  1,  5,  2, 3,  6,  7, 12, 5,  8, 11,  5,  9,  6, 10, 7,  8,  6, 12,  5, 11],
], dtype=torch.long)

IMG_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

NUM_FRAMES = 15
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")


# ──────────────────────────────────────────────
# 2. Path / frame helpers
# ──────────────────────────────────────────────

def angle_suffix(angle_name: str) -> str:
    """Convert Angle1 -> A1, Angle2 -> A2, ..."""
    if angle_name.startswith("Angle"):
        return "A" + angle_name[len("Angle"):]
    return angle_name


def make_angle_roots(root: Path, angles=("Angle1", "Angle2", "Angle3", "Angle4"), table="Table1"):
    """
    Build table-root paths:
        HCMUE-SEGL/Angle1/Table1
        HCMUE-SEGL/Angle2/Table1
        HCMUE-SEGL/Angle3/Table1
        HCMUE-SEGL/Angle4/Table1
    """
    root = Path(root)
    return [root / a / table for a in angles]


# Works with frame_001.png, frame001.png, id_2_frame146.jpg, etc.
_FRAME_NUM_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)


def extract_frame_number(path: Path) -> int:
    """Extract numeric frame index from a filename; unknown names go to the end."""
    m = _FRAME_NUM_RE.search(path.stem)
    return int(m.group(1)) if m else 10**12


def list_images(folder: Path):
    """List image files in a folder and sort them by frame number when possible."""
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(ext))

    # Sort first by extracted frame number, then by filename for stability.
    return sorted(files, key=lambda p: (extract_frame_number(p), p.name))


def select_context_frame(ctx_path: Path):
    """
    New structure expects exactly:
        context_frame.jpg
    But this helper is robust: it also accepts .png/.jpeg or falls back to
    the middle image if the folder contains multiple context images.
    """
    ctx_path = Path(ctx_path)
    if not ctx_path.exists():
        return None

    preferred_names = [
        "context_frame.jpg", "context_frame.jpeg", "context_frame.png",
        "context.jpg", "context.jpeg", "context.png",
    ]
    for name in preferred_names:
        p = ctx_path / name
        if p.exists():
            return p

    candidates = list_images(ctx_path)
    if not candidates:
        return None
    return candidates[len(candidates) // 2]


# ──────────────────────────────────────────────
# 3. Skeleton Feature Helpers
# ──────────────────────────────────────────────

def compute_node_features(xy: torch.Tensor) -> torch.Tensor:
    """
    Compute 4-dimensional node features from (x, y) coordinates.

    Args:
        xy: Tensor of shape (G, N, 2) — G graphs/frames, N joints, 2 coords.

    Returns:
        Tensor of shape (G, N, 4): [x, y, angle_x, angle_y]
    """
    G, N, _ = xy.shape
    f = torch.empty(G, N, 4, device=xy.device)

    for g in range(G):
        for n in range(N):
            x, y = xy[g, n]
            mag = (x * x + y * y).sqrt() + 1e-8
            rx = torch.clamp(x / mag, -1, 1)
            ry = torch.clamp(y / mag, -1, 1)
            f[g, n] = torch.tensor([x, y, rx.acos(), ry.acos()], device=xy.device)

    return f


def compute_edge_features(nf: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute 6-dimensional edge features from node features.

    Args:
        nf         : Tensor of shape (G, N, 4)
        edge_index : Tensor of shape (2, E)

    Returns:
        Tensor of shape (G, E, 6): [dx, dy, dthx, dthy, dist, angle]
    """
    G, E = nf.size(0), edge_index.size(1)
    out = torch.empty(G, E, 6, device=nf.device)

    for g in range(G):
        for ei in range(E):
            i, j = edge_index[:, ei]
            f0, f1 = nf[g, i], nf[g, j]
            dx, dy = f1[0] - f0[0], f1[1] - f0[1]
            dthx = f1[2] - f0[2]
            dthy = f1[3] - f0[3]
            dist = (dx * dx + dy * dy).sqrt()
            dot = f1[0] * f0[0] + f1[1] * f0[1]
            den = ((f1[0] ** 2 + f1[1] ** 2) *
                   (f0[0] ** 2 + f0[1] ** 2)).sqrt() + 1e-8
            ang = torch.clamp(dot / den, -1, 1).acos()
            out[g, ei] = torch.tensor([dx, dy, dthx, dthy, dist, ang], device=nf.device)

    return out


def _clean_skeleton_numeric_array(df: pd.DataFrame) -> np.ndarray:
    """
    Convert skeletons.csv to a numeric array of shape (T, 26).

    Expected final columns are 13 joints x 2 coordinates = 26 numbers.
    The helper tries to remove common non-coordinate numeric columns such as
    frame/frame_id/timestamp if they exist.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    drop_like = {"frame", "frame_id", "frame_idx", "timestamp", "time", "index"}
    cols_to_drop = [c for c in numeric_df.columns if str(c).strip().lower() in drop_like]
    if cols_to_drop:
        numeric_df = numeric_df.drop(columns=cols_to_drop)

    arr = numeric_df.values.astype(np.float32)

    if arr.shape[1] < 26:
        raise ValueError(
            f"skeletons.csv must contain at least 26 numeric coordinate columns, "
            f"but got {arr.shape[1]}."
        )

    # If extra numeric columns remain, keep the first 26 coordinate-like values.
    # This matches the common format: x1,y1,x2,y2,...,x13,y13.
    if arr.shape[1] > 26:
        arr = arr[:, :26]

    return arr


def skeleton_csv_to_graph_sequence(csv_path: Path, num_frames: int = NUM_FRAMES):
    """
    Read skeletons.csv and return a list of PyG Data objects, one graph per frame.
    """
    df = pd.read_csv(csv_path)
    arr = _clean_skeleton_numeric_array(df)

    if arr.shape[0] == 0:
        raise ValueError(f"Empty skeleton file: {csv_path}")

    # Pad or truncate to exactly num_frames.
    if arr.shape[0] < num_frames:
        pad = np.repeat(arr[-1][None], num_frames - arr.shape[0], axis=0)
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:num_frames]

    skel = torch.from_numpy(arr)               # (T, 26)
    nodes = skel.view(num_frames, 13, 2)        # (T, 13, 2)
    nf = compute_node_features(nodes)           # (T, 13, 4)
    ef = compute_edge_features(nf, EDGE_INDEX)  # (T, E, 6)

    return [
        Data(x=nf[t], edge_index=EDGE_INDEX, edge_attr=ef[t])
        for t in range(num_frames)
    ]


# ──────────────────────────────────────────────
# 4. MultiBranchDataset — 4 modalities only
# ──────────────────────────────────────────────

class MultiBranchDataset(Dataset):
    """
    Four-modality dataset for the new HCMUE-SEGL structure.

    Each sample returns:
        beh_imgs  : Tensor (15, 3, 224, 224)  — RGB body frames
        gseq      : List[Data] len=15          — skeleton graph sequence
        face_imgs : Tensor (15, 3, 224, 224)  — face crop frames
        ctx_img   : Tensor (3, 224, 224)       — group-context frame
        label     : int                        — engagement class index

    Args:
        angle_roots : list[Path]
            One path per camera angle, each pointing to Angle<N>/Table1/.
        transform : torchvision transform
            Transform applied to body, face and context images.
        strict : bool
            If True, raises an error when no samples are found.
    """

    def __init__(self, angle_roots, transform=IMG_TF, strict: bool = True):
        self.transform = transform
        self.samples = []
        self.skipped = defaultdict(int)

        for table_root in angle_roots:
            table_root = Path(table_root)
            angle_name = table_root.parent.name        # Angle1, Angle2, ...
            a_tag = angle_suffix(angle_name)           # A1, A2, ...
            ctx_root = table_root / "Context"

            if not table_root.exists():
                self.skipped["missing_table_root"] += 1
                continue

            for label_name, label_idx in LABEL_MAP.items():
                label_dir = table_root / label_name
                if not label_dir.is_dir():
                    self.skipped[f"missing_label_dir:{label_name}"] += 1
                    continue

                x_tag = f"{X_PREFIX[label_name]}_{a_tag}"   # X0_A1, X1_A1, ...
                rgb_root = label_dir / x_tag
                face_root = label_dir / "Face"
                ctx_label_root = ctx_root / x_tag

                if not rgb_root.is_dir():
                    self.skipped[f"missing_rgb_root:{x_tag}"] += 1
                    continue
                if not face_root.is_dir():
                    self.skipped[f"missing_face_root:{label_name}"] += 1
                    continue

                for video_dir in sorted(rgb_root.iterdir()):
                    if not video_dir.is_dir():
                        continue

                    for id_dir in sorted(video_dir.iterdir()):
                        if not id_dir.is_dir():
                            continue

                        skel_csv = id_dir / "skeletons.csv"
                        body_imgs = list_images(id_dir)

                        face_dir = face_root / video_dir.name / id_dir.name
                        face_imgs = list_images(face_dir)

                        ctx_dir = ctx_label_root / video_dir.name / id_dir.name
                        ctx_img = select_context_frame(ctx_dir)

                        if not skel_csv.exists():
                            self.skipped["missing_skeleton_csv"] += 1
                            continue
                        if len(body_imgs) < NUM_FRAMES:
                            self.skipped["body_frames_less_than_15"] += 1
                            continue
                        if len(face_imgs) < NUM_FRAMES:
                            self.skipped["face_frames_less_than_15_padding_zero"] += 1
                            face_imgs = []
                        else:
                            face_imgs = face_imgs[:NUM_FRAMES]
                        if ctx_img is None:
                            self.skipped["missing_context_frame"] += 1
                            continue

                        self.samples.append((
                            body_imgs[:NUM_FRAMES],
                            skel_csv,
                            face_imgs,
                            ctx_img,
                            label_idx,
                            video_dir.name,
                            id_dir.name,
                            angle_name,
                        ))

        print(f"MultiBranchDataset: {len(self.samples)} samples collected.")
        self._print_class_stats()
        self._print_skip_stats()

        if strict and len(self.samples) == 0:
            raise RuntimeError(
                "No samples were collected. Please check ROOT path, label folder names, "
                "X*_A* folders, Face folders, Context folders, and skeletons.csv files."
            )

    def _print_class_stats(self):
        counts = defaultdict(int)
        for sample in self.samples:
            label_idx = sample[4]
            counts[label_idx] += 1

        inv = {v: k for k, v in LABEL_MAP.items()}
        print("Class distribution:")
        for idx in sorted(inv):
            print(f"  {inv[idx]:15s}: {counts[idx]}")

    def _print_skip_stats(self):
        if not self.skipped:
            return
        print("Skipped summary:")
        for key, val in sorted(self.skipped.items()):
            print(f"  {key}: {val}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        body_imgs, skel_csv, face_imgs, ctx_img, label, video_name, id_name, angle_name = self.samples[index]

        # 1) RGB body branch: (15, 3, 224, 224)
        beh = torch.stack([
            self.transform(Image.open(p).convert("RGB"))
            for p in body_imgs
        ])

        # 2) Skeleton branch: list[PyG Data], len = 15
        gseq = skeleton_csv_to_graph_sequence(skel_csv)

        # 3) Face branch: (15, 3, 224, 224)
        if face_imgs is None or len(face_imgs) == 0:
            face = torch.zeros(NUM_FRAMES, 3, 224, 224)
        else:
            face = torch.stack([
                self.transform(Image.open(p).convert("RGB"))
                for p in face_imgs
            ])

        # 4) Context branch: (3, 224, 224)
        ctx = self.transform(Image.open(ctx_img).convert("RGB"))

        return beh, gseq, face, ctx, label


def mb_collate(batch):
    """Collate function for MultiBranchDataset."""
    behs, gseqs, faces, ctxs, labels = zip(*batch)
    return (
        torch.stack(behs),
        list(gseqs),
        torch.stack(faces),
        torch.stack(ctxs),
        torch.tensor(labels, dtype=torch.long),
    )


# ──────────────────────────────────────────────
# 5. Train / Test Split Utility
# ──────────────────────────────────────────────

def split_by_angle(dataset, test_size: float = 0.2, seed: int = 42):
    """
    Stratified train/test split per camera angle, so each angle appears in both sets.

    Returns:
        train_indices, test_indices
    """
    from sklearn.model_selection import train_test_split

    angle_to_indices = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        angle_name = sample[7]
        angle_to_indices[angle_name].append(i)

    train_ids, test_ids = [], []
    for angle_name, indices in angle_to_indices.items():
        labels = [dataset.samples[i][4] for i in indices]

        try:
            tr, ts = train_test_split(
                indices,
                test_size=test_size,
                stratify=labels,
                random_state=seed,
            )
        except ValueError:
            # Fallback when a class has too few samples in an angle for stratification.
            tr, ts = train_test_split(
                indices,
                test_size=test_size,
                shuffle=True,
                random_state=seed,
            )

        train_ids.extend(tr)
        test_ids.extend(ts)

    return train_ids, test_ids

# ──────────────────────────────────────────────
# 6. Quick-test Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ROOT = Path("/content/drive/MyDrive/HCMUE-SEGL")
    angle_roots = make_angle_roots(ROOT)

    ds4 = MultiBranchDataset(angle_roots)

    beh, gseq, face, ctx, label = ds4[0]
    print(
        f"\n[Single sample] beh={beh.shape}, "
        f"gseq_len={len(gseq)}, "
        f"face={face.shape}, "
        f"ctx={ctx.shape}, "
        f"label={label}"
    )

    train_ids, test_ids = split_by_angle(ds4)
    print(f"\n[Split] train={len(train_ids)}, test={len(test_ids)}")

    loader = DataLoader(ds4, batch_size=2, shuffle=False, collate_fn=mb_collate)
    batch = next(iter(loader))
    print(
        f"\n[Batch] beh={batch[0].shape}, "
        f"face={batch[2].shape}, "
        f"ctx={batch[3].shape}, "
        f"labels={batch[4]}"
    )
