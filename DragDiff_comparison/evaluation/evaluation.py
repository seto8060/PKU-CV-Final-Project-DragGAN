import os
import cv2
import csv
import pickle
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from dift_interface import DIFTExtractor
from metrics import ImageFidelity, BackgroundPreservation, MeanDistance

device = "cuda"

ROOTS = {
    "In": Path("dataset/In_Domain"),
    "Out": Path("dataset/Open_Domain"),
}

ORIG = "original_image.png"
META = "meta_data.pkl"
DRAGDIFF = "output_image_dragdiff.png"
DRAGGAN_GLOB = "output_image_draggan_ffhq_1.png"
RECON_GLOB = "recon_image_draggan_ffhq_1.png"
DRAGGAN_PREFIX = "output_image_draggan_"

# ---------------- utils ----------------
def load_image(p: Path):
    img = cv2.imread(str(p))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    return img.permute(2, 0, 1).unsqueeze(0).to(device)

def resize_like(img, ref):
    _, _, H, W = ref.shape
    return torch.nn.functional.interpolate(
        img, size=(H, W), mode="bilinear", align_corners=False
    )

def unpack_points(flat):
    pts = np.array(flat, dtype=np.float32)
    return pts[0::2], pts[1::2]

def parse_domain(p: Path):
    name = p.name[len(DRAGGAN_PREFIX):]
    return name[:-4] if name.endswith(".png") else name

def find_samples(root: Path):
    return [p for p, _, f in os.walk(root) if ORIG in f]

dift = DIFTExtractor()
IFm = ImageFidelity(device)
BPm = BackgroundPreservation(device)
MDm = MeanDistance(dift)

rows = []

# ---------------- main ----------------
for dom_type, root in ROOTS.items():
    if not root.exists():
        continue

    for sample_dir in map(Path, find_samples(root)):
        print(f"\n[Eval] {sample_dir}")

        I_orig = load_image(sample_dir / ORIG)

        with open(sample_dir / META, "rb") as f:
            meta = pickle.load(f)

        mask = torch.from_numpy(meta["mask"]).float().unsqueeze(0).unsqueeze(0).to(device)
        pts_src, pts_tgt = unpack_points(meta["points"])

        # ===== DragDiff =====
        I_diff = resize_like(load_image(sample_dir / DRAGDIFF), I_orig)

        IF_d = IFm(I_orig, I_diff)
        BP_d = BPm(I_orig, I_diff, mask)
        MD_d = MDm(I_orig, I_diff, pts_src, pts_tgt)

        print(
            f"[DragDiff][{dom_type}] {sample_dir.name} | "
            f"IF={float(IF_d):.4f} BP={float(BP_d):.4f} MD={float(MD_d):.2f}"
        )


        rows.append({
            "domain": dom_type,
            "method": "DragDiff",
            "class": sample_dir.name,
            "model": "-",
            "IF": float(IF_d),
            "BP": float(BP_d),
            "MD": float(MD_d),
        })

        # ===== DragGAN =====
        I_gan = resize_like(load_image(sample_dir / DRAGGAN_GLOB), I_orig)
        I_rec = resize_like(load_image(sample_dir / RECON_GLOB), I_orig)

        IF = IFm(I_orig, I_gan)
        BP = BPm(I_orig, I_gan, mask)
        MD = MDm(I_orig, I_gan, pts_src, pts_tgt)

        # If you want to select the closest match in pretrained models:
        
        # gan_imgs = list(sample_dir.glob(DRAGGAN_GLOB))
        # if not gan_imgs:
        #     continue

        # best = None  # (IF, BP, MD, model)

        # for g in gan_imgs:
        #     model = parse_domain(g)
        #     I_g = resize_like(load_image(g), I_orig)

        #     IF = IFm(I_orig, I_g)
        #     BP = BPm(I_orig, I_g, mask)
        #     MD = MDm(I_orig, I_g, pts_src, pts_tgt)

        #     if best is None or IF > best[0]:
        #         best = (IF, BP, MD, model)

        #     print(
        #         f"[DragGAN ][{dom_type}] {sample_dir.name} | "
        #         f"model={model} IF={float(IF):.4f} BP={float(BP):.4f} MD={float(MD):.2f}"
        #     )


        # IF_g, BP_g, MD_g, picked = best

        print(
            f"[DragGAN ][{dom_type}] {sample_dir.name} | "
            f"BEST model=ffhq_1 IF={float(IF):.4f} "
            f"BP={float(BP):.4f} MD={float(MD):.2f}"
        )


        rows.append({
            "domain": dom_type,
            "method": "DragGAN",
            "class": sample_dir.name,
            "model": "ffhq_1",
            "IF": float(IF),
            "BP": float(BP),
            "MD": float(MD),
        })

csv_path = "quantitative_results_orig_based.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["domain", "method", "class", "model", "IF", "BP", "MD"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[OK] Saved to {csv_path}")
