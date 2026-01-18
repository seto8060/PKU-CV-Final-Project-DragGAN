# ============================================================
# run_pti_dataset.py
# Multi-domain PTI inversion for In-Domain / Out-Domain eval
# ============================================================

import os
import shutil
from pathlib import Path
from PIL import Image
import pickle

import torch
from configs import paths_config, hyperparameters, global_config
from scripts.run_pti import run_PTI

PTI_ROOT = Path('/root/PTI')
DATASET_ROOT = Path('/root/dataset')

IMAGE_PROC = PTI_ROOT / 'image_processed'
IMAGE_PROC.mkdir(exist_ok=True)

# Domain -> StyleGAN mapping

STYLEGAN_BY_DOMAIN = {
    'cats': 'pretrained_models/cats.pkl',
    'ffhq_1': 'pretrained_models/ffhq_1.pkl',
    'lions': 'pretrained_models/lions.pkl',
    'horses': 'pretrained_models/horses.pkl',
    'elephants': 'pretrained_models/elephants.pkl',
    'dogs': 'pretrained_models/dogs.pkl',
}

ALL_DOMAINS = list(STYLEGAN_BY_DOMAIN.keys())

global_config.device = 'cuda'

paths_config.checkpoints_dir = str(PTI_ROOT)
paths_config.input_data_path = str(IMAGE_PROC)
paths_config.style_clip_pretrained_mappers = str(PTI_ROOT / 'pretrained_models')

paths_config.latent_space = 'w'

hyperparameters.use_locality_regularization = False
use_multi_id_training = False

def collect_samples():
    samples = []

    # -------- In-Domain --------
    in_root = DATASET_ROOT / 'In_Domain'
    if in_root.exists():
        for domain_dir in in_root.iterdir():
            domain = domain_dir.name
            if domain not in STYLEGAN_BY_DOMAIN:
                continue

            for sample_dir in domain_dir.iterdir():
                img = sample_dir / 'original_image.png'
                if img.exists():
                    samples.append({
                        'name': sample_dir.name,
                        'image_path': img,
                        'output_dir': sample_dir,
                        'is_in_domain': True,
                        'domain': domain
                    })

    # -------- Out-Domain --------
    out_root = DATASET_ROOT / 'Open_Domain'
    if out_root.exists():
        for sample_dir in out_root.iterdir():
            img = sample_dir / 'original_image.png'
            if img.exists():
                samples.append({
                    'name': sample_dir.name,
                    'image_path': img,
                    'output_dir': sample_dir,
                    'is_in_domain': False,
                    'domain': None
                })

    return samples


def get_gan_resolution(pkl_path: str) -> int:
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)['G_ema']
    res = int(getattr(G, 'img_resolution', None) or getattr(G, 'resolution', None))
    return res

def center_crop_to_square(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return pil_img.crop((left, top, left + m, top + m))

def run_pti_on_domain(sample, domain):
    sample_id = sample['name']
    print(f'\n[PTI] sample={sample_id}, domain={domain}')
    if domain != "ffhq_1" :
        return
        
    paths_config.stylegan2_ada_ffhq = STYLEGAN_BY_DOMAIN[domain]

    for p in IMAGE_PROC.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    dst_img = IMAGE_PROC / f'{sample_id}.jpeg'
    img = Image.open(sample['image_path']).convert('RGB')
    
    gan_pkl = paths_config.stylegan2_ada_ffhq
    res = get_gan_resolution(gan_pkl)
    
    img = center_crop_to_square(img)
    img = img.resize((res, res), Image.BICUBIC)
    img.save(dst_img, quality=95)

    safe_id = f'{sample_id}_{domain}'
    paths_config.input_data_id = safe_id

    run_PTI(
        use_wandb=False,
        use_multi_id_training=use_multi_id_training
    )

    ckpt = PTI_ROOT / f'model_{global_config.run_name}_{safe_id}.pkl'
    assert ckpt.exists(), f'PTI checkpoint not found: {ckpt}'
    
    pivot = PTI_ROOT / f'w_pivot_{global_config.run_name}_{safe_id}.pt'
    assert pivot.exists(), f'w_pivot not found: {pivot}'
    
    out_dir = sample['output_dir'] / domain
    out_dir.mkdir(exist_ok=True)
    
    shutil.move(pivot, out_dir / 'w_pivot.pt')
    print(f'[OK] Saved → {out_dir / "w_pivot.pt"}')

    shutil.move(ckpt, out_dir / 'model.pkl')
    print(f'[OK] Saved → {out_dir / "model.pkl"}')


def run_pti_for_sample(sample):
    if sample['is_in_domain']:
        domains_to_run = [sample['domain']]
    else:
        domains_to_run = ALL_DOMAINS

    for domain in domains_to_run:
        run_pti_on_domain(sample, domain)

if __name__ == '__main__':
    samples = collect_samples()
    print(f'[INFO] Found {len(samples)} samples')

    for sample in samples:
        run_pti_for_sample(sample)
