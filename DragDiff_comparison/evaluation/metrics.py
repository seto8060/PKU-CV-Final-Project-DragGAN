import torch
import lpips

def image_to_feature_coords(x, y, H, W, Hf, Wf):
    fx = int(x / W * Wf)
    fy = int(y / H * Hf)
    fx = max(0, min(Wf - 1, fx))
    fy = max(0, min(Hf - 1, fy))
    return fx, fy


def feature_to_image_coords(fx, fy, H, W, Hf, Wf):
    x = fx * W / Wf
    y = fy * H / Hf
    return x, y

class ImageFidelity:
    def __init__(self, device='cuda', net='vgg'):
        self.lpips = lpips.LPIPS(net=net).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, I_orig, I_edit):
        d = self.lpips(
            I_orig.to(self.device),
            I_edit.to(self.device)
        )
        return float(1.0 - d.item())

class BackgroundPreservation:
    def __init__(self, device='cuda', net='vgg'):
        self.lpips = lpips.LPIPS(net=net).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, I_orig, I_edit, mask):
        bg_mask = 1.0 - mask
        I_orig_bg = I_orig * bg_mask
        I_edit_bg = I_edit * bg_mask

        d = self.lpips(
            I_orig_bg.to(self.device),
            I_edit_bg.to(self.device)
        )
        return float(1.0 - d.item())

class MeanDistance:
    def __init__(self, dift_extractor):
        self.dift = dift_extractor

    @torch.no_grad()
    def __call__(self, I_orig, I_edit, points_src, points_tgt):
        assert len(points_src) == len(points_tgt)

        _, _, H, W = I_orig.shape

        feat_orig = self.dift.extract(I_orig)
        feat_edit = self.dift.extract(I_edit)

        Hf, Wf, C = feat_orig.shape

        feat_edit_flat = feat_edit.view(-1, C)

        total_dist = 0.0

        for (px, py), (tx, ty) in zip(points_src, points_tgt):
            fx, fy = image_to_feature_coords(px, py, H, W, Hf, Wf)
            f = feat_orig[fy, fx]  # [C]

            # nearest neighbor in feature space
            dists = torch.norm(feat_edit_flat - f[None, :], dim=1)
            idx = torch.argmin(dists)

            ey = idx // Wf
            ex = idx % Wf

            ex_img, ey_img = feature_to_image_coords(
                ex.item(), ey.item(), H, W, Hf, Wf
            )

            dist = ((ex_img - tx) ** 2 + (ey_img - ty) ** 2) ** 0.5
            total_dist += dist

        return total_dist / len(points_src)