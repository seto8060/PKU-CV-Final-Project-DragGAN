import torch
from src.models.dift_sd import SDFeaturizer


class DIFTExtractor:
    """
    Thin wrapper of official DIFT SDFeaturizer.
    """

    def __init__(
        self,
        model_id="/root/stable-diffusion-v1-5",
        img_size=(768, 768),
        t=261,
        up_ft_index=1,
        prompt="",
        ensemble_size=8,
        device="cuda",
    ):
        self.img_size = img_size
        self.t = t
        self.up_ft_index = up_ft_index
        self.prompt = prompt
        self.ensemble_size = ensemble_size
        self.device = device

        self.dift = SDFeaturizer(model_id)

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        img = image[0].to(self.device)  # [3, H, W]

        # resize if needed (same logic as script)
        if self.img_size[0] > 0:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.img_size[::-1],  # (H, W)
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        ft = self.dift.forward(
            img,
            prompt=self.prompt,
            t=self.t,
            up_ft_index=self.up_ft_index,
            ensemble_size=self.ensemble_size,
        )

        ft = ft.squeeze(0).permute(1, 2, 0).contiguous()

        return ft
