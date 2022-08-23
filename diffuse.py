import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


class Diffuser:
    def __init__(
        self,
        model_id: str,
        scheduler: LMSDiscreteScheduler,
        dtype: torch.dtype,
        device: str = "cuda",
    ) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=dtype,
            use_auth_token=True,
        ).to(device)

    def diffuse(self, prompt: str):
        with autocast("cuda"):
            image = self.pipe(prompt)["sample"][0]
        # image.save("astronaut_rides_horse.png")
        return image
