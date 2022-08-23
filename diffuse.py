import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

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
            torch_dtype=dtype,
            use_auth_token=True,
        ).to(device)

    def diffuse(self, prompt: str):
        with autocast("cuda"):
            image = self.pipe(prompt,height=512,width=512)["sample"][0]
        # image.save("astronaut_rides_horse.png")
        return image


if __name__ == '__main__':
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    diffuser = Diffuser(model_id, lms, torch.float16)
    im = diffuser.diffuse("A beautiful painting of a dancing skeleton")
    im.save("tmp.png")
    