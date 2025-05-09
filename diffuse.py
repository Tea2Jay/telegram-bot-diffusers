import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DPMSolverMultistepScheduler

# model_id = "CompVis/stable-diffusion-v1-4"
# model_id = "CompVis/stable-diffusion-v1-4"
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
        # scheduler: LMSDiscreteScheduler,
        dtype: torch.dtype,
        device: str = "cuda",
    ) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_auth_token=True,
            from_flax=True
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def diffuse(self, prompt: str):
        with autocast("cuda"):
            image = self.pipe(prompt,height=720,width=384)["images"][0]
        # image.save("astronaut_rides_horse.png")
        return image


if __name__ == '__main__':
    model_id = "stabilityai/stable-diffusion-2-1"
    device = "cuda"
    diffuser = Diffuser(model_id, torch.float16)
    # prompt = "An elf sorcerer opening their hands crackling with dark energy as shadows swirl around them, full body, trending in artstaion"
    prompt = "Drow Druid - a dark obsidian skinned elf with grey hair with her hair side braided over her shoulder and wearing a druidic outfit holding a gnarled oaken staff "
    for i in range(5):
        im = diffuser.diffuse(prompt)
        im.save(f"dnd/{prompt}_{i}.png")
    