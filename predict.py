import os
from typing import List
from PIL import Image
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    KandinskyImg2ImgPipeline,
    KandinskyPriorPipeline,
    KandinskyPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    DiffusionPipeline
)


MODEL_CACHE = "save_dir"


import torch

def to_fp16(func):
    def wrapper(*args, **kwargs):
        args = [arg.half() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {key: (value.half() if isinstance(value, torch.Tensor) else value) for key, value in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe_prior = DiffusionPipeline.from_pretrained(
            #"kandinsky-community/kandinsky-2-2-prior",
            os.path.join(MODEL_CACHE, "prior"),
            #cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.t2i_pipe = DiffusionPipeline.from_pretrained(
            #"kandinsky-community/kandinsky-2-2-decoder",
            #cache_dir=MODEL_CACHE,
            os.path.join(MODEL_CACHE, "decoder"),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        #print(dir(self.t2i_pipe))
        
        # movq is bfloat16 incompatible so cast to float16 and wrap decoder
        self.t2i_pipe.movq.to(torch.float16)
        self.t2i_pipe.movq.decode = to_fp16(self.t2i_pipe.movq.decode)
        
        # compile to speed up
        #self.t2i_pipe.unet.to(memory_format=torch.channels_last)
        #self.t2i_pipe.unet = torch.compile(self.t2i_pipe.unet, 
        #                                  )#mode="reduce-overhead", fullgraph=True)
        
        #self.i2i_pipe = KandinskyImg2ImgPipeline.from_pretrained(
        #    "kandinsky-community/kandinsky-2-1",
        #    cache_dir=MODEL_CACHE,
        #    local_files_only=True,
        #    torch_dtype=torch.float16,
        #).to("cuda")
        self.scheduler = "unipc"

    @torch.inference_mode()
    def predict(
        self,
        task: str = Input(
            description="Choose a task",
            choices=["text2img"],
            default="text2img",
        ),
        scheduler: str = Input(
            description="Choose a scheduler",
            choices=["dpm", "ddim", "unipc"],
            default="unipc",
        ),
        prompt: str = Input(
            description="Provide input prompt",
            default="A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output for text2img and text_guided_img2img tasks",
            default="ugly, tiling, oversaturated, undersaturated, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft",
        ),
        image: Path = Input(
            description="Input image for text2img",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Reduce the seeting if hits memory limits",
            ge=64,
            le=1024,
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Reduce the seeting if hits memory limits",
            ge=64,
            le=1024,
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_steps_prior: int = Input(
            description="Number of denoising steps in prior", ge=1, le=500, default=2
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", 
            ge=1, le=500, default=18,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=10, default=4.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        img_weight: float = Input(
            description="Weight of image - larger than 1 means more weight to image, lower than 0 is more weight to text", 
            default=1.0, ge=0.0, le=10.0,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # change scheduler
        if scheduler != self.scheduler:
            self.scheduler = scheduler
            if scheduler == "ddim":
                self.t2i_pipe.scheduler = DDIMScheduler.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="scheduler")
            elif scheduler == "dpm":
                self.t2i_pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="scheduler")
            elif scheduler == "unipc":
                self.t2i_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="scheduler")
        
        guidance_scale_prior = 2
        
        images_and_prompts = [prompt]
        
        weights = [1.0]
        if image is not None:
            original_image = Image.open(str(image)).convert("RGB")
            original_image = original_image.resize((width, height))
            images_and_prompts.append(original_image)
            
            weights = np.array([1.0, img_weight])
            weights = list(weights / weights.sum())
        
        image_embeds, negative_image_embeds = self.pipe_prior.interpolate(
            images_and_prompts,
            weights=weights,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_outputs,
            guidance_scale=guidance_scale_prior,
            num_inference_steps=num_steps_prior,
            generator=generator,
        ).to_tuple()

        images = self.t2i_pipe(
            #prompt=[prompt] * num_outputs,
            #negative_prompt=[negative_prompt] * num_outputs,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
   
        output_paths = []
        for i, img in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            img.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
