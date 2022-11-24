import argparse
import os
import time
from pathlib import Path

from diffusers import StableDiffusionPipeline
import torch


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--path_model",
        type=str,
        default=None,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Prompt",
    )
    parser.add_argument(
        '--sub_dir',
        type=str,
        default=None,
        required=False,
        help="Sub directory to save results in (output/sub_dir/prompt-time.png); otherwise (output/prompt-time.png)"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    model_id = args.path_model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = None

    prompt = args.prompt.strip()
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    path = Path(f"output")
    if args.sub_dir:
        path = Path(os.path.join(path, args.sub_dir))
    path.mkdir(parents=True, exist_ok=True)
    image.save(f"{str(path)}/{prompt.replace(' ', '-').replace(',','')}-{time.time()}.png")
