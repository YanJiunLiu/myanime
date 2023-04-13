from share import *
from cldm.hack import hack_everything

hack_everything(clip_skip=2)

import config
import cv2
import einops
import gradio as gr
import numpy as np
import torch

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
import random
import argparse
from PIL import Image
from numpy import asarray
import os

apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_any3_openpose.pth', location='cpu'))
model.cpu()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps,
            guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


def parse_args():
    """
    :return:
    input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode,
    strength, scale, seed, eta
    """
    desc = "Transform reality video to anime video by using CPU torch <control NET>"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, help='video file')
    parser.add_argument('--image', type=str, help='image file')
    parser.add_argument('--prompt', type=str, default="", help='prompt')
    parser.add_argument('--a_prompt', type=str, default="best quality, extremely detailed", help='prompt')
    parser.add_argument('--n_prompt', type=str, default="longbody, lowres, bad anatomy, bad hands, missing "
                                                        "fingers, extra digit, fewer digits, cropped, worst quality, "
                                                        "low quality", help='prompt')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--image_resolution', type=int, default=512)
    parser.add_argument('--detect_resolution', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--guess_mode', type=bool, default=False)
    parser.add_argument('--strength', type=int, default=1)
    parser.add_argument('--scale', type=int, default=9)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eta', type=int, default=0.0)

    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()
    if arg.seed:
        seed = arg.seed
    else:
        seed = random.randint(0, 2147483647)

    if arg.image:
        image = Image.open(arg.image)
        data = asarray(image)
        kwargs = {
            "input_image": data,
            "prompt": arg.prompt,
            "a_prompt": arg.a_prompt,
            "n_prompt": arg.n_prompt,
            "num_samples": arg.num_samples,
            "image_resolution": arg.image_resolution,
            "detect_resolution": arg.detect_resolution,
            "ddim_steps": arg.ddim_steps,
            "guess_mode": arg.guess_mode,
            "strength": arg.strength,
            "scale": arg.scale,
            "seed": seed,
            "eta": arg.eta
        }

        result = process(**kwargs)
        output = Image.fromarray(result[0])
        output.save(f"output.png")
    elif arg.video:
        cap = cv2.VideoCapture(arg.video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Totol frame: {length}")
        _output_video = os.path.join(os.path.dirname(__file__), f"output.mp4")

        img_width = int(cap.get(3))
        img_height = int(cap.get(4))
        size = (img_width, img_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWrite = cv2.VideoWriter(_output_video, fourcc, 30, size)
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                kwargs = {
                    "input_image": frame,
                    "prompt": arg.prompt,
                    "a_prompt": arg.a_prompt,
                    "n_prompt": arg.n_prompt,
                    "num_samples": arg.num_samples,
                    "image_resolution": arg.image_resolution,
                    "detect_resolution": arg.detect_resolution,
                    "ddim_steps": arg.ddim_steps,
                    "guess_mode": arg.guess_mode,
                    "strength": arg.strength,
                    "scale": arg.scale,
                    "seed": seed,
                    "eta": arg.eta
                }
                result = process(**kwargs)
                output = Image.fromarray(result[0])
                videoWrite.write(output)
            else:
                break
    else:
        raise NotImplementedError("Choose --images or --video")
# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Control Stable Diffusion with Human Pose")
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(source='upload', type="numpy")
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button(label="Run")
#             with gr.Accordion("Advanced options", open=False):
#                 num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#                 image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
#                 strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
#                 guess_mode = gr.Checkbox(label='Guess Mode', value=False)
#                 detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
#                 seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
#                 eta = gr.Number(label="eta (DDIM)", value=0.0)
#                 a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
#                 n_prompt = gr.Textbox(label="Negative Prompt",
#                                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
#         with gr.Column():
#             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
#
#     ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
#     run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
#
#
# block.launch(server_name='0.0.0.0')
