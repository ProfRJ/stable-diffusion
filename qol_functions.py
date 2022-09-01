from torchvision.datasets.utils import download_url
from ldm.util import instantiate_from_config
import torch
import os
# todo ?
from google.colab import files
from IPython.display import Image as ipyimg
import ipywidgets as widgets
from PIL import Image
import numpy as np
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import ismap
import time
from omegaconf import OmegaConf
import requests
import PIL
from PIL import Image
from torchvision.transforms import functional as TF
import math
import subprocess

def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':' 
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight 
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights

def setres(image_shape, W, H):
    image_shape, _, _ = image_shape.partition(' |')
    return {
        "Custom": (W, H),
        "Square": (512, 512),
        "Large Square": (768, 768),
        "Landscape": (704, 512),
        "Large Landscape": (767, 640),
        "Portrait": (512, 704),
        "Large Portrait": (640, 768)  
    }.get(image_shape)

def get_output_folder(output_path,batch_folder=None):
    yearMonth = time.strftime('%Y-%m/')
    out_path = os.path.join(output_path,yearMonth)
    if batch_folder != "":
        out_path = os.path.join(out_path,batch_folder)
        # we will also make sure the path suffix is a slash if linux and a backslash if windows
        if out_path[-1] != os.path.sep:
            out_path += os.path.sep
    os.makedirs(out_path, exist_ok=True)
    return out_path

def load_img(path, shape):
    
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')

    fac = max(shape[0] / image.size[0], shape[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = TF.center_crop(image, shape[::-1])
    return 2.*image - 1.
    

def make_grid(images):
    mode = images[0].mode
    size = images[0].size

    n = len(images)
    x = math.ceil(n**0.5)
    y = math.ceil(n / x)

    output = Image.new(mode, (size[0] * x, size[1] * y))
    for i, image in enumerate(images):
        cur_x, cur_y = i % x, i // x
        output.paste(image, (size[0] * cur_x, size[1] * cur_y))
    return output


def get_gpu_information(image_size):
    memory = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    memory = memory.split(', ')[1].strip(' MiB')
    path = f'{os.getcwd()}/stable-diffusion/helpers/gpu-info'
    f = open(f"{path}/{memory}.txt","r")
    lines = f.readlines()
    max_samples = 0
    for line in lines:
        line = line.split(' | ')
        max_res = int(line[1])
        if max_res >= image_size:
            max_samples = int(line[0])
            continue
        else:

            break
    if max_samples == 0:
        raise error_message("Specified resolution is too large to fit on vram.")
    return max_samples


def split_batches_from_samples(n_samples, image_size):
    remaining_samples = n_samples
    batch_size_schedule = []
    max_samples = get_gpu_information(image_size)
    batch_sequences = int(math.ceil(n_samples/max_samples))
    for batch_sequence in range(batch_sequences):
        while remaining_samples > 0:
            if remaining_samples/max_samples <= 1:
                batch_size_schedule.append(remaining_samples)
                remaining_samples -= remaining_samples
            else:
                batch_size_schedule.append(max_samples)
                remaining_samples -= max_samples
    return (batch_sequences, batch_size_schedule)


def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32)
    return args.seed

class error_message(Exception):
       pass