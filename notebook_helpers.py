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
    return max_samples


def split_batches_from_samples(n_samples, image_size):
    remaining_samples = n_samples
    batch_size_schedule = []
    max_samples = get_gpu_information(image_size)
    batch_sequences = int(math.ceil(n_samples/max_samples))
    for batch_sequence in range(batch_sequences):
        while remaining_samples > 1:
            if remaining_samples/max_samples <= 1:
                batch_size_schedule.append(remaining_samples)
                remaining_samples -= remaining_samples
            else:
                batch_size_schedule.append(max_samples)
                remaining_samples -= max_samples
    return (batch_sequences, batch_size_schedule)


def download_models(mode):

    if mode == "superresolution":
        # this is the small bsr light model
        url_conf = 'https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1'
        url_ckpt = 'https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1'

        path_conf = 'logs/diffusion/superresolution_bsr/configs/project.yaml'
        path_ckpt = 'logs/diffusion/superresolution_bsr/checkpoints/last.ckpt'

        download_url(url_conf, path_conf)
        download_url(url_ckpt, path_ckpt)

        path_conf = path_conf + '/?dl=1' # fix it
        path_ckpt = path_ckpt + '/?dl=1' # fix it
        return path_conf, path_ckpt

    else:
        raise NotImplementedError


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step


def get_model(mode):
    path_conf, path_ckpt = download_models(mode)
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model


def get_custom_cond(mode):
    dest = "data/example_conditioning"

    if mode == "superresolution":
        uploaded_img = files.upload()
        filename = next(iter(uploaded_img))
        name, filetype = filename.split(".") # todo assumes just one dot in name !
        os.rename(f"{filename}", f"{dest}/{mode}/custom_{name}.{filetype}")

    elif mode == "text_conditional":
        w = widgets.Text(value='A cake with cream!', disabled=True)
        display(w)

        with open(f"{dest}/{mode}/custom_{w.value[:20]}.txt", 'w') as f:
            f.write(w.value)

    elif mode == "class_conditional":
        w = widgets.IntSlider(min=0, max=1000)
        display(w)
        with open(f"{dest}/{mode}/custom.txt", 'w') as f:
            f.write(w.value)

    else:
        raise NotImplementedError(f"cond not implemented for mode{mode}")


def get_cond_options(mode):
    path = "data/example_conditioning"
    path = os.path.join(path, mode)
    onlyfiles = [f for f in sorted(os.listdir(path))]
    return path, onlyfiles


def select_cond_path(mode):
    path = "data/example_conditioning"  # todo
    path = os.path.join(path, mode)
    onlyfiles = [f for f in sorted(os.listdir(path))]

    selected = widgets.RadioButtons(
        options=onlyfiles,
        description='Select conditioning:',
        disabled=False
    )
    display(selected)
    selected_path = os.path.join(path, selected.value)
    return selected_path


def get_cond(mode, selected_path):
    example = dict()
    if mode == "superresolution":
        up_f = 4
        visualize_cond_img(selected_path)

        c = Image.open(selected_path)
        c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
        c_up = rearrange(c_up, '1 c h w -> 1 h w c')
        c = rearrange(c, '1 c h w -> 1 h w c')
        c = 2. * c - 1.

        c = c.to(torch.device("cuda"))
        example["LR_image"] = c
        example["image"] = c_up

    return example


def visualize_cond_img(path):
    display(ipyimg(filename=path))


def run(model, selected_path, task, custom_steps, resize_enabled=False, classifier_ckpt=None, global_step=None):

    example = get_cond(task, selected_path)

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.
    eta = 1.
    make_progrow = True
    custom_shape = None

    height, width = example["image"].shape[1:3]
    split_input = height >= 128 and width >= 128

    if split_input:
        ks = 128
        stride = 64
        vqf = 4  #
        model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
    else:
        if hasattr(model, "split_input_params"):
            delattr(model, "split_input_params")

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False , masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):

    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,ddim_use_x0_pred=False):
    log = dict()

    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log
