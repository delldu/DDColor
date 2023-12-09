"""Image Color Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021-2024(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import todos
from . import color_space, ddcolor_arch

import pdb


def get_color_model():
    """Create model."""

    model_path = "models/image_color.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = ddcolor_arch.DDColor()

    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # model = torch.jit.script(model)

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_color.torch"):
    #     model.save("output/image_color.torch")

    return model, device


def image_predict(grey_input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_color_model()

    # load files
    grey_filenames = todos.data.load_files(grey_input_files)

    # start predict
    progress_bar = tqdm(total=len(grey_filenames))
    for g_filename in grey_filenames:
        progress_bar.update(1)

        # orig input
        g_input_tensor = todos.data.load_tensor(g_filename)
        B, C, H, W = g_input_tensor.size()
        g_lab = color_space.rgb2lab(g_input_tensor)
        g_l = g_lab[:, 0:1, :, :].clone()
        g_lab[:, 1:3, :, :] = 0.0
        g_rgb = color_space.lab2rgb(g_lab)

        # g_rgb resize to (512, 512)
        g_rgb = F.interpolate(g_rgb,
            size=(512, 512),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        out_ab = todos.model.forward(model, device, g_rgb)/128.0

        out_ab = F.interpolate(out_ab,
            size=(H, W),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        out_lab = torch.cat((g_l, out_ab), dim=1)
        predict_tensor = color_space.lab2rgb(out_lab)

        output_file = f"{output_dir}/{os.path.basename(g_filename)}"

        todos.data.save_tensor([g_input_tensor, predict_tensor], output_file)
