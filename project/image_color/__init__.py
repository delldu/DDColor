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


def get_ddcolor_model():
    """Create model."""

    model = ddcolor_arch.DDColor()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    if 'cpu' in str(device.type):
        model.float()

    # print(f"Running on {device} ...")
    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_ddcolor.torch"):
    #     model.save("output/image_ddcolor.torch")
    # torch.save(model.state_dict(), "/tmp/image_ddcolor.pth")

    return model, device


def image_predict(grey_input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_ddcolor_model()

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

        ############################################################
        # model forward
        g_rgb_512 = F.interpolate(g_input_tensor,
            size=(512, 512),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        # model = model.half()
        # g_rgb_512 = g_rgb_512.half()
        out_ab = todos.model.forward(model, device, g_rgb_512)
        # out_ab = out_ab.float()
        out_ab = out_ab/128.0
        ############################################################

        out_ab = F.interpolate(out_ab,
            size=(H, W),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        out_lab = torch.cat((g_l, out_ab), dim=1)
        predict_tensor = color_space.lab2rgb(out_lab)

        output_file = f"{output_dir}/{os.path.basename(g_filename)}"

        # (Pdb) g_input_tensor.size() -- [1, 3, 678, 1020]
        # (Pdb) predict_tensor.size() -- [1, 3, 678, 1020]
        todos.data.save_tensor([g_input_tensor, predict_tensor], output_file)
