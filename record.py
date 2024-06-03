import torch

from utils import *
from tqdm import tqdm
import time
import datetime
from termcolor import colored
import torch.nn.functional as F
import wandb
import torch.distributed as dist
from cfgnode import CfgNode
import yaml
from einops import rearrange

colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']


def want_save_now(step):
    # if step <= 200:
    #     return step % 50 == 0
    if step <= 500:
        return step % 50 == 0
    elif step <= 3000:
        return step % 500 == 0
    elif step <= 10000:
        return step % 2000 == 0
    else:
        return step % 5000 == 0


def print_state(name, start_time, prev_time, max_it, global_step, print_period, total_loss, print_color=True):
    current_time = time.time()
    elapsed_sec = current_time - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    total_sec = elapsed_sec * max_it / global_step
    total_time = str(datetime.timedelta(seconds=total_sec)).split('.')[0]
    prog = global_step / max_it * 100
    sec_it = (current_time - prev_time) / print_period
    print_str = (f'{name} - step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, total:{total_time}, '
                 f'sec/it: {sec_it:.3f}, loss: {total_loss:.3f}')
    if print_color:
        print(colored(print_str, 'green'), end="\r")
    else:
        print(colored(print_str, 'yellow'))
    return current_time, sec_it


def extract_single_batch(data):
    out = {}
    for k in data:
        if data[k] is not None:
            if isinstance(data[k], tuple):
                out[k] = tuple([a[0] for a in data[k]])
            else:
                out[k] = data[k][0]
    return out


# data is BCHW
def vis_img(data, is_single=False, is_hdr=False, is_normal=False, normalize=False, is_env=False, size=None):
    if data.ndim > 3:
        data = data[:1]

    if is_env:
        data = torch.clamp(hdr2ldr(data), 0.0, 1.0)
        data = data / (torch.max(data) + TINY_NUMBER)
        _, r, c, l, ch = data.shape
        env_h = 8
        env_w = 16
        data = rearrange(data, 'b r c (h w) C -> b C (r h) (c w)', h=env_h, w=env_w)
    else:
        if is_hdr:
            data = torch.clamp(hdr2ldr(data), 0, 1.0)
        if is_single:
            data = data.expand([-1, 3, -1, -1])
        if is_normal:
            data = 0.5 * (data + 1)
        if normalize:
            data = data / data.max()

    if size is not None:
        data = F.interpolate(data, size=size)
    data = data[0]
    return data


def eval_model(model, loader, gpu, cfg, num_gpu, calculator, phase, step, is_training=False):
    flag = 'train' if is_training else 'test'
    model.switch_to_eval()
    scalars_to_log_all = {}
    scalars_to_log_mean = {}
    total_sample_num = 0
    with torch.no_grad():
        for i, val_data in enumerate(tqdm(loader)):
            names = val_data['name']
            bn = len(val_data['name'])
            val_data = tocuda(val_data, gpu, False)
            pred, gt, mask = model.forward(val_data, cfg, flag)
            _, scalars_to_log = calculator(val_data, pred, gt, mask, phase, step)
            for k in scalars_to_log:
                if k in scalars_to_log_all:
                    scalars_to_log_all[k].append(bn * scalars_to_log[k])
                else:
                    scalars_to_log_all[k] = [bn * scalars_to_log[k], ]
            total_sample_num += bn

    for val_k in scalars_to_log_all:
        scalars_to_log_mean[val_k] = sum(scalars_to_log_all[val_k]) / total_sample_num

    if num_gpu > 1:
        device = torch.device('cuda:{}'.format(gpu))
        tensor_list = [torch.zeros(len(scalars_to_log_mean), dtype=torch.float, device=device) for _ in range(num_gpu)]
        tensor_here = torch.zeros(len(scalars_to_log_mean), dtype=torch.float, device=device)

        scalars_to_log_ddp = sorted(scalars_to_log_mean.items())
        for i, scalar in enumerate(scalars_to_log_ddp):
            tensor_here[i] = scalar[1]

        dist.all_gather(tensor_list, tensor_here)
        tensor_mean = torch.mean(torch.stack(tensor_list, dim=0), dim=0)

        scalars_to_log_mean = {}
        for i, scalar in enumerate(scalars_to_log_ddp):
            scalars_to_log_mean[scalar[0]] = round(tensor_mean[i].item(), 6)

    model.switch_to_train()
    return scalars_to_log_mean, scalars_to_log_all
