import h5py
import time
import nvidia_smi
import cv2
import numpy as np
import struct
from PIL import Image
import os.path as osp
import imageio
from skimage.measure import block_reduce
import torch
import torch.nn.functional as F
import PIL
import math
from cfgnode import CfgNode
import yaml

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision
eps = 1e-7
to_01 = lambda tmp: 0.5 * (torch.clamp(1.01 * torch.tanh(tmp), -1, 1) + 1)


def img_CHW2HWC(x):
    if torch.is_tensor(x):
        return x.permute(1, 2, 0)
    else:
        return np.transpose(x, (1, 2, 0))


img_rgb2bgr = lambda x: x[[2, 1, 0]]
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10. * np.log10(x + TINY_NUMBER)
psnr2mse = lambda x: 10 ** -(x / (10.0))
ldr2hdr = lambda x: x ** 2.2
hdr2ldr = lambda x: x ** (1.0 / 2.2)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def tocuda(vars, gpu, non_blocking=False):
    if isinstance(vars, list):
        out = []
        for var in vars:
            if isinstance(var, torch.Tensor):
                out.append(var.to(gpu, non_blocking=non_blocking))
            elif isinstance(var, str):
                out.append(var)
            else:
                raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
        return out
    elif isinstance(vars, dict):
        out = {}
        for k in vars:
            if isinstance(vars[k], torch.Tensor):
                out[k] = vars[k].to(gpu, non_blocking=non_blocking)
            elif isinstance(vars[k], str):
                out[k] = vars[k]
            elif isinstance(vars[k], list):
                out[k] = vars[k]
            else:
                raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
        return out


def nan_check(tensor):
    return torch.sum(torch.isnan(tensor))


def cv2fromtorch(tensor_or_array, normalize=False, to_ldr=False):
    if torch.is_tensor(tensor_or_array):
        if tensor_or_array.ndim == 4:
            tensor_or_array = tensor_or_array[0]
        if torch.min(tensor_or_array) < 0:
            tensor_or_array = 0.5 * (tensor_or_array + 1)
        if torch.max(tensor_or_array) > 1 or normalize:
            tensor_or_array = tensor_or_array / torch.max(tensor_or_array)
        if to_ldr:
            tensor_or_array = hdr2ldr(tensor_or_array)
        if tensor_or_array.shape[0] <= 4:
            tensor_or_array = img_CHW2HWC(tensor_or_array)
        image = (np.clip(tensor_or_array.detach().cpu().numpy(), 0, 1) * 255.0).astype(np.uint8)[..., ::-1]
    elif type(tensor_or_array).__module__ == np.__name__:
        if np.max(tensor_or_array) > 1:
            tensor_or_array = tensor_or_array / np.max(tensor_or_array)
        tensor_or_array = np.transpose(tensor_or_array, (1, 2, 0))
        image = np.clip((tensor_or_array * 255.0), 0, 255).astype(np.uint8)
    else:
        return None
    return image


class Timer(object):
    def __init__(self):
        self.time_list = [time.time()]

    def reset(self):
        self.time_list = [time.time()]

    def tic(self):
        self.time_list.append(time.time())

    # def toc(self):
    #     self.time_list.append(time.time())

    def __str__(self):
        time_str = ''
        for i in range(len(self.time_list) - 1):
            time_str += str(round(self.time_list[i + 1] - self.time_list[i], 1))
            time_str += ', '
        return time_str


def srgb2rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def np34_to_44(proj):
    if proj.ndim == 3:
        bottom = np.array([0, 0, 0, 1], dtype=float).reshape([1, 1, 4])
        bottom = np.repeat(bottom, proj.shape[0], axis=0)
        return np.concatenate([proj, bottom], 1)
    else:
        bottom = np.array([0, 0, 0, 1], dtype=float).reshape([1, 4])
        return np.concatenate([proj, bottom], 0)


def loadImage(imName, type, size=None, normalize=True):
    def img_proc(im, normalize_01=True):
        if size is not None and im.size[0] != size[0]:
            im = im.resize(size, PIL.Image.LANCZOS)
        im = np.asarray(im, dtype=np.float32)
        if normalize_01:
            im = im / 255.0
        else:
            im = (im - 127.5) / 127.5
        return im

    if type == 's':
        im = Image.open(imName)
        im = img_proc(im, normalize)
    elif type == 'i':
        im = cv2.imread(imName, -1)[..., ::-1]
        # im = np.array(imageio.v2.imread_v2(imName, format='HDR-FI'))
        if size is not None:
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    elif type == 'e':
        env = cv2.imread(imName, -1)[..., ::-1]
        env = env.reshape([120, 8, 160, 16, 3])
        env = env.transpose(0, 2, 1, 3, 4)  # 120 160 8 16 3
        im = env.reshape([120, 160, -1, 3])
    elif type == 'e_d':
        env = cv2.imread(imName, -1)[..., ::-1]
        env = env.reshape([30, 16, 40, 32, 3])
        env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))
        env = block_reduce(env, block_size=(1, 1, 1, 2, 2), func=np.mean)
        im = env.transpose(1, 2, 3, 4, 0)
    elif type == 'a':
        im = Image.open(imName)
        im = img_proc(im, True) ** 2.2
        if not normalize:
            im = im * 2 - 1
    elif type == 'r':
        im = Image.open(imName)
        im = img_proc(im, normalize)
        if len(im.shape) > 2:
            im = im[:, :, :1]
    elif type == 'n':
        im = Image.open(imName)
        im = img_proc(im, False)
        if normalize:
            im = im / np.sqrt(np.maximum(np.sum(im * im, axis=-1, keepdims=True), 1e-5))
    elif type == 'd':
        with open(imName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * width * height)
            depth = np.asarray(struct.unpack('f' * height * width, dBuffer), dtype=np.float32)
            if normalize:
                im = 2.0 * (depth.reshape([height, width]) / depth.max())[..., None] - 1
            else:
                im = depth.reshape([height, width])[..., None]
            if size is not None:
                im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)[..., None]
    return im


# img : B C H W
def saveImage(outname, img, is_hdr=False, is_single=False, size=(640, 480)):
    img = img[0].float().data.cpu().numpy().transpose([1, 2, 0])
    if is_hdr:
        img = hdr2ldr(img)

    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if not is_single:
        img = img[:, :, ::-1]
    cv2.imwrite(outname, img)


def get_hdr_scale(hdr, seg, phase):
    length = hdr.shape[0] * hdr.shape[1] * hdr.shape[2]
    intensityArr = (hdr * seg).flatten()
    intensityArr.sort()
    intensity_almost_max = np.clip(intensityArr[int(0.95 * length)], 0.1, None)

    if phase == 'train':
        x = 0.1 * np.random.random()  # 0.0 ~ 0.1
        scale = (0.95 - x) / intensity_almost_max
    elif phase == 'val' or phase == 'test':
        scale = 0.9 / intensity_almost_max
    else:
        raise Exception('phase type wrong')
    return scale


def loadBinary(imName):
    if not (osp.isfile(imName)):
        print(imName)
        assert (False)
    with open(imName, 'rb') as fIn:
        hBuffer = fIn.read(4)
        height = struct.unpack('i', hBuffer)[0]
        wBuffer = fIn.read(4)
        width = struct.unpack('i', wBuffer)[0]
        dBuffer = fIn.read(4 * width * height)
        depth = np.asarray(struct.unpack('f' * height * width, dBuffer))
        depth = depth.reshape([height, width])
    return depth[np.newaxis, :, :]


def saveBinary(filename, image):
    file = open(filename, "wb")
    h, w = image.shape
    data = struct.pack('i', h)
    file.write(data)
    data = struct.pack('i', w)
    file.write(data)
    depth = image.reshape(-1).astype(np.float32)
    depth.tofile(file)
    file.close()


def loadEnvmap(envName, env_height, env_width, env_rows, env_cols):
    env = np.array(imageio.imread_v2(envName, format='HDR-FI'))
    if not env is None:
        env = env.reshape(env_rows, env_height, env_cols, env_width, 3)
        env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))
        # env = env[::-1] # openrooms env is bgr format. so we should flip... but we flip all env file in advance
        return env
    else:
        raise Exception('env does not exist')


def resizeEnvmap(envName, env_height, env_width, env_rows, env_cols):
    env = np.array(imageio.imread_v2(envName, format='HDR-FI'))
    cv2.imwrite(envName, env)  # bgr 2 rgb save!
    if not env is None:
        env = env.reshape(env_rows, env_height, env_cols, env_width, 3)
        env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))
        env_org = block_reduce(env, block_size=(1, 1, 1, 2, 2), func=np.mean)
        env = np.ascontiguousarray(env_org.transpose([1, 3, 2, 4, 0]))
        env = env.reshape(120 * 8, 160 * 16, 3)

        envName_new = envName.replace('imenv', 'imenvlow').replace('data01', 'data02').replace('FF', 'FF_320')
        cv2.imwrite(envName_new, env)

        # env_new = np.array(imageio.imread_v2(envName_new, format='HDR-FI'))
        # env_new = env_new.reshape(env_rows, 8, env_cols, 16, 3)
        # env_new = np.ascontiguousarray(env_new.transpose([4, 0, 2, 1, 3]))
        # print()
    else:
        raise Exception('env does not exist')


# envmaps : b c h w env_h env_w
def writeEnvToFile(envmaps, envId, envName, nrows=24, ncols=16, envHeight=8, envWidth=16, gap=1):
    envmap = envmaps[envId, :, :, :, :, :].data.cpu().numpy()
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0])
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows)
    interX = int(envCol / ncols)

    lnrows = len(np.arange(0, envRow, interY))
    lncols = len(np.arange(0, envCol, interX))

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY):
        for c in range(0, envCol, interX):
            rId = int(r / interY)
            cId = int(c / interX)

            rs = rId * (envHeight + gap)
            cs = cId * (envWidth + gap)
            envmapLarge[rs: rs + envHeight, cs: cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (hdr2ldr(envmapLarge))).astype(np.uint8)
    cv2.imwrite(envName, envmapLarge[:, :, ::-1])


def envmapToShading(shadWeight, envmap):
    shading = envmap * shadWeight
    shading = torch.sum(shading, dim=[-1, -2])
    return shading


def LSregress(pred, gt, origin, mask, si_max=1000):
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
    nb = pred.size(0)
    origdim = pred.ndim - 1
    pred = pred.reshape(nb, -1)
    gt = gt.reshape(nb, -1)

    coef = (torch.sum(pred * gt, dim=1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5)).detach()
    coef = torch.clamp(coef, 1 / si_max, si_max)
    coef = coef.reshape(coef.shape + (1,) * origdim)
    predNew = origin * coef
    return predNew, coef


def LSDiffSpec_multiview(diff_all, spec_all, imOrig_all, diffOrig, specOrig, mask, proj_err, scale_type=0):
    mask = (F.adaptive_avg_pool2d(mask, (diff_all.shape[1], diff_all.shape[2])) > 0.9).float().permute([0, 2, 3, 1])

    diff = diff_all[..., 0, :]
    spec = spec_all[..., 0, :]
    imOrig = imOrig_all[..., 0, :]
    nb, nc, nh, nw = diff.size()

    # Mask out too bright regions
    mask_tmp = mask * (imOrig < 0.95).float()
    diff = diff * mask_tmp
    spec = spec * mask_tmp
    im = imOrig * mask_tmp

    diff = diff.reshape(nb, -1)
    spec = spec.reshape(nb, -1)
    im = im.reshape(nb, -1)

    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12
    b1 = torch.sum(diff * im, dim=1)
    b2 = torch.sum(spec * im, dim=1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22 - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-2)
    coef2 = coef2 / torch.clamp(frac, min=1e-2)

    if scale_type == 0:
        coef3 = torch.clamp(b1 / torch.clamp(a11, min=1e-5), 0.001, 1000)
        coef4 = coef3.clone() * 0

        frac = (frac / (nc * nh * nw)).detach()
        fracInd = (frac > 1e-2).float()
        cd = fracInd * coef1 + (1 - fracInd) * coef3
        cs = fracInd * coef2 + (1 - fracInd) * coef4
    else:
        cd = coef1
        cs = coef2
    cd = torch.clamp(cd, min=0, max=1000)[:, None, None, None, None]
    cs = torch.clamp(cs, min=0, max=1000)[:, None, None, None, None]

    diffscaled = cd * diffOrig
    specscaled = cs * specOrig
    # Do the regression twice to avoid clamping
    mask = (torch.abs(proj_err) < 0.05).float() * mask[..., None, :]
    rgb_rendered = torch.clamp(diffscaled + specscaled, 0, 1) * mask
    rgb_rendered = rgb_rendered.reshape(nb, -1)
    rgb_all = imOrig_all * mask
    coefIm = (torch.sum(rgb_rendered * rgb_all.reshape(nb, -1), dim=1) / torch.clamp(
        torch.sum(rgb_rendered * rgb_rendered, dim=1), min=1e-5)).detach()
    coefIm = torch.clamp(coefIm, 0.001, 1000)[:, None, None, None, None]
    diffscaled = coefIm * diffscaled
    specscaled = coefIm * specscaled
    return diffscaled, specscaled, mask


def LSDiffSpec_multiview_paper(diff_all, spec_all, imOrig_all, mask, proj_err, scale_type=0):
    mask = (F.adaptive_avg_pool2d(mask, (diff_all.shape[1], diff_all.shape[2])) > 0.9).float().permute([0, 2, 3, 1])

    diff = diff_all[..., 0, :]
    spec = spec_all[..., 0, :]
    imOrig = imOrig_all[..., 0, :]
    nb, nc, nh, nw = diff.size()

    # Mask out too bright regions
    mask_tmp = mask * (imOrig < 0.95).float()
    diff = diff * mask_tmp
    spec = spec * mask_tmp
    im = imOrig * mask_tmp

    diff = diff.reshape(nb, -1)
    spec = spec.reshape(nb, -1)
    im = im.reshape(nb, -1)

    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12
    b1 = torch.sum(diff * im, dim=1)
    b2 = torch.sum(spec * im, dim=1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22 - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-2)
    coef2 = coef2 / torch.clamp(frac, min=1e-2)

    if scale_type == 0:
        coef3 = torch.clamp(b1 / torch.clamp(a11, min=1e-5), 0.001, 1000)
        coef4 = coef3.clone() * 0

        frac = (frac / (nc * nh * nw)).detach()
        fracInd = (frac > 1e-2).float()
        cd = fracInd * coef1 + (1 - fracInd) * coef3
        cs = fracInd * coef2 + (1 - fracInd) * coef4
    else:
        cd = coef1
        cs = coef2
    cd = torch.clamp(cd, min=0, max=1000)[:, None, None, None, None]
    cs = torch.clamp(cs, min=0, max=1000)[:, None, None, None, None]

    diff_all = cd * diff_all
    spec_all = cs * spec_all
    # Do the regression twice to avoid clamping
    mask = (torch.abs(proj_err) < 0.05).float() * mask[..., None, :]
    rgb_rendered = torch.clamp(diff_all + spec_all, 0, 1) * mask
    rgb_rendered = rgb_rendered.reshape(nb, -1)
    rgb_all = imOrig_all * mask
    coefIm = (torch.sum(rgb_rendered * (rgb_all).reshape(nb, -1), dim=1) / torch.clamp(
        torch.sum(rgb_rendered * rgb_rendered, dim=1), min=1e-5))
    coefIm = torch.clamp(coefIm, 0.001, 1000)[:, None, None, None, None]
    rgb_diff = coefIm * diff_all
    rgb_spec = coefIm * spec_all
    return rgb_diff, rgb_spec, mask


def LSregressDiffSpec(diff, spec, imOrig, diffOrig, specOrig, mask=None, scale_type=0):
    nb, nc, nh, nw = diff.size()
    # Mask out too bright regions
    if mask is None:
        mask = (imOrig < 0.95).float()
    else:
        mask = (F.adaptive_avg_pool2d(mask, (nh, nw)) > 0.9).float()
        mask = mask * (imOrig < 0.95).float()

    diff = diff * mask
    spec = spec * mask
    im = imOrig * mask

    diff = diff.reshape(nb, -1)
    spec = spec.reshape(nb, -1)
    im = im.reshape(nb, -1)

    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12
    b1 = torch.sum(diff * im, dim=1)
    b2 = torch.sum(spec * im, dim=1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22 - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-2)
    coef2 = coef2 / torch.clamp(frac, min=1e-2)

    if scale_type == 0:
        coef3 = torch.clamp(b1 / torch.clamp(a11, min=1e-5), 0.001, 1000)
        coef4 = coef3.clone() * 0

        frac = (frac / (nc * nh * nw)).detach()
        fracInd = (frac > 1e-2).float()
        cd = fracInd * coef1 + (1 - fracInd) * coef3
        cs = fracInd * coef2 + (1 - fracInd) * coef4
    else:
        cd = coef1
        cs = coef2
    cd = torch.clamp(cd, min=0, max=1000).reshape(nb, 1, 1, 1)
    cs = torch.clamp(cs, min=0, max=1000).reshape(nb, 1, 1, 1)

    diffScaled = cd * diffOrig
    specScaled = cs * specOrig

    # Do the regression twice to avoid clamping
    renderedImg = torch.clamp(diffScaled + specScaled, 0, 1)
    renderedImg = renderedImg.reshape(nb, -1)
    imOrig = imOrig.reshape(nb, -1)
    coefIm = (torch.sum(renderedImg * imOrig, dim=1) / torch.clamp(torch.sum(renderedImg * renderedImg, dim=1),
                                                                   min=1e-5)).detach()
    coefIm = torch.clamp(coefIm, 0.001, 1000)
    coefIm = coefIm.reshape(nb, 1, 1, 1)

    diffScaled = coefIm * diffScaled
    specScaled = coefIm * specScaled
    return diffScaled, specScaled, cd, cs


def name2path(name):
    tmp = name.replace('-', '/').split('_')
    return tmp[0] + '_' + tmp[1] + '_' + tmp[2] + '{}' + tmp[3] + '{}'


def outdir2xml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/' + b[1].split('_')[0] + '_FF.xml'


def xml2camtxt(xml, k):
    scene_type = xml.split('/')[-1].split('_')[0]
    return f'{osp.dirname(xml)}/{k}_cam_{scene_type}_FF.txt'


def xml2outdir(xml):
    split = xml.split('.')[0].split('/')
    root = xml.split('scenes')[0] + 'data_FF_10_640'
    if 'mainDiffLight' in split[-1]:
        scene_type = 'mainDiffLight'
    elif 'mainDiffMat' in split[-1]:
        scene_type = 'mainDiffMat'
    else:
        scene_type = 'main'
    scene_name = split[-2]
    xml_name = split[-3]
    return osp.join(root, scene_type + '_' + xml_name, scene_name)


def camtxt2outdir(camtxt):
    split = camtxt.split('.')[0].split('/')
    root = camtxt.split('scenes')[0] + 'data_FF_10_640'
    if 'mainDiffLight' in split[-1]:
        scene_type = 'mainDiffLight'
    elif 'mainDiffMat' in split[-1]:
        scene_type = 'mainDiffMat'
    else:
        scene_type = 'main'
    scene_name = split[-2]
    xml_name = split[-3]
    return osp.join(root, scene_type + '_' + xml_name, scene_name)


def imglist2video(imglist, videoname, fps=15):
    h, w = cv2.imread(imglist[0]).shape[:2]
    video = cv2.VideoWriter(videoname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
    for jpg in imglist:
        img = cv2.imread(jpg)
        video.write(img)
    video.release()


# pip install nvidia-ml-py3
def wait_prev_work(gpus=None, sec=60, min_memory=90):
    if isinstance(gpus, str):
        gpus = gpus.split(',')
    gpus = [str(gpu) for gpu in gpus]

    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    while True:
        freem = []
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            if gpus is None:
                freem.append(int(100 * info.free / info.total))
            else:
                if str(i) in gpus:
                    freem.append(int(100 * info.free / info.total))

        print(freem)
        if min(freem) > min_memory:
            print('start!!!')
            break
        else:
            time.sleep(sec)
    nvidia_smi.nvmlShutdown()


def count_digits(input_string):
    digit_count = 0
    for char in input_string:
        if char.isdigit():
            digit_count += 1
    return digit_count


def env2cv(envmaps, envId=0, nrows=12, ncols=8, envHeight=16, envWidth=32, gap=1):
    envmap = envmaps[envId, :, :, :, :, :]
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0])
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows)
    interX = int(envCol / ncols)

    lnrows = len(np.arange(0, envRow, interY))
    lncols = len(np.arange(0, envCol, interX))

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY):
        for c in range(0, envCol, interX):
            rId = int(r / interY)
            cId = int(c / interX)

            rs = rId * (envHeight + gap)
            cs = cId * (envWidth + gap)
            envmapLarge[rs: rs + envHeight, cs: cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (envmapLarge ** (1.0 / 2.2))).astype(np.uint8)
    return envmapLarge


def np_endering(A, R, N, L, imHeight=120, imWidth=160, envWidth=16, envHeight=8, fov=57.9, F0=0.05):
    fov = fov / 180.0 * np.pi
    cameraPos = np.array([0, 0, 0], dtype=np.float32).reshape([1, 3, 1, 1])
    xRange = 1 * np.tan(fov / 2)
    yRange = float(imHeight) / float(imWidth) * xRange
    x, y = np.meshgrid(np.linspace(-xRange, xRange, imWidth),
                       np.linspace(-yRange, yRange, imHeight))
    y = np.flip(y, axis=0)
    z = -np.ones((imHeight, imWidth), dtype=np.float32)

    pCoord = np.stack([x, y, z]).astype(np.float32)  # right, up, back
    pCoord = pCoord[np.newaxis, :, :, :]
    v = cameraPos - pCoord  # viewdir is pixel to camera
    v = v / np.sqrt(np.maximum(np.sum(v * v, axis=1), 1e-12)[:, np.newaxis, :, :])
    v = v.astype(dtype=np.float32)

    up = np.array([0, 1, 0], dtype=np.float32)

    Az = ((np.arange(envWidth) + 0.5) / envWidth - 0.5) * 2 * np.pi
    El = ((np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El)
    Az = Az.reshape(-1, 1)
    El = El.reshape(-1, 1)

    lx = np.sin(El) * np.cos(Az)
    ly = np.sin(El) * np.sin(Az)
    lz = np.cos(El)
    ls = np.concatenate((lx, ly, lz), axis=1)

    envWeight = (np.sin(El) * np.pi * np.pi / envWidth / envHeight)[None, :, :, None, None]

    ldirections = ls[None, :, :, None, None]
    camyProj = np.einsum('b,abcd', up, N)[:, None] * N
    camy = up[None, :, None, None] - camyProj
    camy = (camy / np.sqrt(np.maximum(np.sum(camy * camy, axis=1, keepdims=True), 1e-5))).astype(np.float32)
    camx = np.cross(camy, N, axis=1)
    camx = -(camx / np.sqrt(np.maximum(np.sum(camx * camx, axis=1, keepdims=True), 1e-5))).astype(np.float32)

    l = ldirections[:, :, 0:1, :, :] * camx[:, None] \
        + ldirections[:, :, 1:2, :, :] * camy[:, None] \
        + ldirections[:, :, 2:3, :, :] * N[:, None]

    h = (v[:, None] + l) / 2
    h = h / np.sqrt(np.clip(np.sum(h * h, axis=2, keepdims=True), a_min=1e-6, a_max=None))

    vdh = np.sum((v * h), axis=2, keepdims=True)
    frac0 = F0 + (1 - F0) * np.power(2.0, (-5.55472 * vdh - 6.98316) * vdh)

    diffuseBatch = (A) / np.pi
    roughBatch = (R + 1.0) / 2.0

    k = (roughBatch + 1) * (roughBatch + 1) / 8.0
    alpha = roughBatch * roughBatch
    alpha2 = alpha * alpha

    ndv = np.clip(np.sum(N * v, axis=1, keepdims=True), 0, 1)[:, None]
    ndh = np.clip(np.sum(N[:, None] * h, axis=2, keepdims=True), 0, 1)
    ndl = np.clip(np.sum(N[:, None] * l, axis=2, keepdims=True), 0, 1)

    frac = alpha2[:, None] * frac0
    nom0 = ndh * ndh * (alpha2[:, None] - 1) + 1
    nom1 = ndv * (1 - k[:, None]) + k[:, None]
    nom2 = ndl * (1 - k[:, None]) + k[:, None]
    nom = np.clip(4 * np.pi * nom0 * nom0 * nom1 * nom2, 1e-6, 4 * np.pi)
    specPred = frac / nom

    envmap = L.reshape([1, 3, 120, 160, envWidth * envHeight])
    envmap = np.transpose(envmap, [0, 4, 1, 2, 3])

    brdfDiffuse = diffuseBatch[:, None] * ndl
    colorDiffuse = np.sum(brdfDiffuse * envmap * envWeight, axis=1)

    brdfSpec = specPred * ndl
    colorSpec = np.sum(brdfSpec * envmap * envWeight, axis=1)
    return colorDiffuse, colorSpec


def get_N2C(normal, up):
    assert normal.shape[-1] == 3
    origdim = normal.ndim - 1
    up = up.reshape((1,) * origdim + up.shape)
    camyProj = torch.sum(up * normal, dim=-1, keepdim=True) * normal
    camy = F.normalize(up - camyProj, dim=-1)
    camx = F.normalize(torch.cross(normal, camy, dim=-1), dim=-1)
    N2C = torch.stack([camx, camy, normal], dim=-1)
    return N2C


def closest_square_factors(num):
    sqrt_num = math.sqrt(num)
    factor1 = math.floor(sqrt_num)
    while num % factor1 != 0:
        factor1 -= 1
    factor2 = num // factor1
    return max(factor1, factor2), min(factor1, factor2)


class MouseHandler:
    def __init__(self):
        self.clicked_point = []

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:  # 좌클릭 이벤트인 경우
            self.clicked_point.append((x, y))

    def get_clicked_point(self):
        return self.clicked_point


def loadcfg(config):
    with open(config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    return cfg


def bounding_box(mask, chunk):
    nonzero = np.nonzero(mask)
    if len(nonzero[0]) == 0:
        return None

    l = np.min(nonzero[1])
    r = np.max(nonzero[1])
    t = np.min(nonzero[0])
    b = np.max(nonzero[0])
    return (l, r, t, b), split_box(l, r, t, b, chunk)


def split_box(left, right, top, bottom, area_per_box):
    boxes = []
    width = right - left
    total_area = (bottom - top) * width
    num_boxes = int(total_area // area_per_box)

    if num_boxes == 0:
        return [(left, right, top, bottom)]

    height_per_box = area_per_box / width
    current_top = top

    for i in range(num_boxes):
        current_bottom = current_top + height_per_box
        boxes.append((left, right, math.floor(current_top), math.floor(current_bottom)))
        current_top = current_bottom

    if current_top < bottom:
        boxes.append((left, right, math.floor(current_top), bottom))

    return boxes
