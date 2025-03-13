import shutil
from functools import partial
import re

import cv2
import numpy as np
import torch

from utils import *
import socket
import os
import random
from torch.utils.data import Dataset
import glob
import scipy.ndimage as ndimage
import os.path as osp
from tqdm import tqdm
import json

rng = np.random.RandomState(234)

ORdirs = ['main_xml', 'main_xml1', 'mainDiffLight_xml', 'mainDiffLight_xml1', 'mainDiffMat_xml',
          'mainDiffMat_xml1']


class OpenroomsFF(Dataset):
    def __init__(self, dataRoot, cfg, phase, debug=False, debug_scenes=20000):
        self.rgb_suffix = 'hdr'
        hostname = socket.gethostname()
        if hostname == 'vigtitan168-MS-7B10':
            self.is_titan = True
            self.titan = '168'
        elif hostname == 'vigtitan-System-Product-Name':
            self.is_titan = True
            self.titan = '118'
        else:
            self.is_titan = False

        self.xy_offset = 1.3
        self.size = (cfg.imWidth, cfg.imHeight)
        self.nar_size = self.size
        self.env_size = (cfg.envCols, cfg.envRows)
        self.num_view_all = cfg.num_view_all
        self.dataRoot = dataRoot
        self.cfg = cfg
        self.mode = cfg.mode
        self.d_type = cfg.d_type

        self.K = None
        if hasattr(cfg, 'num_K'):
            self.K = cfg.num_K

        if self.mode == 'MG':
            self.image_key = ['i', 'cds', 'm', 'n', 'd', 'a', 'r']

        elif self.mode == 'incident' or self.mode == 'exitant':
            self.image_key = ['i', 'cds', 'm', 'cam', cfg.e_type]
            if self.mode == 'exitant':
                self.box_length = cfg.ExDLVSG.box_length
            if cfg.version == 'MAIR++':
                self.nar_size = self.env_size
                self.image_key += ['n', 'a', 'r', 'multi']

        elif self.mode == 'BRDF' or self.mode == 'AlbedoFusion':
            if cfg.version == 'MAIR':
                # feat map size.
                self.env_size = (160, 120)
            self.image_key = ['i', 'cds', 'm', 'cam', 'a', 'r', 'multi']
            if self.mode == 'AlbedoFusion':
                if hasattr(cfg.AlbedoFusion, 'is_rough') and cfg.AlbedoFusion.is_rough:
                    self.image_key.remove('a')
                else:
                    self.image_key.remove('r')

        elif self.mode == 'VSG':
            if cfg.VSGEncoder.vsg_type == 'voxel':
                # self.box_length = cfg.VSGDecoder.box_length
                self.box_length = 0
                x, y, z = np.meshgrid(np.arange(self.cfg.VSGEncoder.vsg_res),
                                      np.arange(self.cfg.VSGEncoder.vsg_res),
                                      np.arange(self.cfg.VSGEncoder.vsg_res // 2), indexing='xy')
                x = x.astype(dtype=np.float32) + 0.5  # add half pixel
                y = y.astype(dtype=np.float32) + 0.5
                z = z.astype(dtype=np.float32) + 0.5
                z = z / (self.cfg.VSGEncoder.vsg_res // 2)
                if self.box_length == 0:
                    x = self.xy_offset * (2.0 * x / self.cfg.VSGEncoder.vsg_res - 1)
                    y = self.xy_offset * (2.0 * y / self.cfg.VSGEncoder.vsg_res - 1)
                # elif self.box_length > 0:
                #     x = self.box_length * (2.0 * x / self.cfg.VSGEncoder.vsg_res - 1)
                #     y = self.box_length * (2.0 * y / self.cfg.VSGEncoder.vsg_res - 1)
                self.voxel_grid = [x, y, z]
            self.image_key = ['i', 'cds', 'm', 'cam', 'multi', 'e']
            # for GT re-rendering
            # print('\033[95m' + 'warning!! gt nar is loaded' + '\033[0m')
            # self.image_key = ['i', 'cds', 'm', 'cam', 'multi', 'e', 'n', 'a', 'r']

        self.phase = phase
        # scene type, scene name, image index, view index(1~9)
        self.inputList = self.load_txt()
        # print('\033[95m' + 'warning!! only main_xml is loaded' + '\033[0m')
        # self.inputList = [item for item in self.inputList if item[-1] == '5' and (item[0] == 'main_xml1' or item[0] == 'main_xml')]
        # self.inputList = [item for item in self.inputList if (item[0] == 'main_xml1' or item[0] == 'main_xml')]

        # elif self.phase == 'test':
        #     self.inputList = [item for item in self.inputList if item[-1] in ['1', '3', '5', '7', '9']]
        # if 'e' in self.image_key and self.phase != 'train':

        if not hasattr(cfg, 'full_load'):
            if (self.phase == 'test' and 'e' in self.image_key) or self.phase == 'val' or cfg.get('only_center_view',
                                                                                                  False):
                print(f'{phase}: dataset only center view!')
                self.inputList = [item for item in self.inputList if item[-1] == '5']

        # if self.phase == 'train':
        #     tmp_a = [item for item in self.inputList if item[0] == 'mainDiffMat_xml1' and
        #                       item[1] == 'scene0469_02' and
        #                       item[2] == '2']
        #     tmp_b = [item for item in self.inputList if item[0] == 'mainDiffMat_xml' and
        #              item[1] == 'scene0314_00' and
        #              item[2] == '4']
        #     self.inputList = tmp_a + tmp_b

        # if debug:
        #     self.inputList = self.inputList[:debug_scenes]
        self.length = len(self.inputList)

    def __len__(self):
        return self.length

    def load_txt(self):
        if self.phase == 'train':
            txt_file = osp.join(self.dataRoot, 'train.txt')
        elif self.phase == 'val':
            txt_file = osp.join(self.dataRoot, 'val.txt')
        elif self.phase == 'test':
            txt_file = osp.join(self.dataRoot, 'test.txt')
        else:
            raise Exception('Unrecognized phase for data loader')

        txt_file_processed = txt_file.replace('.', '_processed.')
        if osp.exists(txt_file_processed):
            with open(txt_file_processed, 'r') as f:
                inputList = json.load(f)
        else:
            with open(osp.join(self.dataRoot, txt_file), "r") as f:
                sceneList = f.read().splitlines()

            shapeList = [osp.join(self.dataRoot, d, x) for x in sceneList for d in ORdirs]
            inputList = []
            for shape in tqdm(shapeList):
                imNames = glob.glob(osp.join(shape, '*_immask_*.png'))
                inputList = inputList + imNames

            inputList = [x.replace('\\', '/').replace(self.dataRoot, '').split('/') for x in
                         sorted(list(set(inputList)))]
            inputList = [[x[1], x[2], x[3].split('_')[0], x[3].split('_')[2].split('.')[0]] for x in inputList]

            with open(txt_file_processed, 'w') as f:
                json.dump(inputList, f)
        return inputList

    def __getitem__(self, batch_id):
        name = self.inputList[batch_id]
        view_idx = int(name[3])

        name_ = osp.join(self.dataRoot, f'{name[0]}/{name[1]}/{name[2]}' + '_{}_' + f'{name[3]}' + '.{}')
        batch = {'name': osp.join(self.dataRoot, f'{name[0]}/{name[1]}/{name[2]}&{name[3]}'), }
        seg_name = name_.format('immask', 'png')
        seg_large = (loadImage(seg_name, type='s'))[..., :1]
        seg_small = cv2.resize(seg_large, self.size, interpolation=cv2.INTER_AREA)[:, :, None]
        mask = (seg_small > 0.9)
        mask = ndimage.binary_erosion(mask.squeeze(), structure=np.ones((5, 5)), border_value=1)[..., None]

        cam_mats = np.load((name_[:-5] + name_[-3:]).format('cam_mats', 'npy'))
        ratio = self.size[0] / cam_mats[1, 4, 0]
        cam_mats[:, 4, :] *= ratio
        max_depth = cam_mats[1, -1, view_idx - 1]
        for t in self.image_key:
            if t == 's':
                img = seg_small * 2.0 - 1
            elif t == 'i':
                img = loadImage(name_.format('im', self.rgb_suffix), t)
                scale = get_hdr_scale(img, seg_large > 0.9, self.phase)
                img = cv2.resize(img * scale, self.size, interpolation=cv2.INTER_AREA)
                img = np.clip(img, 0, 1.0)
            elif t == 'e_d':
                img = loadImage(name_.format('imenvDirect', 'hdr'), 'e_d') * scale
            elif t == 'e':
                load_zeros = False
                if self.is_titan:
                    if self.titan == '118':
                        load_zeros = True
                        print('\033[95m' + 'warning!! env is not loaded!!!' + '\033[0m')
                    elif self.titan == '168':
                        env_name = name_.format('imenvlow', 'hdr').replace('/home/vig-titan-168/Data/OpenRoomsFF320',
                                                                           '/media/vig-titan-168/Seagate Backup Plus Drive/OpenRoomsFF')
                else:
                    env_name = name_.format('imenvlow', 'hdr')
                if load_zeros:
                    img = np.ones([120, 160, 128, 3])
                else:
                    img = loadImage(env_name, 'e') * scale
            elif t == 'n':
                img = loadImage(name_.format('imnormal', 'png'), t, self.nar_size, normalize=True)
            elif t == 'cds':
                cds_depth = loadImage(name_.format('cdsdepthest', 'dat'), 'd', self.size,
                                      normalize=False).transpose([2, 0, 1])
                # for netdepth output!
                # batch['cds_depth'] = cds_depth
                batch['cds_dn'] = np.clip(cds_depth / max_depth, 0, 1)
                grad_x = cv2.Sobel(batch['cds_dn'][0], -1, 1, 0)
                grad_y = cv2.Sobel(batch['cds_dn'][0], -1, 0, 1)
                batch['cds_dg'] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)[None]
                batch['cds_conf'] = loadImage(name_.format('cdsconf', 'dat'), 'd', self.size,
                                              normalize=False).transpose([2, 0, 1])
            elif t == 'd':
                img = loadImage(name_.format('imdepth', 'dat'), 'd', self.size, normalize=False)
                img = img / max_depth
            elif t == 'cam':
                poses_hwf_bounds = cam_mats[..., view_idx - 1]
                h, w, f = poses_hwf_bounds[:, -2]
                assert self.size == (w, h)
                intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float).astype(np.float32)
                batch['cam'] = intrinsic
                if hasattr(self, 'box_length'):
                    if self.box_length == 0:
                        fov_x = intrinsic[0, 2] / intrinsic[0, 0]
                        fov_y = intrinsic[1, 2] / intrinsic[0, 0]
                        batch['bb'] = np.array([self.xy_offset * fov_x, self.xy_offset * fov_y, 1.05], dtype=np.float32)
                        if hasattr(self, 'voxel_grid'):
                            x = self.voxel_grid[0] * fov_x
                            y = self.voxel_grid[1] * fov_y
                            z = self.voxel_grid[2] * 1.05
                            batch['voxel_grid_front'] = np.stack([x, y, z], axis=-1)
                    else:
                        batch['bb'] = np.array([self.box_length, self.box_length, self.box_length], dtype=np.float32)
                        if hasattr(self, 'voxel_grid'):
                            batch['voxel_grid_front'] = np.stack(self.voxel_grid, axis=-1)

            elif t == 'a':
                img = loadImage(name_.format('imbaseColor', 'png'), t, self.nar_size, normalize=True)
            elif t == 'r':
                img = loadImage(name_.format('imroughness', 'png'), t, self.nar_size, normalize=True)
            elif t == 'm':
                img = mask.astype(np.float32)
            elif t == 'multi':
                view_idxs = [i for i in range(1, self.num_view_all + 1) if i != view_idx]
                if self.K is not None:
                    view_idxs = random.sample(view_idxs, self.K)
                view_idxs = [int(name[3]), ] + view_idxs
                name_m = osp.join(self.dataRoot, f'{name[0]}/{name[1]}/{name[2]}' + '_{}_{}.{}')
                src_c2w_list = []
                src_int_list = []
                rgb_list = []
                depth_list = []
                rgb_size = self.env_size

                if self.d_type == 'cds':
                    depth_name = 'cdsdepthest'
                elif self.d_type == 'net':
                    depth_name = 'netdepth'

                fac = self.env_size[1] / self.size[1]
                for idx in view_idxs:
                    im = loadImage(name_m.format('im', idx, self.rgb_suffix), 'i', rgb_size)
                    im = np.clip(im * scale, 0, 1.0)
                    rgb_list.append(im)
                    poses_hwf_bounds = cam_mats[..., idx - 1]
                    src_c2w_list.append(np34_to_44(poses_hwf_bounds[:, :4]))
                    h, w, f = poses_hwf_bounds[:, -2]
                    intrinsic = np.array([[f * fac, 0, w * fac / 2], [0, f * fac, h * fac / 2], [0, 0, 1]], dtype=float)
                    src_int_list.append(intrinsic)
                    depth = loadImage(name_m.format(depth_name, idx, 'dat'), 'd', self.env_size, False)
                    depth_list.append(depth)
                batch['all_i'] = np.stack(rgb_list, axis=0).transpose([0, 3, 1, 2])
                batch['all_cam'] = np.stack(src_int_list, axis=0).astype(np.float32)
                w2target = np.linalg.inv(src_c2w_list[0])
                batch['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)
                batch['all_depth'] = np.stack(depth_list, axis=0).transpose([0, 3, 1, 2])
            else:
                raise Exception('type Error')

            if img is not None:
                if img.ndim == 3:
                    img = img.transpose([2, 0, 1])
                    # if not ('multi' in self.image_key and t in ['d', 'd_gt']):
                # elif img.ndim == 4:
                #     img = img.transpose([3, 0, 1, 2])
                batch[t] = img
            img = None
        # n range is -1~1 and normalized
        # range is for 0~1 (rgb, d, c, a, r), 0~ (env), 0 or 1 (mask)
        return batch


# for netdepth of MGNet
class realworld_FF_singleview(Dataset):
    def __init__(self, dataRoot, cfg, img_w=320, img_h=240):
        self.img_w = img_w
        self.img_h = img_h
        self.cfg = cfg
        self.size = (img_w, img_h)
        # colmap depth is so big compared to openrooms(meter)
        self.max_depth_type = 'pose'
        self.depth_max_scale = 10.0

        sceneList = glob.glob(osp.join(dataRoot, '*'))
        tmp = []
        index_to_remove = []  #
        for i in range(len(sceneList)):
            if 'main_xml' in sceneList[i]:
                tmp += (glob.glob(osp.join(sceneList[i], '*')))
                index_to_remove.append(i)
        for index in reversed(index_to_remove):
            del sceneList[index]
        sceneList += tmp

        self.nameList = []
        tmp = f'images_{self.img_w}x{self.img_h}'
        for j, scene in enumerate(sceneList):
            if tmp in os.listdir(scene):
                tmp_list = glob.glob(osp.join(scene, tmp, 'im_*'))
                for t in tmp_list:
                    self.nameList.append(t)
                    # if not osp.exists(t.replace('.png', '.dat').replace('im_', 'netdepth_')):
                    #     self.nameList.append(t)
            else:
                tmp_list = glob.glob(osp.join(scene, '*_im_*'))
                for t in tmp_list:
                    self.nameList.append(t)
                    # if not osp.exists(t.replace('.rgbe', '.dat').replace('_im_', '_netdepth_')):
                    #     self.nameList.append(t)
        self.length = len(self.nameList)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        batch = {}
        name = self.nameList[ind]

        if name.endswith('.rgbe'):
            is_real = False
            cam_name = osp.join(osp.dirname(name), osp.basename(name).split('_')[0] + '_cam_mats.npy')
            cds_depth_name = name.replace('.rgbe', '.dat').replace('_im_', '_cdsdepthest_')
            cds_conf_name = name.replace('.rgbe', '.dat').replace('_im_', '_cdsconf_')
            view_idx = int(osp.basename(name).split('_')[2].split('.')[0]) - 1

            seg_name = name.replace('.rgbe', '.png').replace('im', 'immask')
            seg_large = (loadImage(seg_name, type='s'))[..., :1]
            seg_small = cv2.resize(seg_large, self.size, interpolation=cv2.INTER_AREA)[:, :, None]
            mask = (seg_small > 0.9)
            mask = ndimage.binary_erosion(mask.squeeze(), structure=np.ones((5, 5)), border_value=1)[..., None]
            mask = mask.astype(np.float32).transpose([2, 0, 1])

            img = loadImage(name, 'i')
            scale = get_hdr_scale(img, seg_large > 0.9, 'test')
            img = cv2.resize(img * scale, self.size, interpolation=cv2.INTER_AREA)
            img = np.clip(img, 0, 1.0).transpose([2, 0, 1])
        else:
            is_real = True
            im = cv2.imread(name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            im = im[..., ::-1].astype(np.float32) / 255.0
            img = ldr2hdr(im).transpose([2, 0, 1])
            mask = np.ones_like(img[:1])
            cam_name = osp.join(osp.dirname(name), 'cam_mats.npy')
            cds_depth_name = name.replace('.png', '.dat').replace('im_', 'cdsdepthest_')
            cds_conf_name = name.replace('.png', '.dat').replace('im_', 'cdsconf_')
            view_idx = int(osp.basename(name).split('_')[1].split('.')[0]) - 1

        cds_depth = loadImage(cds_depth_name, 'd', self.size, normalize=False).transpose([2, 0, 1])
        cds_conf = loadImage(cds_conf_name, 'd', self.size, normalize=False).transpose([2, 0, 1])
        if self.max_depth_type == 'pose':
            cam_mats = np.load(cam_name)
            ratio = self.size[0] / cam_mats[1, 4, 0]
            cam_mats[:, 4, :] *= ratio
            max_depth = cam_mats[1, -1, view_idx - 1]
        elif self.max_depth_type == 'est':
            target_conf = cds_conf > 0.6
            max_depth = np.max(target_conf * cds_depth)

        if is_real:
            # if scene's max depth is larger than depth_max_scale, we scale down depth.
            depth_scale = max(1.0, max_depth / self.depth_max_scale)
            cds_depth = cds_depth / depth_scale
            max_depth = max_depth / depth_scale

        batch['cds_depth'] = cds_depth
        batch['cds_conf'] = cds_conf
        batch['cds_dn'] = np.clip(cds_depth / max_depth, 0, 1)
        grad_x = cv2.Sobel(batch['cds_dn'][0], -1, 1, 0)
        grad_y = cv2.Sobel(batch['cds_dn'][0], -1, 0, 1)
        batch['cds_dg'] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)[None]
        batch['i'] = img
        batch['m'] = mask
        batch['name'] = cds_depth_name.replace('cdsdepthest', 'netdepth')
        return batch


class realworld_FF(Dataset):
    def __init__(self, dataRoot, cfg, img_w=320, img_h=240):
        self.img_w = img_w
        self.img_h = img_h
        self.cfg = cfg
        self.d_type = cfg.d_type
        self.env_size = (160, 120)
        self.size = (img_w, img_h)

        # colmap depth is so big compared to openrooms(meter)
        self.max_depth_type = 'pose'
        self.depth_max_scale = 10.0
        print(self.max_depth_type, self.depth_max_scale,
              'this must be same with realworld_FF_singleview(netdepth) value! ')

        sceneList = sorted(glob.glob(osp.join(dataRoot, '*')))
        outroot = osp.join(osp.dirname(dataRoot), f'output/{cfg.version}')
        tmp = []
        index_to_remove = []  #
        for i in range(len(sceneList)):
            if 'main_xml' in sceneList[i] or sceneList[i].endswith('oi_only'):
                tmp += sorted(glob.glob(osp.join(sceneList[i], '*')))
                index_to_remove.append(i)
        for index in reversed(index_to_remove):
            del sceneList[index]
        sceneList += tmp

        all_idx = []
        for i in range(9):
            all_idx.append(str(i + 1))
        all_idx.remove('5')

        self.nameList = []
        self.idx_list = []
        self.is_real = []
        self.outname = []

        self.xy_offset = 1.3
        x, y, z = np.meshgrid(np.arange(self.cfg.VSGEncoder.vsg_res),
                              np.arange(self.cfg.VSGEncoder.vsg_res),
                              np.arange(self.cfg.VSGEncoder.vsg_res // 2), indexing='xy')
        x = x.astype(dtype=np.float32) + 0.5  # add half pixel
        y = y.astype(dtype=np.float32) + 0.5
        z = z.astype(dtype=np.float32) + 0.5
        z = z / (self.cfg.VSGEncoder.vsg_res // 2)
        x = self.xy_offset * (2.0 * x / self.cfg.VSGEncoder.vsg_res - 1)
        y = self.xy_offset * (2.0 * y / self.cfg.VSGEncoder.vsg_res - 1)
        self.voxel_grid = [x, y, z]

        self.hdr_postfix = 'rgbe'
        for j, scene in enumerate(sceneList):
            if osp.exists(osp.join(scene, 'pair.txt')):
                # continue
                pair_file = osp.join(scene, 'pair.txt')
                with open(osp.join(scene, pair_file), 'r') as f:
                    num_viewpoint = int(f.readline().strip())
                    # viewpoints
                    for view_idx in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        if ref_view % 4 != 1:
                            continue
                        if len(src_views) == 0:
                            continue
                        # filter by no src view and fill to nviews
                        src_views = src_views[:8]

                        outfilename_org = osp.join(outroot, osp.basename(scene))
                        outfilename = f'{outfilename_org}_{(ref_view + 1):03d}'
                        os.makedirs(outfilename, exist_ok=True)
                        if cfg.version == 'MAIR++':
                            if len(os.listdir(outfilename)) == 20:
                                continue
                        if cfg.version == 'MAIR':
                            if len(os.listdir(outfilename)) == 12:
                                continue
                        self.nameList.append(scene + '$' + str(ref_view + 1))
                        self.idx_list.append(list(map(lambda x: str(x + 1), src_views)))
                        self.is_real.append(True)
                        self.outname.append(outfilename)

            else:
                a = sorted(list(set([b.split('_')[0] for b in os.listdir(scene)])))
                for t in a:
                    outfilename_org = osp.join(outroot, osp.basename(osp.dirname(scene)) + '_' + osp.basename(scene))
                    outfilename = f'{outfilename_org}_{int(t):03d}'
                    os.makedirs(outfilename, exist_ok=True)
                    if cfg.version == 'MAIR++':
                        if len(os.listdir(outfilename)) == 20:
                            continue
                    if cfg.version == 'MAIR':
                        if len(os.listdir(outfilename)) == 12:
                            continue
                    self.nameList.append(scene + '$' + t)
                    self.idx_list.append(all_idx)
                    self.is_real.append(False)
                    self.outname.append(outfilename)
        self.length = len(self.nameList)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        batch = {}
        training_idx = self.idx_list[ind].copy()
        is_real = self.is_real[ind]
        batch['outname'] = self.outname[ind]
        if is_real:
            scene, target_idx = self.nameList[ind].split('$')
            all_idx = [target_idx, ] + training_idx
            name_list = [osp.join(scene, 'images_320x240', '{}_' + f'{int(a):03d}' + '.{}') for a in all_idx]
            cam_name = osp.join(scene, 'images_320x240/cam_mats.npy')

            im = cv2.imread(name_list[0].format('im', 'png'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            im = im[..., ::-1].astype(np.float32) / 255.0
            im = ldr2hdr(im).transpose([2, 0, 1])

            cam_mats = np.load(cam_name)
        else:
            scene, scene_idx = self.nameList[ind].split('$')
            target_idx = '5'
            all_idx = [target_idx, ] + training_idx
            assert training_idx == ['1', '2', '3', '4', '6', '7', '8', '9']
            name_list = [osp.join(scene, scene_idx + '_{}_' + a + '.{}') for a in all_idx]
            cam_name = osp.join(scene, f'{scene_idx}_cam_mats.npy')

            seg_name = name_list[0].format('immask', 'png')
            seg_large = (loadImage(seg_name, type='s'))[..., :1]
            im = loadImage(name_list[0].format('im', self.hdr_postfix), 'i')
            scale = get_hdr_scale(im, seg_large > 0.9, 'test')
            im = cv2.resize(im * scale, self.size, interpolation=cv2.INTER_AREA)
            im = np.clip(im, 0, 1.0).transpose([2, 0, 1])

            cam_mats = np.load(cam_name)
            h, w, f = cam_mats[:, 4, 0]
            cam_mats[:, 4, :] = cam_mats[:, 4, :] / (w / self.size[0])

        batch['i'] = im
        batch['m'] = np.ones_like(im[:1])

        cds_conf_name = name_list[0].format('cdsconf', 'dat')
        cds_conf = loadImage(cds_conf_name, 'd', self.size, normalize=False).transpose([2, 0, 1])
        batch['cds_conf'] = cds_conf

        if self.max_depth_type == 'pose':
            # mvsd_pose : for openrooms or for oi and real-world
            max_depth = cam_mats[1, -1, int(target_idx) - 1].astype(np.float32)
        elif self.max_depth_type == 'est':
            # mvsd_est : for ir and real-world
            target_conf = loadBinary(name_list[0].format('cdsconf', 'dat'))
            target_conf = target_conf > 0.6
            target_depth = loadBinary(name_list[0].format('cdsdepthest', 'dat'))
            max_depth = np.max(target_conf * target_depth)

        cds_depth_name = name_list[0].format('cdsdepthest', 'dat')
        cds_depth = loadImage(cds_depth_name, 'd', self.size, normalize=False).transpose([2, 0, 1])
        batch['cds_dn'] = np.clip(cds_depth / max_depth, 0, 1)
        grad_x = cv2.Sobel(batch['cds_dn'][0], -1, 1, 0)
        grad_y = cv2.Sobel(batch['cds_dn'][0], -1, 0, 1)
        batch['cds_dg'] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)[None]

        poses_hwf_bounds = cam_mats[..., int(target_idx) - 1]
        h, w, f = poses_hwf_bounds[:, -2]
        intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float).astype(np.float32)
        batch['cam'] = intrinsic
        batch['hwf'] = np.array([h, w, f])

        if hasattr(self, 'voxel_grid'):
            fov_x = intrinsic[0, 2] / intrinsic[0, 0]
            fov_y = intrinsic[1, 2] / intrinsic[0, 0]
            batch['bb'] = np.array([self.xy_offset * fov_x, self.xy_offset * fov_y, 1.05], dtype=np.float32)
            x = self.voxel_grid[0] * fov_x
            y = self.voxel_grid[1] * fov_y
            z = self.voxel_grid[2] * 1.05
            batch['voxel_grid_front'] = np.stack([x, y, z], axis=-1)

        depth_scale = 1.0
        if is_real:
            # if scene's max depth is larger than depth_max_scale, we scale down depth.
            depth_scale = max(1.0, max_depth / self.depth_max_scale)
        cam_mats[:, 3, :] /= depth_scale

        src_c2w_list = []
        src_int_list = []
        rgb_list = []
        depthest_list = []
        fac = self.env_size[1] / self.size[1]
        for name, idx in zip(name_list, all_idx):
            if is_real:
                im = cv2.imread(name.format('im', 'png'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                im = cv2.resize(im, self.env_size, interpolation=cv2.INTER_AREA)
                im = ldr2hdr(im[..., ::-1].astype(np.float32) / 255.0)
            else:
                im = loadImage(name.format('im', self.hdr_postfix), 'i', self.env_size)
                im = np.clip(im * scale, 0, 1.0)
            rgb_list.append(im)

            poses_hwf_bounds = cam_mats[..., int(idx) - 1]
            src_c2w_list.append(np34_to_44(poses_hwf_bounds[:, :4]))
            cy2, cx2, fx = poses_hwf_bounds[:, -2]
            fy = poses_hwf_bounds[-1, -1]
            if fy == 0:
                fy = fx
            intrinsic = np.array([[fx * fac, 0, cx2 / 2 * fac], [0, fy * fac, cy2 / 2 * fac], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)
            if self.d_type == 'cds':
                depth = loadImage(name.format('cdsdepthest', 'dat'), 'd', self.env_size, False)
                depth = depth / depth_scale
            elif self.d_type == 'net':
                # netdepth is already divided by depth scale.
                depth = loadImage(name.format('netdepth', 'dat'), 'd', self.env_size, False)
            depthest_list.append(depth)

        batch['all_i'] = np.stack(rgb_list, axis=0).transpose([0, 3, 1, 2])
        batch['all_cam'] = np.stack(src_int_list, axis=0).astype(np.float32)
        w2target = np.linalg.inv(src_c2w_list[0])
        batch['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)
        batch['all_depth'] = np.stack(depthest_list, axis=0).transpose([0, 3, 1, 2])
        return batch