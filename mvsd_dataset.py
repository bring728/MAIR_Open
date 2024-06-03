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