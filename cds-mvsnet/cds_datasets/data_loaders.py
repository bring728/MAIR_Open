from torch.utils.data import DataLoader
import cv2
import torch
import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

np.random.seed(1234)


class colmapDataset(Dataset):
    def __init__(self, datapath, nviews, ndepths=192, max_h=1, max_w=1, refine=False):
        super(colmapDataset, self).__init__()
        self.datapath = datapath
        self.nviews = nviews
        self.ndepths = ndepths
        self.max_h, self.max_w = max_h, max_w
        self.refine = refine
        self.metas = self.build_list()

    def build_list(self):
        metas = []  # {}
        scans = self.datapath
        for scan in scans:
            pair_file = "{}/pair.txt".format(osp.dirname(scan))
            # read the pair file
            if osp.exists(pair_file):
                with open(os.path.join(scan, pair_file)) as f:
                    num_viewpoint = int(f.readline())
                    # viewpoints
                    for view_idx in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                        # filter by no src view and fill to nviews
                        if len(src_views) > 0:
                            if len(src_views) < self.nviews:
                                print("{}< num_views:{}".format(len(src_views), self.nviews))
                                src_views += [src_views[0]] * (self.nviews - len(src_views))
                            src_views = src_views[:(self.nviews - 1)]
                            metas.append((scan, ref_view, src_views, scan))
            else:
                # for openrooms
                nviews = 9
                for view_idx in range(nviews):
                    view_idxs = list(range(nviews))
                    view_idxs.remove(view_idx)
                    metas.append((self.datapath, view_idx, view_idxs, self.datapath))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        # h, w = np_img.shape[:2]
        # np_img = cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        return np_img

    def scale_mvs_input(self, img, intrinsics, max_w, max_h):
        h, w = img.shape[:2]
        new_h, new_w = max_h, max_w

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        if scale_w != 1.0:
            print('resize')
            intrinsics[0, :] *= scale_w
            intrinsics[1, :] *= scale_h
            img = cv2.resize(img, (int(new_w), int(new_h)))
        return img, intrinsics

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, _ = meta
        view_ids = [ref_view] + src_views
        imgs = []
        poses = np.load(osp.join(scan, 'cam_mats.npy'))
        z_min, z_max = poses[:2, -1, ref_view]
        z_min *= 0.7
        z_max *= 2.0

        depth_interval = float(z_max - z_min) / self.ndepths
        depth_values = np.arange(z_min, depth_interval * (self.ndepths - 0.5) + z_min, depth_interval, dtype=np.float32)
        proj_matrices = []

        out_name = osp.join(scan, '{}' + f'_{view_ids[0] + 1:03d}' + '{}')
        for i, vid in enumerate(view_ids):
            if osp.exists(osp.join(scan, f'im_{(vid + 1):03d}.png')):
                img_filename = osp.join(scan, f'im_{(vid + 1):03d}.png')
            else:
                img_filename = f'{scan}imvis_{vid + 1}.jpg'
            img = self.read_img(img_filename)

            pose = poses[:3, :, vid]
            if not pose.shape[:2] == (3, 6):
                raise Exception(img_filename, ' cam mat shape error!')

            cy2, cx2, f = pose[:3, -2]
            if pose[-1, -1] != 0:
                fy = pose[-1, -1]
                intrinsics = np.array([[f, 0, cx2 / 2], [0, fy, cy2 / 2], [0, 0, 1]], dtype=float)
                intrinsics[:2, :] /= 4.0
            else:
                intrinsics = np.array([[f, 0, cx2 / 2], [0, f, cy2 / 2], [0, 0, 1]], dtype=float)
                intrinsics[:2, :] /= 4.0

            bottom = np.array([0, 0, 0, 1], dtype=float).reshape([1, 4])
            extrinsics = np.linalg.inv(np.concatenate([pose[:3, :4], bottom], 0))
            # extrinsics = np.concatenate([pose[:3, :4], bottom], 0)
            # print('warning extrinsic w2c!!')

            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)
            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5

        if self.refine:
            proj_matrices_ms = {
                "stage1": stage0_pjmats,
                "stage2": proj_matrices,
                "stage3": stage2_pjmats,
                "stage4": stage3_pjmats
            }
        else:
            proj_matrices_ms = {
                "stage1": proj_matrices,
                "stage2": stage2_pjmats,
                "stage3": stage3_pjmats
            }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": out_name}


class FFLoader(DataLoader):
    def __init__(self, data_list, num_srcs, num_depths, batch_size=1, num_worker=1, max_h=None, max_w=None, refine=True,
                 is_DDP=True):
        self.mvs_dataset = colmapDataset(data_list, num_srcs, num_depths, max_h, max_w, refine=refine)
        if is_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.mvs_dataset, shuffle=False)
            super().__init__(self.mvs_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True,
                             drop_last=False, sampler=train_sampler)
        else:
            super().__init__(self.mvs_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True,
                             drop_last=False, shuffle=False)
        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)


class singleDataset(Dataset):
    def __init__(self, datapath):
        super(singleDataset, self).__init__()
        self.datapath = datapath
        self.metas = self.build_list()

    def build_list(self):
        metas = []  # {}
        scans = self.datapath
        # scans
        for scan in scans:
            for view_idx in range(9):
                view_idxs = list(range(9))
                view_idxs.remove(view_idx)
                metas.append((scan, view_idx, view_idxs, scan))

        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        # key, real_idx = self.generate_img_index[idx]
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta

        img_filename = f'{scan}imvis_{ref_view + 1}.jpg'
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_filename = f'{scan}immask_{ref_view + 1}.png'
        mask = 0.5 * (loadImage(mask_filename) + 1)[0:1]
        mask = (mask > 0.55)
        return {"imgs": img,
                'mask': mask,
                "filename": scan + '{}' + f'_{ref_view + 1}' + '{}'}


class singleLoader(DataLoader):
    def __init__(self, data_list, batch_size=1, num_worker=1, ):
        self.mvs_dataset = singleDataset(data_list)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.mvs_dataset, shuffle=False)
        super().__init__(self.mvs_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True,
                         drop_last=False, sampler=train_sampler)
        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)
