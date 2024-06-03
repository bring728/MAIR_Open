import os
from torch.optim import lr_scheduler
from network.net_custom import *
from utils import *
from network.MVANet import AggregationNet
import os.path as osp
from einops import rearrange
from functools import reduce

g_sam = lambda x, y: F.grid_sample(x, y, align_corners=False, mode='bilinear')
g_sam_nn = lambda x, y: F.grid_sample(x, y, align_corners=False, mode='nearest')


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


def count_params(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


class Model(object):
    def __init__(self, cfg, gpu, experiment, is_train, is_DDP=True):
        self.cfg = cfg
        self.gpu = gpu
        self.is_DDP = is_DDP
        device = torch.device('cuda:{}'.format(gpu))
        root = osp.dirname(experiment)
        run_id = osp.basename(experiment)
        self.re_arr = lambda tmp: rearrange(tmp[..., :1, :], 'b h w v c-> b (v c) h w')
        self.empty = torch.tensor([], device=gpu)

        u, v = np.meshgrid(np.arange(cfg.imWidth), np.arange(cfg.imHeight), indexing='xy')
        u = u.astype(dtype=np.float32) + 0.5  # add half pixel
        v = v.astype(dtype=np.float32) + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
        pixels = torch.from_numpy(pixels)
        pixels = pixels.to(gpu, non_blocking=cfg.pinned)[None, :, :, :, None]
        self.pixels = pixels

        up = torch.from_numpy(np.array([0, 1.0, 0], dtype=np.float32))
        self.up = up.to(gpu, non_blocking=cfg.pinned)

        rub_to_rdf = torch.from_numpy(np.array([1.0, -1.0, -1.0], dtype=np.float32)).to(gpu, non_blocking=cfg.pinned)
        self.rub_to_rdf = rub_to_rdf
        if hasattr(cfg, 'envCols'):
            self.envCols = cfg.envCols
            self.envRows = cfg.envRows
            self.env_width = cfg.env_width
            self.env_height = cfg.env_height

        if hasattr(cfg, 'env_width'):
            env_width = cfg.env_width
            env_height = cfg.env_height
            # azimuth, elevation
            Az_org = ((np.arange(env_width) + 0.5) / env_width - 0.5) * 2 * np.pi  # -pi ~ pi
            El_org = ((np.arange(env_height) + 0.5) / env_height) * np.pi / 2.0  # 0 ~ pi/2
            Az, El = np.meshgrid(Az_org, El_org)
            lx_dir = np.sin(El) * np.cos(Az)
            ly_dir = np.sin(El) * np.sin(Az)
            lz_dir = np.cos(El)
            # Az_flat = Az.reshape(-1, 1)
            El_flat = El.reshape(-1, 1)
            # lx_dir = np.sin(El_flat) * np.cos(Az_flat)
            # ly_dir = np.sin(El_flat) * np.sin(Az_flat)
            # lz_dir = np.cos(El_flat)
            ls = np.stack((lx_dir, ly_dir, lz_dir), axis=-1).astype(np.float32)
            ls = np.concatenate([ls, ls[:, -2:-1, :]], axis=1)
            ls = np.concatenate([ls, ls[-2:-1, :, :]], axis=0)
            env_dx_norm = np.linalg.norm(np.diff(ls, axis=1), axis=-1)[:env_height, :env_width]
            env_dy_norm = np.linalg.norm(np.diff(ls, axis=0), axis=-1)[:env_height, :env_width]
            radii = (0.5 * (env_dx_norm + env_dy_norm)).reshape([-1, 1]) * 2 / np.sqrt(12)
            self.radii = torch.from_numpy(radii).to(gpu, non_blocking=cfg.pinned)[None, None, None].unsqueeze(-1)

            ls = ls[:env_height, :env_width].reshape([-1, 3])
            self.ls = torch.from_numpy(ls).to(gpu, non_blocking=cfg.pinned)[None, None, None].unsqueeze(-1)
            ndotl = np.clip(np.sum(ls * np.array([[0.0, 0.0, 1.0]], dtype=np.float32), axis=1), 0.0, 1.0)
            self.ndotl = torch.from_numpy(ndotl).to(gpu, non_blocking=cfg.pinned)[None, None, None, None].unsqueeze(-1)

            envWeight = np.sin(El_flat) * np.pi * np.pi / env_width / env_height
            envWeight_ndotl = ndotl[:, None] * envWeight
            envWeight_ndotl = torch.from_numpy(envWeight_ndotl.astype(np.float32))
            self.envWeight_ndotl = envWeight_ndotl[None, None, None, None].to(gpu, non_blocking=cfg.pinned)

        if not is_train:
            for k in self.train_key:
                cfg[k].path = run_id
            self.train_key = []

        all_params = []
        for k in self.net_key:
            cfg_k = getattr(cfg, k)
            setattr(self, k, globals()[k](cfg, cfg_k).to(device))
            if k in self.train_key:
                all_params.append({'params': getattr(self, k).parameters(), 'lr': float(cfg.scheduler.init_lr)})
            else:
                root_k = osp.join(root, cfg_k.path)
                ckpt_path = [osp.join(root_k, f) for f in os.listdir(root_k) if f.endswith('best.pth')]
                if len(ckpt_path) > 1:
                    raise Exception('multiple best ckpt!')
                ckpt_path = ckpt_path[0]
                print(f'read {k} from ', ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
                getattr(self, k).load_state_dict(ckpt[k])
                for param in getattr(self, k).parameters():
                    param.requires_grad = False

        if all_params:
            self.optimizer = torch.optim.Adam(all_params)
        else:
            self.optimizer = None
        self.scheduler_load = None
        self.start_step = self.load_from_ckpt(experiment) if is_train else 0

        for k in self.net_key:
            if k in self.train_key:
                getattr(self, k).train()
                getattr(self, k).is_train = True
            else:
                getattr(self, k).eval()
                getattr(self, k).is_train = False

        if self.is_DDP:
            for k in self.train_key:
                if len([p for p in getattr(self, k).parameters() if p.requires_grad]) > 0:
                    setattr(self, k, torch.nn.parallel.DistributedDataParallel(getattr(self, k), device_ids=[gpu], ))
                else:
                    print(f'{k} module has no learnable parameters. so we dont wrap {k} ddp.')

    def switch_to_eval(self):
        for k in self.net_key:
            getattr(self, k).eval()
            de_parallel(getattr(self, k)).is_train = False

    def switch_to_train(self):
        for k in self.train_key:
            getattr(self, k).train()
            de_parallel(getattr(self, k)).is_train = True

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict()}

        for k in self.train_key:
            to_save[k] = de_parallel(getattr(self, k)).state_dict()
        torch.save(to_save, filename)

    def load_model(self, filename):
        to_load = torch.load(filename, map_location=torch.device('cpu'))
        step = int(filename.split('_')[-1].split('.')[0])

        # print('\033[95m' + 'warning!! optimizer and scheduler is not loaded!!!' + '\033[0m')
        self.optimizer.load_state_dict(to_load['optimizer'])
        self.scheduler_load = to_load['scheduler']
        for k in self.train_key:
            getattr(self, k).load_state_dict(to_load[k])
            print(f'{k} is restored from  {step} steps.')
        return step

    def load_from_ckpt(self, out_folder):
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f) for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]
        # remove best pth. because this function only executed when train
        ckpts = [ckpt for ckpt in ckpts if not ckpt.endswith('best.pth')]

        if ckpts:
            fpath = ckpts[-1]
            step = self.load_model(fpath)
        else:
            print('No ckpts found, training from scratch...')
            step = 0
        return step

    def set_scheduler(self, cfg, total_steps):
        if not hasattr(cfg, 'scheduler') or cfg.scheduler.type == 'None':
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1.0)
            print('not using scheduler')

        elif cfg.scheduler.type == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.scheduler.T_max,
                                                       eta_min=cfg.scheduler.eta_min)
            print(f'use CosineAnnealingLR scheduler T_max: {cfg.scheduler.T_max}, eta_min: {cfg.scheduler.eta_min}')

        elif cfg.scheduler.type == 'StepLR':
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=cfg.scheduler.step_size,
                                            gamma=cfg.scheduler.gamma)
            print(f'use StepLR scheduler step_size: {cfg.scheduler.step_size}, gamma: {cfg.scheduler.gamma}')

        elif cfg.scheduler.type == 'OneCycleLR':
            max_lr = float(cfg.scheduler.max_lr)
            init_lr = float(cfg.scheduler.init_lr)
            final_lr = float(cfg.scheduler.final_lr)
            div_factor = max_lr / init_lr
            final_div_factor = init_lr / final_lr
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr,
                                                            total_steps=total_steps, div_factor=div_factor,
                                                            final_div_factor=final_div_factor)
            print(f'use OneCycleLR scheduler init_lr: {init_lr}, max_lr: {max_lr}, final_lr: {final_lr}')

        else:
            print(cfg.scheduler)
            raise Exception("sadfsdf")
        self.scheduler = scheduler
        if self.scheduler_load is not None:
            self.scheduler.load_state_dict(self.scheduler_load)

    def scheduler_step(self):
        self.scheduler.step()

    def get_last_lr(self, log_lr):
        lr_list = self.scheduler.get_last_lr()
        log_lr[f'train/lr'] = lr_list[0]
        # for k, lr in zip(self.train_key, lr_list):
        #     log_lr[f'train/lr_{k}'] = lr
        return log_lr

    # def vsg_dec_wrap(self, vsg, cam_coord, ls_rdf, bb, env_h, env_w):
    #     if hasattr(self, 'VSGDecoder'):
    #         e, e_depth, e_weight = self.VSGDecoder(vsg, cam_coord, ls_rdf, bb, self.radii)
    #         e = rearrange(e, 'b h w (q p) c -> b c h w q p', q=env_h, p=env_w)
    #         # pred['e_depth'] = rearrange(e_depth, 'b h w (q p) c -> b c h w q p', q=env_h, p=env_w)
    #         # pred['e_weight'] = rearrange(e_weight, 'b h w (q p) c -> b c h w q p', q=env_h, p=env_w)
    #     else:
    #         e = envmapfromVSG(vsg, cam_coord, ls_rdf, self.r_dist, bb[:, None, None, None])
    #     return e


def sg_to_rgb(sgs, dir, include_rgb=False):
    if dir.ndim != sgs.ndim:
        dir = dir.unsqueeze(-2)

    sgs = torch.clamp(1.01 * torch.tanh(sgs), -1, 1)
    axis = sgs[..., :3]
    axis = F.normalize(axis, p=2.0, dim=-1)
    sharp = sgs[..., 3:4]
    sharp = 0.5 * (sharp + 1)
    sharp = torch.tan(torch.pi / 2 * 0.999 * sharp)
    intensity = sgs[..., 4:7]
    intensity = 0.5 * (intensity + 1)
    intensity = torch.tan(torch.pi / 2 * 0.999 * intensity)
    mi = sharp * (torch.sum(axis * -dir, dim=-1, keepdim=True) - 1)
    colors_mid = intensity * torch.exp(mi)
    if include_rgb:
        uniform_rgb = sgs[..., 7:]
        uniform_rgb = 0.5 * (uniform_rgb + 1)
        uniform_rgb = torch.tan(torch.pi / 2 * 0.999 * uniform_rgb)
        colors_mid += uniform_rgb
    return colors_mid

# VSG : b c X Y Z
# cam_coord_rdf : b h w 1 3
# ls_rdf : b h w l 3
# bb : b 3
# r_dist : 1 1 1 1 r
def envmapfromVSG(VSG, cam_coord_rdf, ls_rdf, r_dist, bb, sg_order):
    Bn, Cn, _, _, _ = VSG.shape
    _, h, w, _, _ = cam_coord_rdf.shape
    _, _, _, _, r = r_dist.shape
    bb = bb[:, None, None, None]
    include_rgb = True if Cn == 11 else False

    dir_reciprocal = torch.ones_like(ls_rdf)
    dir_reciprocal[ls_rdf == 0.0] = torch.inf
    dir_reciprocal[ls_rdf != 0.0] = torch.div(1.0, ls_rdf[ls_rdf != 0.0])
    t1 = (-bb[..., 0] - cam_coord_rdf[..., 0]) * dir_reciprocal[..., 0]
    t2 = (bb[..., 0] - cam_coord_rdf[..., 0]) * dir_reciprocal[..., 0]
    t3 = (-bb[..., 1] - cam_coord_rdf[..., 1]) * dir_reciprocal[..., 1]
    t4 = (bb[..., 1] - cam_coord_rdf[..., 1]) * dir_reciprocal[..., 1]
    t5 = (-bb[..., 2] - cam_coord_rdf[..., 2]) * dir_reciprocal[..., 2]
    t6 = (bb[..., 2] - cam_coord_rdf[..., 2]) * dir_reciprocal[..., 2]
    tmax = torch.min(torch.stack([torch.max(t1, t2), torch.max(t3, t4), torch.max(t5, t6)], dim=-1), dim=-1)[
        0].unsqueeze(-1)
    # tmin = torch.max(torch.stack([torch.min(t1, t2), torch.min(t3, t4), torch.min(t5, t6)], dim=-1), dim=-1)[0].unsqueeze(-1)
    ls_cam_pts = (tmax * r_dist).unsqueeze(-1) * ls_rdf.unsqueeze(-2) + cam_coord_rdf.unsqueeze(-2)
    ls_cam_pts = ls_cam_pts.reshape([Bn, h, w, -1, 3])
    ls_cam_pts = ls_cam_pts / bb  # we have to normalize pts
    VSG_ray = g_sam(VSG.permute([0, 1, 4, 2, 3]), ls_cam_pts)
    VSG_ray = rearrange(VSG_ray, 'b c h w (l r) -> b h w l r c', r=r)

    sgs = VSG_ray[..., :-1]
    alpha = VSG_ray[..., -1:]
    alpha = 0.5 * (torch.clamp(1.01 * torch.tanh(alpha), -1, 1) + 1)
    alpha_shifted = torch.cat([torch.ones_like(alpha[..., :1, :]), 1 - alpha + 1e-10], -2)
    weights = alpha * torch.cumprod(alpha_shifted, -2)[..., :-1, :]

    if sg_order == 'before':
        sgs = sg_to_rgb(sgs, ls_rdf, include_rgb)
        composite_rgb = torch.sum(weights * sgs, -2)
    elif sg_order == 'after':
        sgs = torch.sum(weights * sgs, -2)
        composite_rgb = sg_to_rgb(sgs, ls_rdf, include_rgb)
    else:
        raise Exception('check sg order.')
    return composite_rgb


def get_mask_dict(k, mask):
    scalar = (reduce(lambda x, y: x * y, mask.shape[1:]) /
              (torch.sum(mask, dim=list(range(len(mask.shape)))[1:], keepdim=True) + 1e-5))
    return {k: (mask, scalar)}


def compute_projection(pixel_batch, int_list, c2w_list, d_and_f_list, im_list):
    bn, vn, depth_ch, h, w = d_and_f_list.shape
    w2c_list = torch.inverse(c2w_list)
    norm_coord = torch.inverse(int_list[:, None, None, 0]) @ pixel_batch[:, :h, :w]
    cam_coord = d_and_f_list[:, 0, 0, ..., None, None] * norm_coord
    # because cam_0 is world
    world_coord = torch.cat([cam_coord, torch.ones_like(cam_coord[:, :, :, :1, :])], dim=-2)

    # get projection coord
    cam_coord_k = (w2c_list[:, :, None, None] @ world_coord[:, None])[..., :3, :]
    pixel_depth_k_est = torch.clamp(cam_coord_k[..., 2:3, :], min=1e-5)
    norm_coord_k = cam_coord_k / pixel_depth_k_est
    pixel_coord_k_est = (int_list[:, :, None, None] @ norm_coord_k)[..., :2, 0]

    scale_wh = torch.tensor([w, h], dtype=int_list.dtype, device=int_list.device).view(1, 1, 1, 1, 2)
    pixel_coord_k_norm = (2.0 * (pixel_coord_k_est / scale_wh) - 1.).reshape([bn * vn, h, w, 2])
    pixel_rgbd_k = g_sam_nn(torch.cat([im_list, d_and_f_list], dim=2).reshape([bn * vn, 3 + depth_ch, h, w]),
                            pixel_coord_k_norm).reshape([bn, vn, 3 + depth_ch, h, w])

    proj_err = (pixel_depth_k_est[..., 0] - pixel_rgbd_k[:, :, 3, ..., None]).permute(0, 2, 3, 1, 4)
    feat = None
    if depth_ch > 1:
        feat = pixel_rgbd_k[:, :, 4:].permute(0, 3, 4, 1, 2)

    rgb_sampled = pixel_rgbd_k[:, :, :3].permute(0, 3, 4, 1, 2)
    viewdir = F.normalize((c2w_list[:, :, None, None, :3, :3] @ norm_coord_k)[..., 0], dim=-1)
    viewdir[..., :1] = -viewdir[..., :1]  # view dir is surface -> camera, rdf to rub
    return rgb_sampled, viewdir.permute(0, 2, 3, 1, 4), proj_err, feat


def get_visible_surface_volume(voxel_grid, source, intrinsic, use_conf=True, sigma=0.15):
    bn, c, h, w = source.shape
    bn, c1, c2, c3, _ = voxel_grid.shape

    # get projection coord
    pixel_coord = (intrinsic[:, None, None, None] @ voxel_grid[..., None])[..., 0]
    resize_factor = torch.tensor([w, h]).to(pixel_coord.device)[None, None, None, None, :]
    pixel_coord_ = pixel_coord[..., :2] / resize_factor
    pixel_coord_ = pixel_coord_ / pixel_coord[..., 2:3]

    pixel_coord_norm = (2 * pixel_coord_ - 1.).reshape(bn, c1, c2 * c3, 2)
    unprojected_volume = g_sam(source, pixel_coord_norm).reshape(bn, c, c1, c2, c3)
    if use_conf:
        visible_surface_volume = torch.cat([unprojected_volume[:, :3], unprojected_volume[:, 5:]], dim=1)
        volume_weight_k = torch.exp(
            unprojected_volume[:, 4, ...] * -torch.pow(unprojected_volume[:, 3, ...] - voxel_grid[..., -1], 2))
    else:
        visible_surface_volume = torch.cat([unprojected_volume[:, :3], unprojected_volume[:, 4:]], dim=1)
        volume_weight_k = torch.exp(
            -torch.pow(unprojected_volume[:, 3, ...] - voxel_grid[..., -1], 2) / (2 * sigma ** 2))
    return visible_surface_volume * volume_weight_k.unsqueeze(1)


# viewdir, normal_rub, albedo, rough : B H W V C
# ls_rub, envmap : B H W L 3
# ndl, envWeight_ndotl : 1 1 1 1 L 1
def pbr(viewdir, ls_rub, normal_rub, albedo, rough, ndl, envWeight_ndotl, envmap):
    if envmap.ndim == 6:
        raise Exception('env shape error')
        # envmap = rearrange(envmap, 'b c r c h w -> b c r c (h w)')
    ls_rub = ls_rub.unsqueeze(-3)
    albedo = albedo / torch.pi
    rough = rough.unsqueeze(-2)
    normal_rub = normal_rub.unsqueeze(-2)
    viewdir = viewdir.unsqueeze(-2)

    h = viewdir + ls_rub
    h = F.normalize(h, p=2.0, dim=-1)

    vdh = torch.sum((viewdir * h), dim=-1, keepdim=True)
    F0 = 0.05
    frac0 = F0 + (1 - F0) * torch.pow(2.0, (-5.55472 * vdh - 6.98316) * vdh)

    k = (rough + 1) ** 2 / 8.0
    alpha = rough * rough
    alpha2 = (alpha * alpha)

    ndv = torch.clamp(torch.sum(normal_rub * viewdir, dim=-1, keepdim=True), 0, 1)
    ndh = torch.clamp(torch.sum(normal_rub * h, dim=-1, keepdim=True), 0, 1)
    # ndl = torch.clamp(torch.sum(normal_rub * ls_rub, dim=-1, keepdim=True), 0, 1)

    frac = alpha2 * frac0
    nom0 = ndh * ndh * (alpha2 - 1) + 1
    nom1 = ndv * (1 - k) + k
    nom2 = ndl * (1 - k) + k
    nom = torch.clamp(4 * np.pi * nom0 * nom0 * nom1 * nom2, 1e-6, 4 * np.pi)
    specPred = frac / nom

    if envmap.shape[1:3] != albedo.shape[1:3]:
        envmap = rearrange(envmap, 'b h w l c -> b (l c) h w')
        envmap = F.interpolate(envmap, scale_factor=2, mode='bilinear')
        envmap = rearrange(envmap, 'b (l c) h w -> b h w l c', c=3)
    envmap = envmap.unsqueeze(-3)
    # brdfDiffuse = albedo * ndl
    # colorDiffuse = torch.sum(brdfDiffuse * envmap * envWeight, dim=-2)
    # brdfSpec = specPred * ndl
    # colorSpec = torch.sum(brdfSpec * envmap * envWeight, dim=-2)
    shading = torch.sum(envmap * envWeight_ndotl, dim=-2)
    colorDiffuse = albedo * shading
    colorSpec = torch.sum(specPred * envmap * envWeight_ndotl, dim=-2)
    return colorDiffuse, colorSpec, shading
