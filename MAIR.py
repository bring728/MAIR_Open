import torch
from model import *


class MAIRorg(Model):
    def __init__(self, cfg, gpu, experiment, is_train, is_DDP=True):
        self.mode = cfg.mode
        if self.mode == 'incident':
            self.net_key = ['MGNet', 'InLightSG']
            self.train_key = ['InLightSG']

        elif self.mode == 'exitant':
            self.net_key = ['MGNet', 'ExDLVSG']
            self.train_key = ['ExDLVSG']
            r_res = 32
            r_dist = (np.arange(r_res) + 0.5) / r_res
            r_dist = torch.from_numpy(r_dist.astype(np.float32))
            r_dist = r_dist.to(gpu, non_blocking=cfg.pinned)[None, None, None, None]
            self.r_dist = r_dist.to(gpu, non_blocking=cfg.pinned)

        elif self.mode == 'BRDF':
            self.net_key = ['MGNet', 'InLightSG', 'AggregationNet']
            self.train_key = ['AggregationNet']
            if cfg.ContextNet.use:
                self.net_key += ['ContextNet', ]
                self.train_key += ['ContextNet', ]
            if cfg.RefineNet.use:
                self.net_key += ['RefineNet', ]
                self.train_key += ['RefineNet', ]

        elif self.mode == 'VSG':
            r_res = 64
            r_dist = (np.arange(r_res) + 0.5) / r_res
            r_dist = torch.from_numpy(r_dist.astype(np.float32))
            r_dist = r_dist.to(gpu, non_blocking=cfg.pinned)[None, None, None, None]
            self.r_dist = r_dist.to(gpu, non_blocking=cfg.pinned)

            self.net_key = ['MGNet', 'InLightSG', 'ContextNet', 'AggregationNet', 'RefineNet',
                            'VSGEncoder']
            self.train_key = ['VSGEncoder']
            if cfg.VSGEncoder.src_type == 'exi':
                self.net_key += ['ExDLVSG', ]
            elif cfg.VSGEncoder.src_type == 'train':
                self.net_key += ['ExDLVSG', ]
                self.train_key += ['ExDLVSG', ]
            # if cfg.VSGDecoder.use:
            #     self.net_key += ['VSGDecoder', ]
            #     self.train_key += ['VSGDecoder', ]
        super().__init__(cfg, gpu, experiment, is_train, is_DDP)

    def forward(self, data, cfg, forward_mode='train'):
        with autocast(enabled=cfg.autocast):
            empty = torch.tensor([], device=data['i'].device)
            pred = {}
            gt = {k: data[k] for k in set(cfg.losskey) & set(data)}
            mask = data.pop('m')
            if 'e' in gt or 'e_d' in gt:
                Bn, env_rows, env_cols, _, _ = gt.get('e', gt.get('e_d')).shape
                mask = F.adaptive_avg_pool2d(mask, (env_rows, env_cols))
                if 'e' in gt:
                    mask = mask * (torch.mean(gt['e'], dim=(3, 4)) > 0.001).float()[:, None]
                else:
                    mask = (mask > 0.9).float()

            assert data['i'].shape[1:] == (3, 240, 320)
            n, d, _, _ = self.MGNet(data['i'], data['cds_dn'], data['cds_conf'], data['cds_dg'])

            if cfg.d_type == 'net':
                d = d / torch.amax(d, dim=[1, 2, 3], keepdim=True)
                c = empty
            elif cfg.d_type == 'cds':
                d = data['cds_dn']
                c = data['cds_conf']
            ### end mode incident
            if cfg.mode == 'incident':
                axis, sharp, intensity, pred['vis'], pred['e_d'] = self.InLightSG(data['i'], d, c, n, empty, empty,
                                                                                  empty, mode=0)
                pred['vis'] = pred['vis'][:, :, 0, :, :, None, None]
                return pred, gt, get_mask_dict('default', mask)

            if cfg.mode == 'exitant' or (cfg.mode == 'VSG' and (cfg.VSGEncoder.src_type == 'exi' or
                                                                cfg.VSGEncoder.src_type == 'train')):
                VSG_DL = self.ExDLVSG(data['i'], d, c, n)
                if cfg.mode == 'exitant':
                    cam_mat = data['cam'] / (cfg.imWidth / env_cols)
                    cam_mat[:, -1, -1] = 1.0
                    pixels_DL = self.pixels[:, :env_rows, :env_cols]
                    depth_low = F.adaptive_avg_pool2d(d, (env_rows, env_cols)).permute([0, 2, 3, 1])
                    if cfg.d_type == 'cds':
                        conf_low = F.adaptive_avg_pool2d(c, (env_rows, env_cols))
                        mask = mask * conf_low

                    cam_coord = (depth_low * (torch.inverse(cam_mat[:, None, None]) @ pixels_DL)[..., 0])[..., None, :]
                    normal_low = F.adaptive_avg_pool2d(n, (env_rows, env_cols)).permute([0, 2, 3, 1])
                    normal_low = F.normalize(normal_low, p=2.0, dim=-1)
                    N2C = get_N2C(normal_low, self.up)
                    ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
                    vsg_alpha = 0.5 * (torch.clamp(1.01 * torch.tanh(VSG_DL[:, -1]), -1, 1) + 1)
                    pred['vis'] = vsg_alpha
                    pred['e_d'] = envmapfromVSG(VSG_DL, cam_coord, ls_rdf, self.r_dist, data['bb'], cfg.sg_order)
                    return pred, gt, get_mask_dict('env', mask[:, 0, ..., None, None])

            axis, sharp, intensity, vis = self.InLightSG(data['i'], d, c, n, empty, empty, empty, mode=1)
            featmaps = self.ContextNet(data['i'], d, c, n, empty, empty) if cfg.ContextNet.use else empty
            all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], data['all_i'])

            bn, h_, w_, v_, _ = all_rgb.shape
            n_low = F.adaptive_avg_pool2d(n, (axis.shape[-2], axis.shape[-1])).permute([0, 2, 3, 1])
            N2C = get_N2C(F.normalize(n_low, p=2.0, dim=-1), self.up)
            axis_cam = torch.einsum('bhwqp,bsphw->bsqhw', N2C, axis)
            DL_flat = F.interpolate(
                torch.cat([axis_cam, sharp, intensity], dim=2).reshape(bn, -1, axis.shape[-2], axis.shape[-1]),
                scale_factor=4, mode='nearest')
            DL = rearrange(DL_flat, 'b (q p) h w  -> b h w () q p', q=cfg.InLightSG.SGNum)

            n_low = F.adaptive_avg_pool2d(n, (h_, w_)).permute([0, 2, 3, 1])
            n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
            brdf_feat = self.AggregationNet(all_rgb, None, None, None, proj_err, featmaps, viewdir, n_low, DL)
            if cfg.RefineNet.use:
                a, r, _ = self.RefineNet(data['i'], d, c, n, brdf_feat)
            else:
                brdf_feat = F.interpolate(brdf_feat, scale_factor=2.0, mode='bilinear')
                a = 0.5 * (torch.clamp(1.01 * torch.tanh(brdf_feat[:, :3]), -1, 1) + 1)
                r = 0.5 * (torch.clamp(1.01 * torch.tanh(brdf_feat[:, 3:]), -1, 1) + 1)
            ### end mode BRDF
            if cfg.mode == 'BRDF':
                pred['a'], pred['r'] = a, r
                return pred, gt, get_mask_dict('default', mask)

            assert a.shape[0] == 1
            if forward_mode == 'output':
                env_rows, env_cols, env_w, env_h = self.envRows, self.envCols, self.env_width, self.env_height

            source = torch.cat([data['i'], d, c, n, a, r], dim=1)
            if cfg.VSGEncoder.src_type == 'exi' or cfg.VSGEncoder.src_type == 'train':
                VSG_DL = torch.clamp(1.01 * torch.tanh(VSG_DL), -1, 1)
                vsg_tmp1 = 0.5 * (F.normalize(VSG_DL[:, :3], p=2.0, dim=1) + 1)
                vsg_tmp2 = 0.5 * (VSG_DL[:, 3:] + 1)
                VSG_DL = torch.cat([vsg_tmp1, vsg_tmp2], dim=1)
                del vsg_tmp1, vsg_tmp2

            elif cfg.VSGEncoder.src_type == 'inci':
                DL_flatten = F.interpolate(DL_flat, scale_factor=2, mode='nearest')
                source = torch.cat([source, DL_flatten], dim=1)
                VSG_DL = None
            elif cfg.VSGEncoder.src_type == 'none':
                VSG_DL = torch.zeros([1, 8, 32, 32, 32], dtype=a.dtype, device=a.device)
            vsg_in = get_visible_surface_volume(data['voxel_grid_front'], source, data['cam'])
            vsg = self.VSGEncoder(vsg_in, VSG_DL)  # normal
            vsg_alpha = 0.5 * (torch.clamp(1.01 * torch.tanh(vsg[:, -1]), -1, 1) + 1)
            pred['vis'] = vsg_alpha

            a_low = F.adaptive_avg_pool2d(a, (env_rows, env_cols))
            r_low = F.adaptive_avg_pool2d(r, (env_rows, env_cols))
            d_low = F.adaptive_avg_pool2d(d, (env_rows, env_cols))
            N2C = get_N2C(n_low.squeeze(-2), self.up)
            cam_mat = data['cam'] / (cfg.imWidth / env_cols)
            cam_mat[:, -1, -1] = 1.0
            pixels = self.pixels[:, :env_rows, :env_cols]
            cam_coord = (d_low[:, 0, ..., None, None] * torch.inverse(cam_mat[:, None, None]) @ pixels)[..., None, :, 0]
            ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
            if forward_mode == 'train':
                assert Bn == 1
                nonzero_idxs = torch.nonzero(mask[0, 0, :, :])
                idxs = nonzero_idxs[torch.randperm(nonzero_idxs.size(0))[:cfg.num_of_samples]]
                gt['e'] = gt['e'][:, :, idxs[:, 0], idxs[:, 1], :, :][:, :, None]
                ls_rdf = ls_rdf[:, idxs[:, 0], idxs[:, 1]][:, None]
                cam_coord = cam_coord[:, idxs[:, 0], idxs[:, 1]]
                cam_coord = cam_coord.repeat(1, 1, ls_rdf.shape[-2], 1)[:, None]
                mask = torch.ones([1, 1, 1, cam_coord.shape[2]], device=cam_coord.device, dtype=cam_coord.dtype)
                if cam_coord.shape[2] < cfg.num_of_samples:
                    pad = torch.zeros([1, 1, cfg.num_of_samples - cam_coord.shape[2], ls_rdf.shape[-2], 3],
                                      device=cam_coord.device, dtype=cam_coord.dtype)
                    cam_coord = torch.cat([cam_coord, pad], dim=2)
                    ls_rdf = torch.cat([ls_rdf, pad + 0.1], dim=2)

                    pad = torch.zeros([1, 1, 1, cfg.num_of_samples - mask.shape[3]],
                                      device=cam_coord.device, dtype=cam_coord.dtype)
                    mask = torch.cat([mask, pad], dim=3)

                pred['e'] = envmapfromVSG(vsg, cam_coord, ls_rdf, self.r_dist, data['bb'], cfg.sg_order)
                return pred, gt, get_mask_dict('env', mask[:, 0, ..., None, None])
            elif forward_mode == 'test' or forward_mode == 'output':
                chunk = 10000
                bn, h, w, l, _ = ls_rdf.shape
                ls_rdf = ls_rdf.reshape([bn, 1, h * w, l, 3])
                cam_coord = cam_coord.reshape([bn, 1, h * w, 1, 3]).repeat(1, 1, 1, l, 1)
                x_list = []
                for j in range(0, h * w, chunk):
                    cam_coord_j = cam_coord[:, :, j:j + chunk]
                    ls_rdf_j = ls_rdf[:, :, j:j + chunk]
                    xj = envmapfromVSG(vsg, cam_coord_j, ls_rdf_j, self.r_dist, data['bb'], cfg.sg_order)
                    x_list.append(xj)

                pred_env_vsg = torch.cat(x_list, dim=2).reshape([bn, h, w, l, 3])
                env_constant_scale = 1000.0
                ls_rub = (ls_rdf * self.rub_to_rdf).reshape([bn, h, w, l, 3])
                diffuse, specular, _ = pbr(viewdir, ls_rub, n_low, a_low.permute([0, 2, 3, 1]).unsqueeze(-2),
                                           r_low.permute([0, 2, 3, 1]).unsqueeze(-2), self.ndotl, self.envWeight_ndotl,
                                           pred_env_vsg * env_constant_scale)

                diffuse = self.re_arr(diffuse)
                specular = self.re_arr(specular)
                diffscaled, specscaled, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                                 diffuse, specular, mask, scale_type=1)

                if forward_mode == 'test':
                    pred['e'] = pred_env_vsg
                    pred['rgb'] = torch.clamp(diffscaled + specscaled, 0, 1.0)
                    gt['rgb'] = self.re_arr(all_rgb)
                    mask_dict = get_mask_dict('default', mask)
                    mask_dict.update(get_mask_dict('env', mask[:, 0, ..., None, None]))
                    return pred, gt, mask_dict

                if forward_mode == 'output':
                    cDiff = (torch.sum(diffscaled) / torch.sum(diffuse)).data.item()
                    cSpec = (torch.sum(specscaled)) / (torch.sum(specular)).data.item()
                    if cSpec < 1e-3:
                        cAlbedo = 1 / a.max()
                        cLight = cDiff / cAlbedo
                    else:
                        cLight = cSpec
                        cAlbedo = cDiff / cLight
                        cAlbedo = torch.clamp(cAlbedo, 1e-3, 1 / a.max())
                        cLight = cDiff / cAlbedo

                    pred['vsg'] = (vsg, cLight.item())
                    pred['n'], pred['d'], pred['a'], pred['r'] = n, d, a * cAlbedo, r
                    pred['diff_vsg'], pred['spec_vsg'] = diffscaled, specscaled
                    env = pred_env_vsg * cLight * env_constant_scale
                    pred['e_vsg'] = rearrange(env, 'b r c (h w) C -> b C r c h w', h=env_h, w=env_w)
                    pred['rgb_vsg'] = diffscaled + specscaled
                    return pred, gt
