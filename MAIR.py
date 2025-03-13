from model import *
from torch.cuda.amp.autocast_mode import autocast


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


class MAIRplusplus(Model):
    def __init__(self, cfg, gpu, experiment, is_train, is_DDP=True):
        self.mode = cfg.mode
        self.im_down = lambda x: F.adaptive_avg_pool2d(x, (cfg.envRows, cfg.envCols))
        if self.mode == 'MG':
            self.net_key = ['MGNet', ]
            self.train_key = ['MGNet', ]
        elif self.mode == 'mat_edit':
            self.net_key = ['ShadingNet', 'SpecularNet']
            self.train_key = []
        else:
            if hasattr(cfg, 'InLightSG'):
                inlight_key = ['InLightSG']
            else:
                inlight_key = ['InLightEncoder', 'InLightDecoder', 'ShadingNet', 'SpecularNet']

            if self.mode == 'incident':
                self.net_key = ['MGNet', ] + inlight_key
                self.train_key = inlight_key
            else:
                if 'InLightDecoder' in inlight_key:
                    inlight_key.remove('InLightDecoder')
                if self.mode == 'BRDF':
                    self.net_key = ['MGNet', 'ContextNet', 'AggregationNet', 'RefineNet'] + inlight_key
                    self.train_key = ['ContextNet', 'AggregationNet', 'RefineNet']

                elif self.mode == 'AlbedoFusion':
                    self.net_key = ['MGNet', 'ContextNet', 'AggregationNet', 'RefineNet', 'AlbedoFusion'] + inlight_key
                    self.train_key = ['AlbedoFusion', ]

                elif self.mode == 'VSG':
                    self.net_key = ['MGNet', 'ContextNet', 'AggregationNet', 'RefineNet', 'AlbedoFusion',
                                    'VSGEncoder', 'InLightDecoder'] + inlight_key
                    self.train_key = ['VSGEncoder']

                    if cfg.VSGEncoder.src_type == 'exi':
                        self.net_key += ['ExDLVSG', ]
                        self.train_key += ['ExDLVSG', ]
                    r_res = 64
                    r_dist = (np.arange(r_res) + 0.5) / r_res
                    r_dist = torch.from_numpy(r_dist.astype(np.float32))
                    r_dist = r_dist.to(gpu, non_blocking=cfg.pinned)[None, None, None, None]
                    self.r_dist = r_dist.to(gpu, non_blocking=cfg.pinned)

        super().__init__(cfg, gpu, experiment, is_train, is_DDP)

    def forward(self, data, cfg, forward_mode='train'):
        with autocast(enabled=cfg.autocast):
            pred = {}
            gt = {k: data[k] for k in set(cfg.losskey) & set(data)}
            mask = data.pop('m')
            if 'e' in gt:
                mask = F.adaptive_avg_pool2d(mask, gt['e'].shape[1:3])
                mask = mask * (torch.mean(gt['e'], dim=(3, 4)) > 0.001).float()[:, None]

            assert data['i'].shape[1:] == (3, 240, 320)

            n, d, a, r = self.MGNet(data['i'], data['cds_dn'], data['cds_conf'], data['cds_dg'])
            if cfg.mode == 'MG' or forward_mode == 'custom_MG':
                pred['n'], pred['d'], pred['a'], pred['r'] = n, d, a, r
                return pred, gt, get_mask_dict('default', mask)

            d = d / torch.amax(d, dim=[1, 2, 3], keepdim=True)
            norm_coord = (torch.inverse(data['cam'][:, None, None]) @ self.pixels).squeeze(-1)
            viewdir = F.normalize(norm_coord, dim=-1, p=2.0)
            viewdir[..., :1] = -viewdir[..., :1]  # view dir is surface -> camera, rdf to rub
            viewdir = viewdir.permute([0, 3, 1, 2])
            NdotV = torch.sum(viewdir * n, dim=1, keepdim=True)
            del viewdir
            all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], data['all_i'])

            use_gt = True
            if hasattr(cfg, 'use_gt'):
                use_gt = cfg.use_gt
            if cfg.mode == 'incident' and use_gt:
                # in incident mode, we use GT.
                n_low = data['n'].permute([0, 2, 3, 1]).unsqueeze(-2)
                a_low = data['a'].permute([0, 2, 3, 1]).unsqueeze(-2)
                r_low = data['r'].permute([0, 2, 3, 1]).unsqueeze(-2)
            else:
                n_low = self.im_down(n).permute([0, 2, 3, 1])
                n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
                a_low = self.im_down(a).permute([0, 2, 3, 1]).unsqueeze(-2)
                r_low = self.im_down(r).permute([0, 2, 3, 1]).unsqueeze(-2)
            N2C = get_N2C(n_low, self.up).squeeze(-3)
            if hasattr(self, 'InLightSG'):
                axis, sharp, intensity, vis, pred['e'] = self.InLightSG(data['i'], d, self.empty, n, a, r, NdotV,
                                                                        mode=0)
            else:
                light_feat = self.InLightEncoder(data['i'], d, n, a, r, NdotV)
                light_feat = light_feat.permute([0, 2, 3, 1])
                if forward_mode == 'output':
                    shading = self.ShadingNet(light_feat).unsqueeze(-2)
                    diffuse = a_low * shading
                    C2N = torch.linalg.pinv(N2C)
                    ref_dir = 2 * torch.sum(viewdir * n_low, dim=-1, keepdim=True) * n_low - viewdir
                    # ref_dir in normal coord
                    ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
                    n_dot_v = torch.sum(n_low * viewdir, dim=-1, keepdim=True)
                    specular, spec_feat, spec_in = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
                    diffuse = self.re_arr(diffuse)
                    specular = self.re_arr(specular)
                    diffscaled, specscaled, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                                     diffuse, specular, mask, scale_type=1)

                    cDiff = (torch.sum(diffscaled) / torch.sum(diffuse)).data.item()
                    cSpec = (torch.sum(specscaled)) / (torch.sum(specular)).data.item()
                    if cSpec < 1e-3:
                        cAlbedo = 1 / a.max().data.item()
                    else:
                        cLight = cSpec
                        cAlbedo = cDiff / cLight
                        cAlbedo = torch.clamp(cAlbedo, 1e-3, 1 / a.max().data.item())
                    pred['a_s'] = a * cAlbedo
                    pred['r_s'] = r
                    pred['ilr'] = light_feat

            ### end mode incident
            if cfg.mode == 'incident':
                ls_rub = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1)
                if hasattr(self, 'InLightDecoder'):
                    shading = self.ShadingNet(light_feat).unsqueeze(-2)
                    diffuse = a_low * shading
                    # when normal == (0,1,0), N2C is singular matrix..
                    # so we should use pseudo-inverse matrix.
                    # C2N = torch.linalg.inv(N2C)
                    C2N = torch.linalg.pinv(N2C)
                    ref_dir = 2 * torch.sum(viewdir * n_low, dim=-1, keepdim=True) * n_low - viewdir
                    # ref_dir in normal coord
                    ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
                    n_dot_v = torch.sum(n_low * viewdir, dim=-1, keepdim=True)
                    specular, spec_feat, spec_in = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
                    pred['e'], pred['vis'] = self.InLightDecoder(light_feat)

                    # tmp_shading = self.re_arr(shading)
                    # gt['shading'] = torch.sum(gt['e'] * self.envWeight_ndotl[0], dim=-2).permute([0, 3, 1, 2])
                    # pred['shading_record'], _ = LSregress(tmp_shading, gt['shading'], tmp_shading, mask)

                    # random material loss..
                    # a_random = torch.rand_like(a_low)
                    # r_random = torch.rand_like(r_low)
                    # pred['diffuse_random'] = self.re_arr(a_random * shading)
                    # spec_ran, _, _ = self.SpecularNet(None, None, None, spec_feat[..., :1, :], r_random)
                    # pred['specular_random'] = self.re_arr(spec_ran)
                    # diff_ran_gt, spec_ran_gt, _ = pbr(viewdir[..., :1, :], ls_rub, n_low, a_random, r_random,
                    #                                   self.ndotl, self.envWeight_ndotl, gt['e'])
                    # gt['diffuse_random'], gt['specular_random'] = self.re_arr(diff_ran_gt), self.re_arr(spec_ran_gt)

                else:
                    diffuse, specular, shading = pbr(viewdir, ls_rub, n_low, a_low, r_low, self.ndotl,
                                                     self.envWeight_ndotl, pred['e'])

                diff_scaled, spec_scaled, mask_all_view = LSDiffSpec_multiview(diffuse.detach(), specular.detach(),
                                                                               all_rgb, diffuse, specular, mask,
                                                                               proj_err, scale_type=1)
                # diff_scaled, spec_scaled, mask_all_view = LSDiffSpec_multiview_paper(diffuse, specular, all_rgb, mask,
                #                                                                proj_err, scale_type=1)
                rgb_pred = torch.clamp(diff_scaled + spec_scaled, 0, 1.0)
                # rgb_error = (((rgb_pred - all_rgb) * mask_all_view).abs().sum(dim=-2) /
                #              (mask_all_view.sum(dim=-2) + 1e-5)).permute([0, 3, 1, 2])
                gt['diffuse'], gt['specular'], gt['shading'] = pbr(viewdir, ls_rub, n_low, a_low, r_low, self.ndotl,
                                                                   self.envWeight_ndotl, gt['e'])
                gt['rgb'] = all_rgb
                pred['rgb'], pred['specular'] = rgb_pred, spec_scaled
                # pred['diffuse'], gt['diffuse'] = self.re_arr(diff_scaled), self.re_arr(gt['diffuse'])
                pred['shading'], gt['shading'] = self.re_arr(shading), self.re_arr(gt['shading'])
                mask_dict = get_mask_dict('default', mask)
                mask_dict.update(get_mask_dict('env', mask[:, 0, ..., None, None]))
                mask_dict.update(get_mask_dict('all_view', mask_all_view))
                return pred, gt, mask_dict

            DL_target, shading, spec_feat, spec_in, diffuse, specular = None, None, None, None, None, None
            if hasattr(self, 'InLightSG'):
                axis_cam = torch.einsum('bhwqp,bsphw->bsqhw', N2C, axis)
                DL_target = rearrange(torch.cat([axis_cam, sharp, intensity], dim=2),
                                      'b s c h w-> b h w () s c')
            else:
                shading = self.ShadingNet(light_feat).unsqueeze(-2)
                diffuse = self.re_arr(a_low * shading)
                C2N = torch.linalg.pinv(N2C)
                ref_dir = 2 * torch.sum(viewdir * n_low, dim=-1, keepdim=True) * n_low - viewdir
                # ref_dir in normal coord
                ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
                n_dot_v = torch.sum(n_low * viewdir, dim=-1, keepdim=True)
                specular, spec_feat, spec_in = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
                specular = self.re_arr(specular)
                diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                            diffuse, specular, mask, scale_type=1)

            featmaps = self.ContextNet(data['i'], d, self.empty, n, self.empty,
                                       self.empty) if cfg.ContextNet.use else self.empty
            brdf_feat = self.AggregationNet(all_rgb, shading, spec_feat, spec_in, proj_err, featmaps, viewdir,
                                            n_low, DL_target)
            albedo, rough, refine_feat = self.RefineNet(data['i'], d, self.empty, n, brdf_feat)
            pred['a_m'] = albedo
            pred['r_m'] = rough
            ### end mode BRDF
            if cfg.mode == 'BRDF':
                # test mode only. To predict RGB error.
                if forward_mode == 'test':
                    a_low = self.im_down(albedo).permute([0, 2, 3, 1]).unsqueeze(-2)
                    r_low = self.im_down(rough).permute([0, 2, 3, 1]).unsqueeze(-2)
                    # a_gt_low = self.im_down(data['a']).permute([0, 2, 3, 1]).unsqueeze(-2)
                    # r_gt_low = self.im_down(data['r']).permute([0, 2, 3, 1]).unsqueeze(-2)
                    if hasattr(self, 'InLightSG'):
                        ls_rub = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1)
                        diffuse, specular, _ = pbr(viewdir, ls_rub, n_low, a_low, r_low, self.ndotl,
                                                   self.envWeight_ndotl, pred['e'])
                        # diffuse2, specular2, _ = pbr(viewdir, ls_rub, n_low, a_gt_low, r_gt_low, self.ndotl,
                        #                              self.envWeight_ndotl, pred['e'])
                    else:
                        diffuse = a_low * shading
                        specular, _, _ = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
                        # diffuse2 = a_gt_low * shading
                        # specular2, _, _ = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_gt_low)

                    specular = self.re_arr(specular)
                    diffuse = self.re_arr(diffuse)
                    diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                                diffuse, specular, mask, scale_type=1)

                    pred['rgb'] = F.interpolate(torch.clamp(diffuse + specular, 0, 1.0), scale_factor=2.0,
                                                mode='bilinear')
                    gt['rgb'] = F.interpolate(self.re_arr(all_rgb), scale_factor=2.0, mode='bilinear')
                    # specular2 = self.re_arr(specular2)
                    # diffuse2 = self.re_arr(diffuse2)
                    # diffuse2, specular2, _, _ = LSregressDiffSpec(diffuse2, specular2, self.re_arr(all_rgb),
                    #                                               diffuse2, specular2, mask, scale_type=1)
                    # pred['rgb2'] = F.interpolate(torch.clamp(diffuse2 + specular2, 0, 1.0), scale_factor=2.0,
                    #                              mode='bilinear')
                    # gt['rgb2'] = F.interpolate(self.re_arr(all_rgb), scale_factor=2.0, mode='bilinear')

                pred['a'], pred['r'] = albedo, rough
                return pred, gt, get_mask_dict('default', mask)

            albedo = self.AlbedoFusion(data['i'], refine_feat, a, r, diffuse, specular)
            if cfg.mode == 'AlbedoFusion':
                if hasattr(cfg.AlbedoFusion, 'is_rough') and cfg.AlbedoFusion.is_rough:
                    pred['r'] = albedo
                else:
                    pred['a'] = albedo
                return pred, gt, get_mask_dict('default', mask)

            assert albedo.shape[0] == 1
            env_rows, env_cols = light_feat.shape[1:3]
            a, r = albedo, rough
            a_low = self.im_down(a)
            r_low = rearrange(self.im_down(r), 'b c h w-> b h w () c')
            d_low = rearrange(self.im_down(d), 'b c h w-> b h w () c')
            cam_mat = data['cam'] / (cfg.imWidth / env_cols)
            cam_mat[:, -1, -1] = 1.0

            specular_ilr, _, _ = self.SpecularNet(light_feat, n_dot_v[..., :1, :], ref_dir[..., :1, :], None, r_low)
            light_feat = light_feat.permute([0, 3, 1, 2])
            shading = self.re_arr(shading)
            specular_ilr = self.re_arr(specular_ilr)
            source = torch.cat([data['all_i'][:, 0], self.re_arr(d_low), self.re_arr(n_low), a_low,
                                self.re_arr(r_low), shading, specular_ilr], dim=1)
            if hasattr(self, 'ExDLVSG'):
                light_vol = self.ExDLVSG(light_feat, self.re_arr(d_low), self.empty, self.re_arr(n_low))
            else:
                source = torch.cat([source, light_feat], dim=1)
                light_vol = None

            if cfg.VSGEncoder.vsg_type == 'voxel':
                vsg_in = get_visible_surface_volume(data['voxel_grid_front'], source, cam_mat, False)
            else:
                vsg_in = source
            vsg = self.VSGEncoder(vsg_in, light_vol)  # normal
            vsg_alpha = 0.5 * (torch.clamp(1.01 * torch.tanh(vsg[:, -1]), -1, 1) + 1)
            pred['vis'] = vsg_alpha

            pixels = self.pixels[:, :env_rows, :env_cols]
            cam_coord = (d_low * torch.inverse(cam_mat[:, None, None]) @ pixels)[..., None, :, 0]
            ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
            if forward_mode == 'train':
                nonzero_idxs = torch.nonzero(mask[0, 0, :, :])
                idxs = nonzero_idxs[torch.randperm(nonzero_idxs.size(0))[:cfg.num_of_samples]]
                gt['e'] = gt['e'][:, idxs[:, 0], idxs[:, 1]][:, None]
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
                                           r_low, self.ndotl, self.envWeight_ndotl, pred_env_vsg * env_constant_scale)

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

                elif forward_mode == 'output':
                    env_w, env_h = self.env_width, self.env_height

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

                    pred['vsg'] = (vsg, cLight.item() * env_constant_scale)
                    pred['n'], pred['d'], pred['r'] = n, d, r
                    pred['diff_vsg'], pred['spec_vsg'] = diffscaled, specscaled
                    env = pred_env_vsg * cLight * env_constant_scale
                    pred['e_vsg'] = rearrange(env, 'b r c (h w) C -> b C r c h w', h=env_h, w=env_w)
                    pred['rgb_vsg'] = diffscaled + specscaled

                    diffuse_ilr = a_low * shading
                    with autocast(enabled=False):
                        pred_env_ilr, _ = self.InLightDecoder(light_feat.permute([0, 2, 3, 1]).float())
                    diffscaled, specscaled, _, _ = LSregressDiffSpec(diffuse_ilr, specular_ilr, self.re_arr(all_rgb),
                                                                     diffuse_ilr, specular_ilr, mask, scale_type=1)

                    cDiff = (torch.sum(diffscaled) / torch.sum(diffuse_ilr)).data.item()
                    cSpec = (torch.sum(specscaled)) / (torch.sum(specular_ilr)).data.item()
                    if cSpec < 1e-3:
                        cAlbedo = 1 / a.max().data.item()
                        cLight = cDiff / cAlbedo
                    else:
                        cLight = cSpec
                        cAlbedo = cDiff / cLight
                        cAlbedo = torch.clamp(cAlbedo, 1e-3, 1 / a.max().data.item())
                        cLight = cDiff / cAlbedo

                    pred['diff_ilr'], pred['spec_ilr'], pred['a'] = diffscaled, specscaled, a * cAlbedo
                    env = pred_env_ilr * 100 * cLight
                    pred['e_ilr'] = rearrange(env, 'b r c (h w) C -> b C r c h w', h=env_h, w=env_w)
                    pred['rgb_ilr'] = diffscaled + specscaled
                    pred['shad_ilr'] = shading
                    return pred, gt

    @autocast()
    def re_rendering(self, data, mode):
        mask = data['m']
        mask = (F.adaptive_avg_pool2d(mask, (120, 160)) > 0.9).float()

        if mode == 'pred_mg_pred_l':
            n, d, a, r = self.MGNet(data['i'], data['cds_dn'], data['cds_conf'], data['cds_dg'])
            d = d / torch.amax(d, dim=[1, 2, 3], keepdim=True)
            norm_coord = (torch.inverse(data['cam'][:, None, None]) @ self.pixels).squeeze(-1)
            viewdir = F.normalize(norm_coord, dim=-1, p=2.0)
            viewdir[..., :1] = -viewdir[..., :1]  # view dir is surface -> camera, rdf to rub
            viewdir = viewdir.permute([0, 3, 1, 2])
            NdotV = torch.sum(viewdir * n, dim=1, keepdim=True)
            del viewdir
            all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], data['all_i'])
            n_low = self.im_down(n).permute([0, 2, 3, 1])
            n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
            a_low = self.im_down(a).permute([0, 2, 3, 1]).unsqueeze(-2)
            r_low = self.im_down(r).permute([0, 2, 3, 1]).unsqueeze(-2)
            N2C = get_N2C(n_low, self.up).squeeze(-3)
            light_feat = self.InLightEncoder(data['i'], d, n, a, r, NdotV)
            light_feat = light_feat.permute([0, 2, 3, 1])

            shading = self.ShadingNet(light_feat).unsqueeze(-2)
            diffuse = self.re_arr(a_low * shading)
            C2N = torch.linalg.pinv(N2C)
            ref_dir = 2 * torch.sum(viewdir * n_low, dim=-1, keepdim=True) * n_low - viewdir
            # ref_dir in normal coord
            ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
            n_dot_v = torch.sum(n_low * viewdir, dim=-1, keepdim=True)
            specular, spec_feat, spec_in = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
            specular = self.re_arr(specular)
            diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                        diffuse, specular, mask, scale_type=1)

            featmaps = self.ContextNet(data['i'], d, self.empty, n, self.empty, self.empty)
            brdf_feat = self.AggregationNet(all_rgb, shading, spec_feat, spec_in, proj_err, featmaps, viewdir,
                                            n_low, None)
            albedo, rough, refine_feat = self.RefineNet(data['i'], d, self.empty, n, brdf_feat)
            ### end mode BRDF
            albedo = self.AlbedoFusion(data['i'], refine_feat, a, r, diffuse, specular)
            a_low = self.im_down(albedo).permute([0, 2, 3, 1]).unsqueeze(-2)
            diffuse = a_low * shading
            r_low = self.im_down(rough).permute([0, 2, 3, 1]).unsqueeze(-2)
            specular, _, _ = self.SpecularNet(light_feat, n_dot_v[..., :1, :], ref_dir[..., :1, :], None, r_low)
            diffuse = self.re_arr(diffuse)
            specular = self.re_arr(specular)
            diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                        diffuse, specular, mask, scale_type=1)
            diffuse = torch.clamp(diffuse, 0, 1.0)
            specular = torch.clamp(specular, 0, 1.0)
            pred_rgb = torch.clamp(diffuse + specular, 0, 1.0)
            return diffuse, specular, pred_rgb, self.re_arr(all_rgb), get_mask_dict('default', mask)

        elif mode == 'gt_mg_pred_l':
            n, d, a, r = self.MGNet(data['i'], data['cds_dn'], data['cds_conf'], data['cds_dg'])
            d = d / torch.amax(d, dim=[1, 2, 3], keepdim=True)
            norm_coord = (torch.inverse(data['cam'][:, None, None]) @ self.pixels).squeeze(-1)
            viewdir = F.normalize(norm_coord, dim=-1, p=2.0)
            viewdir[..., :1] = -viewdir[..., :1]  # view dir is surface -> camera, rdf to rub
            viewdir = viewdir.permute([0, 3, 1, 2])
            NdotV = torch.sum(viewdir * n, dim=1, keepdim=True)
            del viewdir
            all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], data['all_i'])

            light_feat = self.InLightEncoder(data['i'], d, n, a, r, NdotV)

            n_low = self.im_down(data['n']).permute([0, 2, 3, 1])
            n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
            a_low = self.im_down(data['a']).permute([0, 2, 3, 1]).unsqueeze(-2)
            r_low = self.im_down(data['r']).permute([0, 2, 3, 1]).unsqueeze(-2)

            N2C = get_N2C(n_low, self.up).squeeze(-3)
            light_feat = light_feat.permute([0, 2, 3, 1])
            shading = self.ShadingNet(light_feat).unsqueeze(-2)
            diffuse = self.re_arr(a_low * shading)
            C2N = torch.linalg.pinv(N2C)
            ref_dir = 2 * torch.sum(viewdir * n_low, dim=-1, keepdim=True) * n_low - viewdir
            # ref_dir in normal coord
            ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
            n_dot_v = torch.sum(n_low * viewdir, dim=-1, keepdim=True)
            specular, spec_feat, spec_in = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
            specular = self.re_arr(specular)
            diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                        diffuse, specular, mask, scale_type=1)

            diffuse = torch.clamp(diffuse, 0, 1.0)
            specular = torch.clamp(specular, 0, 1.0)
            pred_rgb = torch.clamp(diffuse + specular, 0, 1.0)
            return diffuse, specular, pred_rgb, self.re_arr(all_rgb), get_mask_dict('default', mask)

        elif mode == 'pred_mg_gt_l':
            n, d, a, r = self.MGNet(data['i'], data['cds_dn'], data['cds_conf'], data['cds_dg'])
            d = d / torch.amax(d, dim=[1, 2, 3], keepdim=True)
            norm_coord = (torch.inverse(data['cam'][:, None, None]) @ self.pixels).squeeze(-1)
            viewdir = F.normalize(norm_coord, dim=-1, p=2.0)
            viewdir[..., :1] = -viewdir[..., :1]  # view dir is surface -> camera, rdf to rub
            viewdir = viewdir.permute([0, 3, 1, 2])
            NdotV = torch.sum(viewdir * n, dim=1, keepdim=True)
            del viewdir
            all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], data['all_i'])
            n_low = self.im_down(n).permute([0, 2, 3, 1])
            n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
            a_low = self.im_down(a).permute([0, 2, 3, 1]).unsqueeze(-2)
            r_low = self.im_down(r).permute([0, 2, 3, 1]).unsqueeze(-2)
            N2C = get_N2C(n_low, self.up).squeeze(-3)
            light_feat = self.InLightEncoder(data['i'], d, n, a, r, NdotV)
            light_feat = light_feat.permute([0, 2, 3, 1])

            shading = self.ShadingNet(light_feat).unsqueeze(-2)
            diffuse = self.re_arr(a_low * shading)
            C2N = torch.linalg.pinv(N2C)
            ref_dir = 2 * torch.sum(viewdir * n_low, dim=-1, keepdim=True) * n_low - viewdir
            # ref_dir in normal coord
            ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
            n_dot_v = torch.sum(n_low * viewdir, dim=-1, keepdim=True)
            specular, spec_feat, spec_in = self.SpecularNet(light_feat, n_dot_v, ref_dir, None, r_low)
            specular = self.re_arr(specular)
            diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                        diffuse, specular, mask, scale_type=1)

            featmaps = self.ContextNet(data['i'], d, self.empty, n, self.empty, self.empty)
            brdf_feat = self.AggregationNet(all_rgb, shading, spec_feat, spec_in, proj_err, featmaps, viewdir,
                                            n_low, None)
            albedo, rough, refine_feat = self.RefineNet(data['i'], d, self.empty, n, brdf_feat)
            albedo = self.AlbedoFusion(data['i'], refine_feat, a, r, diffuse, specular)
            a_low = self.im_down(albedo).permute([0, 2, 3, 1]).unsqueeze(-2)
            r_low = self.im_down(rough).permute([0, 2, 3, 1]).unsqueeze(-2)
            viewdir = viewdir[..., :1, :]
            ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
            ls_rub = (ls_rdf * self.rub_to_rdf)
            diffuse, specular, _ = pbr(viewdir, ls_rub, n_low, a_low,
                                       r_low, self.ndotl, self.envWeight_ndotl, data['e'])
            diffuse = self.re_arr(diffuse)
            specular = self.re_arr(specular)
            diffuse, specular, _, _ = LSregressDiffSpec(diffuse, specular, self.re_arr(all_rgb),
                                                        diffuse, specular, mask, scale_type=1)
            diffuse = torch.clamp(diffuse, 0, 1.0)
            specular = torch.clamp(specular, 0, 1.0)
            pred_rgb = torch.clamp(diffuse + specular, 0, 1.0)
            return diffuse, specular, pred_rgb, self.re_arr(all_rgb), get_mask_dict('default', mask)

        elif mode == 'gt_mg_gt_l':
            all_rgb, viewdir, proj_err, _ = compute_projection(self.pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], data['all_i'])
            viewdir = viewdir[..., :1, :]
            n_low = self.im_down(data['n']).permute([0, 2, 3, 1])
            n_low = F.normalize(n_low, p=2.0, dim=-1).unsqueeze(-2)
            a_low = self.im_down(data['a']).permute([0, 2, 3, 1]).unsqueeze(-2)
            r_low = self.im_down(data['r']).permute([0, 2, 3, 1]).unsqueeze(-2)

            N2C = get_N2C(n_low, self.up).squeeze(-3)
            ls_rdf = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1) * self.rub_to_rdf
            ls_rub = (ls_rdf * self.rub_to_rdf)

            diffuse, specular, _ = pbr(viewdir, ls_rub, n_low, a_low,
                                       r_low, self.ndotl, self.envWeight_ndotl, data['e'])
            diffuse = self.re_arr(diffuse)
            specular = self.re_arr(specular)
            diffuse = torch.clamp(diffuse, 0, 1.0)
            specular = torch.clamp(specular, 0, 1.0)
            pred_rgb = torch.clamp(diffuse + specular, 0, 1.0)
            return diffuse, specular, pred_rgb, self.re_arr(all_rgb), get_mask_dict('default', mask)

    def mat_edit_func(self, mask, cam, im, n, a, r, light, is_ilr):
        norm_coord = (torch.inverse(cam[:, None, None]) @ self.pixels).squeeze(-1)
        viewdir = F.normalize(norm_coord, dim=-1, p=2.0)
        viewdir[..., :1] = -viewdir[..., :1]  # view dir is surface -> camera, rdf to rub

        viewdir = viewdir.unsqueeze(-2)
        n = n.permute([0, 2, 3, 1]).unsqueeze(-2)
        a = a.permute([0, 2, 3, 1]).unsqueeze(-2)
        r = r.permute([0, 2, 3, 1]).unsqueeze(-2)

        N2C = get_N2C(n, self.up).squeeze(-3)
        if is_ilr:
            light_feat = F.interpolate(light.permute([0, 3, 1, 2]), scale_factor=2.0, mode='bilinear')
            light_feat = light_feat.permute([0, 2, 3, 1])

            shading = self.ShadingNet(light_feat).unsqueeze(-2)
            diffuse = a * shading
            C2N = torch.linalg.pinv(N2C)
            ref_dir = 2 * torch.sum(viewdir * n, dim=-1, keepdim=True) * n - viewdir
            ref_dir = torch.einsum('bhwqp,bhwvp->bhwvq', C2N, ref_dir)
            NdotV = torch.sum(viewdir * n, dim=-1, keepdim=True)
            specular, _, _ = self.SpecularNet(light_feat, NdotV, ref_dir, None, r)
        else:
            light = rearrange(light, 'b C r c h w -> b r c (h w) C')
            ls_rub = (N2C.unsqueeze(-3) @ self.ls).squeeze(-1)
            diffuse, specular, _ = pbr(viewdir, ls_rub, n, a, r, self.ndotl, self.envWeight_ndotl, light)

        diffuse = self.re_arr(diffuse)
        specular = self.re_arr(specular)
        diff_scaled, spec_scaled, cd, cs = LSregressDiffSpec(diffuse, specular, im,
                                                             diffuse, specular, mask,
                                                             scale_type=1)
        rgb_pred = torch.clamp(diff_scaled + spec_scaled, 0, 1.0)
        return rgb_pred
