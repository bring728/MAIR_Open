from utils import *
from .net_backbone import *


class MGNet(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.enc = EncoderDenseNet(6, cfg.init_feature)
        self.normal_dec = DecoderDenseNet(3, cfg.init_feature)
        self.depth_dec = DecoderDenseNet(1, cfg.init_feature)
        self.albedo_dec = DecoderDenseNet(3, cfg.init_feature)
        self.rough_dec = DecoderDenseNet(1, cfg.init_feature)

    def forward(self, rgb, d, c, dg, out_key='ndar'):
        with torch.set_grad_enabled(self.is_train):
            x = torch.cat([rgb, d, c, dg], dim=1)
            normal, depth, albedo, rough = None, None, None, None
            hs = self.enc(x)
            if 'n' in out_key:
                x_out = self.normal_dec(hs)
                x_out = torch.clamp(1.01 * torch.tanh(x_out), -1, 1)
                normal = F.normalize(x_out, p=2.0, dim=1)
            if 'd' in out_key:
                depth = to_01(self.depth_dec(hs))
            if 'a' in out_key:
                albedo = to_01(self.albedo_dec(hs))
            if 'r' in out_key:
                rough = to_01(self.rough_dec(hs))
            return normal, depth, albedo, rough


class InLightSG(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.e_type = cfg.e_type if hasattr(cfg, 'e_type') else cfg_all.e_type

        self.env_w = cfg_all.env_width
        self.env_h = cfg_all.env_height
        self.ang_res = self.env_w * self.env_h
        Az_org = ((np.arange(self.env_w) + 0.5) / self.env_w - 0.5) * 2 * np.pi  # -pi ~ pi
        El_org = ((np.arange(self.env_h) + 0.5) / self.env_h) * np.pi / 2.0  # 0 ~ pi/2
        Az, El = np.meshgrid(Az_org, El_org)
        Az_flat = Az.reshape(-1, 1)
        El_flat = El.reshape(-1, 1)
        lx_dir = np.sin(El_flat) * np.cos(Az_flat)
        ly_dir = np.sin(El_flat) * np.sin(Az_flat)
        lz_dir = np.cos(El_flat)
        ls = torch.from_numpy(np.concatenate((lx_dir, ly_dir, lz_dir), axis=-1).astype(np.float32))

        self.SGNum = cfg.SGNum
        if self.e_type == 'e_d':
            if cfg_all.d_type == 'cds':
                in_ch = 8
            elif cfg_all.d_type == 'net':
                in_ch = 7
            self.enc = EncoderBasic(in_ch, 'batch', [64, 128, 256, 256, 512, 512, 1024])
            self.dec_axis = DecoderBasic(cfg.SGNum * 3, 'batch', [128, 256, 512, 512, 1024])
            self.dec_sharp = DecoderBasic(cfg.SGNum, 'batch', [128, 256, 512, 512, 1024])
            self.dec_vis = DecoderBasic(cfg.SGNum, 'batch', [128, 256, 512, 512, 1024])
            self.layer_light_final = make_layer(in_ch=1024, out_ch=1024, kernel=3, stride=1, norm_layer='None',
                                                act='None')
            self.layer_intensity = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(inplace=True),
                                                 nn.Linear(128, 128), nn.ReLU(inplace=True),
                                                 nn.Linear(128, 128), nn.ReLU(inplace=True),
                                                 nn.Linear(128, cfg.SGNum * 3), )
        elif self.e_type == 'e':
            in_ch = 12
            self.enc = EncoderDenseNet(in_ch, cfg.init_feature, (6, 12, 24, 16, 16), downsize=False)
            self.dec_axis = DecoderDenseNetSVL(cfg.SGNum * 3, cfg.init_feature)
            self.dec_sharp = DecoderDenseNetSVL(cfg.SGNum, cfg.init_feature)
            self.dec_vis = DecoderDenseNetSVL(cfg.SGNum * 3, cfg.init_feature)

        ls = ls.reshape([self.env_h, self.env_w, 3])
        ls = rearrange(ls, 'r c p -> () () p () () r c')
        self.register_buffer("ls", ls, persistent=False)

    def fromSGtoIm(self, axis, sharp, intensity):
        axis = axis[..., None, None]
        intensity = intensity[..., None, None]
        intensity = torch.tan(np.pi / 2 * 0.999 * intensity)
        sharp = sharp[..., None, None]
        sharp = torch.tan(np.pi / 2 * 0.999 * sharp)
        mi = sharp * (torch.sum(axis * self.ls, dim=2, keepdim=True) - 1)
        envmaps = intensity * torch.exp(mi)
        envmaps = torch.sum(envma xczax c  ps, dim=1)
        return envmaps

    # mode 0 : input x,       return env
    # mode 1 : input x        return feature
    # mode 2 : input feature, return env
    def forward(self, rgb, d, c, n, a, r, ndotv, mode=0):
        with torch.set_grad_enabled(self.is_train):
            x = torch.cat([rgb, d, c, n, a, r, ndotv], dim=1)  # rub
            hs = self.enc(x)
            axis = self.dec_axis(hs)
            sharp = self.dec_sharp(hs)
            vis = self.dec_vis(hs)
            bn, _, row, col = vis.size()

            axis = torch.clamp(1.01 * torch.tanh(axis), -1, 1)
            axis = axis.view(bn, self.SGNum, 3, row, col)
            axis = F.normalize(axis, p=2.0, dim=2)
            sharp = to_01(sharp)
            sharp = sharp.view(bn, self.SGNum, 1, row, col)
            vis = to_01(vis)

            if self.e_type == 'e_d':
                vis = vis.view(bn, self.SGNum, 1, row, col)

                x_light = self.layer_light_final(hs[-1])
                x_color = self.layer_intensity(F.adaptive_avg_pool2d(x_light, (1, 1))[..., 0, 0])
                intensity = to_01(x_color)
                intensity = intensity.view(bn, self.SGNum, 3, 1, 1)
                intensity = intensity * vis
            elif self.e_type == 'e':
                intensity = vis.view(bn, self.SGNum, 3, row, col)
                vis = None

            if mode == 1:
                return axis, sharp, intensity, vis
            else:
                return axis, sharp, intensity, vis, self.fromSGtoIm(axis, sharp, intensity)


class ExDLVSG(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        cube_res = 32
        self.cfg = cfg

        if hasattr(cfg, 'dim'):
            # ILR input, feature volume output
            in_ch = cfg_all.InLightEncoder.dim + 4
            filters = cfg.filters
            self.out_dim = cfg.dim
        else:
            # Exitant direct lighting
            if cfg_all.d_type == 'cds':
                in_ch = 8
            else:
                in_ch = 7
            filters = [64, 128, 256, 256, 512, 512, 512]
            self.out_dim = 8

        self.enc = EncoderBasic(in_ch, cfg.norm, filters)
        self.cube_res = cube_res
        x, y, z = np.meshgrid(np.arange(cube_res), np.arange(cube_res), np.arange(cube_res), indexing='ij')
        x = x.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
        x = 2.0 * x / cube_res - 1
        y = y.reshape(-1).astype(dtype=np.float32) + 0.5
        y = 2.0 * y / cube_res - 1
        z = z.reshape(-1).astype(dtype=np.float32) + 0.5
        z = 2.0 * z / cube_res - 1
        coords = np.stack([x, y, z], axis=0)
        self.register_buffer("volume_coord", torch.from_numpy(coords).unsqueeze(0), persistent=False)
        self.decoder = DecoderCBatchNorm2(c_dim=filters[-1], out_dim=self.out_dim)

    def forward(self, x, d, c, n):
        with torch.set_grad_enabled(self.is_train):
            x = torch.cat([x, d, c, n], dim=1)  # rub
            hs = self.enc(x)
            feature = F.adaptive_avg_pool2d(hs[-1], (1, 1))[..., 0, 0]
            x_out = self.decoder(self.volume_coord, feature).reshape(
                [x.size(0), self.out_dim, self.cube_res, self.cube_res, self.cube_res])
            return x_out

class ContextNet(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.net_in = cfg.net_in
        input_ch = 0
        input_ch += 3 if 'rgb' in cfg.net_in else 0
        input_ch += 3 if 'n' in cfg.net_in else 0
        input_ch += 1 if 'd' in cfg.net_in else 0
        input_ch += 1 if 'c' in cfg.net_in else 0
        input_ch += 3 if 'a' in cfg.net_in else 0
        input_ch += 1 if 'r' in cfg.net_in else 0
        self.net = ResUNet(input_ch, cfg.dim, cfg.norm)

    def forward(self, rgb, d, c, n, a, r):
        with torch.set_grad_enabled(self.is_train):
            # rgb = rgb if 'rgb' in self.net_in else torch.tensor([], device=rgb.device)
            # d = d if 'd' in self.net_in else torch.tensor([], device=rgb.device)
            # c = c if 'c' in self.net_in else torch.tensor([], device=rgb.device)
            # n = n if 'n' in self.net_in else torch.tensor([], device=rgb.device)
            # a = a if 'a' in self.net_in else torch.tensor([], device=rgb.device)
            # r = r if 'r' in self.net_in else torch.tensor([], device=rgb.device)
            x = torch.cat([rgb, d, c, n, a, r], dim=1)
            return self.net(x)


class RefineNet(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.type = cfg.type
        self.net_in = cfg.net_in
        self.interp = lambda x: F.interpolate(x, [cfg_all.imHeight, cfg_all.imWidth], mode='bilinear',
                                              align_corners=True)
        empty = torch.tensor([])
        self.register_buffer("empty", empty, persistent=False)

        input_ch = 0
        input_ch += 3 if 'rgb' in cfg.net_in else 0
        input_ch += 3 if 'n' in cfg.net_in else 0
        input_ch += 1 if 'd' in cfg.net_in else 0
        input_ch += 1 if 'c' in cfg.net_in else 0
        input_ch += 3 if 'a' in cfg.net_in else 0
        input_ch += 1 if 'r' in cfg.net_in else 0
        input_ch += 3 if 'diff' in cfg.net_in else 0
        input_ch += 3 if 'spec' in cfg.net_in else 0
        input_ch += 3 if 'error' in cfg.net_in else 0
        input_ch += cfg_all.AggregationNet.out_dim if 'brdf_feat' in cfg.net_in else 0
        norm = cfg.norm
        drop = cfg.get('dropout', 0.0)

        if self.type == 'old':
            self.refine_d_1 = make_layer(pad_type='rep', in_ch=input_ch, out_ch=128, kernel=4, stride=2,
                                         norm_layer=norm, dropout=drop)
            self.refine_d_2 = make_layer(in_ch=128, out_ch=128, kernel=4, stride=2, norm_layer=norm, dropout=drop)
            self.refine_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, norm_layer=norm, dropout=drop)
            self.refine_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, norm_layer=norm, dropout=drop)
            self.refine_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, norm_layer=norm, dropout=drop)
            self.refine_d_6 = make_layer(in_ch=512, out_ch=1024, kernel=3, stride=1, norm_layer=norm, dropout=drop)

            self.refine_albedo_u_1 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, norm_layer=norm,
                                                dropout=drop)
            self.refine_albedo_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, norm_layer=norm,
                                                dropout=drop)
            self.refine_albedo_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, norm_layer=norm,
                                                dropout=drop)
            self.refine_albedo_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, norm_layer=norm,
                                                dropout=drop)
            self.refine_albedo_u_5 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, norm_layer=norm,
                                                dropout=drop)
            self.refine_albedo_u_6 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, norm_layer=norm,
                                                dropout=drop)
            self.refine_albedo_final = make_layer(pad_type='rep', in_ch=128, out_ch=3, kernel=3, stride=1, act='None',
                                                  norm_layer='None')

            self.refine_rough_u_1 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, norm_layer=norm,
                                               dropout=drop)
            self.refine_rough_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, norm_layer=norm,
                                               dropout=drop)
            self.refine_rough_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, norm_layer=norm, dropout=drop)
            self.refine_rough_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, norm_layer=norm, dropout=drop)
            self.refine_rough_u_5 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, norm_layer=norm, dropout=drop)
            self.refine_rough_u_6 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, norm_layer=norm, dropout=drop)
            self.refine_rough_final = make_layer(pad_type='rep', in_ch=128, out_ch=3, kernel=3, stride=1, act='None',
                                                 norm_layer='None')

        else:
            filters = cfg.filters
            self.length = len(filters)
            self.enc1 = nn.ModuleList()
            for i in range(self.length):
                if i == 0:
                    layer_tmp1 = nn.Sequential(nn.Conv2d(input_ch, filters[i], kernel_size=1, stride=1, padding=0),
                                               nn.ReLU(inplace=True))

                elif 0 < i < self.length - 1:
                    layer_tmp1 = nn.Sequential(
                        make_layer(in_ch=filters[i - 1], out_ch=filters[i], kernel=5, stride=2, padding=2,
                                   norm_layer=norm),
                        make_layer(in_ch=filters[i], out_ch=filters[i], kernel=3, stride=1, norm_layer=norm,
                                   dropout=drop))

                elif i == self.length - 1:
                    layer_tmp1 = make_layer(in_ch=filters[i - 1], out_ch=filters[i], kernel=3, stride=1,
                                            norm_layer=norm)
                self.enc1.append(layer_tmp1)

            filters = filters[::-1]
            filters.append(filters[-1] // 2)
            self.dec_albedo = nn.ModuleList()
            self.dec_rough = nn.ModuleList()
            self.length = len(filters) - 1
            for i in range(self.length):
                if i == 0:
                    layer_tmp1 = make_layer(in_ch=filters[i], out_ch=filters[i + 1], norm_layer=norm, dropout=drop)
                    layer_tmp2 = make_layer(in_ch=filters[i], out_ch=filters[i + 1], norm_layer=norm, dropout=drop)
                else:
                    layer_tmp1 = make_layer(in_ch=filters[i] * 2, out_ch=filters[i + 1], norm_layer=norm, dropout=drop)
                    layer_tmp2 = make_layer(in_ch=filters[i] * 2, out_ch=filters[i + 1], norm_layer=norm, dropout=drop)
                self.dec_albedo.append(layer_tmp1)
                self.dec_rough.append(layer_tmp2)
            self.dec_albedo_final = nn.Conv2d(filters[-1], 3, kernel_size=3, stride=1, padding=1)
            self.dec_rough_final = nn.Conv2d(filters[-1], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, d, c, n, brdf_feat):
        with torch.set_grad_enabled(self.is_train):
            # rgb = rgb if 'rgb' in self.net_in else self.empty
            # d = d if 'd' in self.net_in else self.empty
            # c = c if 'c' in self.net_in else self.empty
            # n = n if 'n' in self.net_in else self.empty
            # a = a if 'a' in self.net_in else self.empty
            # r = r if 'r' in self.net_in else self.empty
            # diff = self.interp(diff) if 'diff' in self.net_in else self.empty
            # spec = self.interp(spec) if 'spec' in self.net_in else self.empty
            # error = self.interp(error) if 'error' in self.net_in else self.empty
            brdf_feat = self.interp(brdf_feat) if 'brdf_feat' in self.net_in else self.empty
            # x = torch.cat([rgb, d, n, a, r, diff, spec, error, context_feat, brdf_feat], dim=1)
            x = torch.cat([rgb, d, c, n, brdf_feat], dim=1)

            if self.type == 'old':
                x1 = self.refine_d_1(x)
                x2 = self.refine_d_2(x1)
                x3 = self.refine_d_3(x2)
                x4 = self.refine_d_4(x3)
                x5 = self.refine_d_5(x4)
                x6 = self.refine_d_6(x5)

                dx1 = self.refine_albedo_u_1(x6)
                dx2 = self.refine_albedo_u_2(cat_up(dx1, x5))
                dx3 = self.refine_albedo_u_3(cat_up(dx2, x4))
                dx4 = self.refine_albedo_u_4(cat_up(dx3, x3))
                dx5 = self.refine_albedo_u_5(cat_up(dx4, x2))
                dx6_albedo = self.refine_albedo_u_6(cat_up(dx5, x1))
                albedo = to_01(self.refine_albedo_final(dx6_albedo))

                dx1 = self.refine_rough_u_1(x6)
                dx2 = self.refine_rough_u_2(cat_up(dx1, x5))
                dx3 = self.refine_rough_u_3(cat_up(dx2, x4))
                dx4 = self.refine_rough_u_4(cat_up(dx3, x3))
                dx5 = self.refine_rough_u_5(cat_up(dx4, x2))
                dx6_rough = self.refine_rough_u_6(cat_up(dx5, x1))
                rough = to_01(self.refine_rough_final(dx6_rough))
                rough = torch.mean(rough, dim=1, keepdim=True)
                return albedo, rough, dx6_albedo
            else:
                hs1 = []
                hs2 = []
                for layer in self.enc1:
                    x = layer(x)
                    hs1.append(x)

                for i in range(self.length):
                    h1 = hs1[-i - 1]
                    h2 = hs2[-i - 1] if hs2 else h1

                    if i == 0:
                        x_albedo = self.dec_albedo[i](h1)
                        x_rough = self.dec_rough[i](h2)
                    elif 0 < i < self.length - 1:
                        x_albedo = self.dec_albedo[i](cat_up(x_albedo, h1))
                        x_rough = self.dec_rough[i](cat_up(x_rough, h2))
                    elif i == self.length - 1:
                        x_albedo = self.dec_albedo[i](torch.cat([x_albedo, h1], dim=1))
                        x_rough = self.dec_rough[i](torch.cat([x_rough, h2], dim=1))

                albedo = to_01(self.dec_albedo_final(x_albedo))
                rough = to_01(self.dec_rough_final(x_rough))
                return albedo, rough


class VSGEncoder(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.use_checkpoint = cfg_all.use_checkpoint

        self.vsg_type = cfg.vsg_type
        use_exitant = False
        exi_dim = 0
        if self.vsg_type == 'voxel':
            if cfg.src_type == 'inci':
                if hasattr(cfg_all, 'InLightEncoder'):
                    input_ch = cfg_all.InLightEncoder.dim + 16
                elif hasattr(cfg_all, 'InLightSG'):
                    input_ch = cfg_all.InLightSG.SGNum * 7 + 10
                else:
                    raise Exception('inlight..')
            else:
                use_exitant = True
                if cfg.src_type == 'exi' and hasattr(cfg_all.ExDLVSG, 'dim'):
                    # ILR input, feature volume output
                    exi_dim = cfg_all.ExDLVSG.dim
                    input_ch = 16
                else:
                    # Exitant direct lighting input
                    exi_dim = 8
                    input_ch = 10

            n_blocks = len(cfg.filters)
            filters = [input_ch, ] + cfg.filters
            self.parameterize = cfg.parameterize
            if self.parameterize == 'sg':
                vsg_out_ch = 8
                # if cfg_all.VSGDecoder.use and cfg_all.VSGDecoder.vsg_rgb:
                #     vsg_out_ch += 3
            elif self.parameterize == 'feat':
                vsg_out_ch = 33
            self.unet = Unet3D(n_blocks, cfg.norm, filters, vsg_out_ch, use_exitant, exi_dim)

    def forward(self, x, global_vol=None):
        with torch.set_grad_enabled(self.is_train):
            if self.vsg_type == 'voxel':
                x = self.unet(x, global_vol)
                # if self.parameterize == 'sg':
                #     x = 1.01 * torch.tanh(x)
                #     x = torch.clamp(x, -1, 1)
                #     x[:, 3:] = 0.5 * (x[:, 3:] + 1)
                #     return x
                #
                # elif self.parameterize == 'feat':
                #     x[:, -1:] = to_01(x[:, -1:])
                #     return x
                return x