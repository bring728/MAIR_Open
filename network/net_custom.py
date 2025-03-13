from utils import *
# from eg3d.volumetric_rendering.renderer import ImportanceRenderer
# from eg3d.triplane import OSGDecoder
from .net_backbone import *
from .network_utils import *


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
        envmaps = torch.sum(envmaps, dim=1)
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


class InLightEncoder(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        in_ch = 12
        self.enc = EncoderDenseNet(in_ch, cfg.init_feature, (6, 12, 24, 16, 16), downsize=False)
        self.dec_feat = DecoderDenseNetSVL(cfg.dim, cfg.init_feature)

    def forward(self, rgb, d, n, a, r, ndotv):
        with torch.set_grad_enabled(self.is_train):
            x = torch.cat([rgb, d, n, a, r, ndotv], dim=1)  # rub
            hs = self.enc(x)
            feat = self.dec_feat(hs)
            return feat


class InLightDecoder(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        if not hasattr(cfg, 'angular'):
            self.angular = True
        else:
            self.angular = cfg.angular
        self.chunk_size = cfg.chunk_size
        if self.angular:
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
            dir_enc = FreqEncoder(input_dim=3, max_freq_log2=3, N_freqs=4, log_sampling=True)
            ls = dir_enc(ls)
            self.register_buffer("ls", ls, persistent=False)
            in_dim = cfg_all.InLightEncoder.dim + dir_enc.output_dim
            last_dim = 3
        else:
            in_dim = cfg_all.InLightEncoder.dim
            self.ang_res = 1
            last_dim = cfg_all.env_height * cfg_all.env_width * 3

        hidden = cfg.hidden
        self.num_layers = cfg.num_layers
        self.net_act = nn.ReLU(inplace=True)
        env_net = []
        out_dim = hidden
        for l in range(self.num_layers - 1):
            env_net.append(nn.Linear(in_dim, out_dim, bias=True))
            in_dim = out_dim

        if hasattr(cfg, 'last_bias'):
            env_net.append(nn.Linear(in_dim, last_dim, bias=cfg.last_bias))
        else:
            env_net.append(nn.Linear(in_dim, last_dim))

        self.dec_env = nn.ModuleList(env_net)
        init_seq(self.dec_env, 'xavier_uniform', self.net_act)

    def forward(self, feat):
        with torch.set_grad_enabled(self.is_train):
            b, h, w, _ = feat.shape
            feat_flat = rearrange(feat, 'b h w c -> (b h w) c')
            feat_flat = repeat(feat_flat, 'n c -> n l c', l=self.ang_res)

            if self.angular:
                if self.chunk_size > 0:
                    x_list = []
                    for j in range(0, feat_flat.shape[0], self.chunk_size):
                        xj = feat_flat[j:j + self.chunk_size]
                        xj = torch.cat([xj, repeat(self.ls, 'l c -> n l c', n=xj.shape[0])], dim=-1)
                        xj = rearrange(xj, 'n l c -> (n l) c')
                        for i in range(self.num_layers):
                            xj = self.dec_env[i](xj)
                            if i != self.num_layers - 1:
                                xj = self.net_act(xj)
                        x_list.append(xj)
                    x = torch.cat(x_list, dim=0)
                else:
                    ls = repeat(self.ls, 'l c -> n l c', n=b * h * w)
                    x = torch.cat([feat_flat, ls], dim=-1)
                    x = rearrange(x, 'n l c -> (n l) c')
                    for i in range(self.num_layers):
                        x = self.dec_env[i](x)
                        if i != self.num_layers - 1:
                            x = self.net_act(x)
            else:
                x = rearrange(feat_flat, 'n l c -> (n l) c')
                for i in range(self.num_layers):
                    x = self.dec_env[i](x)
                    if i != self.num_layers - 1:
                        x = self.net_act(x)

            x = x.reshape([b, h, w, -1, 3])
            x_01 = to_01(x)
            x = torch.tan(np.pi / 2 * 0.999 * x_01)
            return x, x_01


class ShadingNet(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        in_dim = cfg_all.InLightEncoder.dim
        net = []
        self.net_act = nn.ReLU(inplace=True)
        out_dim = cfg.hidden
        self.num_layers = cfg.num_layers
        for l in range(self.num_layers - 1):
            net.append(nn.Linear(in_dim, out_dim, bias=True))
            in_dim = out_dim

        if hasattr(cfg, 'last_bias'):
            net.append(nn.Linear(in_dim, 3, bias=cfg.last_bias))
        else:
            net.append(nn.Linear(in_dim, 3))
        self.net = nn.ModuleList(net)
        init_seq(self.net, 'xavier_uniform', self.net_act)

    def forward(self, x):
        with torch.set_grad_enabled(self.is_train):
            for i in range(self.num_layers):
                x = self.net[i](x)
                if i != self.num_layers - 1:
                    x = self.net_act(x)
            x_01 = to_01(x)
            x = torch.tan(np.pi / 2 * 0.999 * x_01)
            return x


class SpecularNet(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.dir_enc = FreqEncoder(input_dim=3, max_freq_log2=3, N_freqs=4, log_sampling=True)
        net = []
        self.net_act = nn.ReLU(inplace=True)
        out_dim = cfg.hidden
        self.num_layers = cfg.num_layers
        if self.num_layers == 5:
            self.rough_after = True
            in_dim = cfg_all.InLightEncoder.dim + 1 + self.dir_enc.output_dim
            for l in range(self.num_layers - 2):
                net.append(nn.Linear(in_dim, out_dim, bias=True))
                in_dim = out_dim
            net.append(nn.Linear(in_dim + 1, out_dim))
            net.append(nn.Linear(out_dim, 3))
        elif self.num_layers == 4:
            self.rough_after = False
            in_dim = cfg_all.InLightEncoder.dim + 2 + self.dir_enc.output_dim
            for l in range(self.num_layers - 1):
                net.append(nn.Linear(in_dim, out_dim, bias=True))
                in_dim = out_dim
            if hasattr(cfg, 'last_bias'):
                net.append(nn.Linear(in_dim, 3, bias=cfg.last_bias))
            else:
                net.append(nn.Linear(in_dim, 3))

        self.net = nn.ModuleList(net)
        init_seq(self.net, 'xavier_uniform', self.net_act)

    def forward(self, light_feat, n_dot_v, ref_dir, spec_feat, r):
        with torch.set_grad_enabled(self.is_train):
            if self.rough_after:
                spec_in = None
                if spec_feat is None:
                    v = n_dot_v.shape[-2]
                    spec_in = torch.cat([repeat(light_feat, 'b h w c -> b h w v c', v=v),
                                         n_dot_v, self.dir_enc(ref_dir)], dim=-1)
                    x = spec_in
                    for i in range(self.num_layers - 2):
                        x = self.net[i](x)
                        if i != self.num_layers - 3:
                            x = self.net_act(x)
                    spec_feat = x

                v = spec_feat.shape[-2]
                x = torch.cat([spec_feat, repeat(r.squeeze(-2), 'b h w c -> b h w v c', v=v)], dim=-1)
                x = self.net[-2](x)
                x = self.net_act(x)
                x = self.net[-1](x)
                x_01 = to_01(x)
                specular = torch.tan(np.pi / 2 * 0.999 * x_01)
                return specular, spec_feat, spec_in
            else:
                v = n_dot_v.shape[-2]
                x_in = torch.cat([repeat(light_feat, 'b h w c -> b h w v c', v=v),
                                  repeat(r.squeeze(-2), 'b h w c -> b h w v c', v=v),
                                  n_dot_v, self.dir_enc(ref_dir)], dim=-1)

                x = x_in
                for i in range(self.num_layers):
                    x = self.net[i](x)
                    if i != self.num_layers - 1:
                        x = self.net_act(x)
                x_01 = to_01(x)
                specular = torch.tan(np.pi / 2 * 0.999 * x_01)
                return specular, None, torch.cat(
                    [x_in[..., :-self.dir_enc.output_dim - 2],
                     x_in[..., -self.dir_enc.output_dim - 1:]], -1)


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


class AlbedoFusion(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.is_rough = False
        out_ch = 3
        in_ch = 128 + 12
        if hasattr(cfg, 'is_rough'):
            self.is_rough = cfg.is_rough
        if self.is_rough:
            out_ch = 1
            in_ch = 128 + 10

        self.interp = lambda x: F.interpolate(x, [cfg_all.imHeight, cfg_all.imWidth], mode='bilinear',
                                              align_corners=True)

        self.sv_refine_1 = make_layer(in_ch=in_ch, out_ch=256, kernel=3, stride=1, norm_layer='group')
        self.sv_refine_2 = make_layer(pad_type='rep', in_ch=256, out_ch=out_ch, kernel=3, stride=1, act='None',
                                      norm_layer='None')

    def forward(self, rgb, refine_feat, a, r, diff, spec):
        with torch.set_grad_enabled(self.is_train):
            x_in = r if self.is_rough else a
            x = torch.cat([refine_feat, rgb, x_in, self.interp(diff), self.interp(spec)], dim=1)
            x = self.sv_refine_1(x)
            albedo_sv = to_01(self.sv_refine_2(x))
            return albedo_sv

    # def __init__(self, cfg_all, cfg):
    #     super().__init__()
    #     self.interp = lambda x: F.interpolate(x, [cfg_all.imHeight, cfg_all.imWidth], mode='bilinear',
    #                                           align_corners=True)
    #     norm = 'group'
    #     self.refine_d_1 = make_layer(pad_type='rep', in_ch=128 + 12, out_ch=128, kernel=4, stride=2,
    #                                  norm_layer=norm)
    #     self.refine_d_2 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, norm_layer=norm)
    #     self.refine_d_3 = make_layer(in_ch=256, out_ch=512, kernel=3, stride=1, norm_layer=norm)
    #
    #     self.refine_albedo_u_1 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, norm_layer=norm)
    #     self.refine_albedo_u_2 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, norm_layer=norm)
    #     self.refine_albedo_u_3 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, norm_layer=norm)
    #
    #     self.refine_albedo_final = make_layer(pad_type='rep', in_ch=128, out_ch=3, kernel=3, stride=1, act='None',
    #                                           norm_layer='None')
    #
    # def forward(self, rgb, refine_feat, a, diff, spec):
    #     with torch.set_grad_enabled(self.is_train):
    #         x = torch.cat([refine_feat, rgb, a, self.interp(diff), self.interp(spec)], dim=1)
    #
    #         x1 = self.refine_d_1(x)
    #         x2 = self.refine_d_2(x1)
    #         x3 = self.refine_d_3(x2)
    #
    #         dx1 = self.refine_albedo_u_1(x3)
    #         dx2 = self.refine_albedo_u_2(cat_up(dx1, x2))
    #         dx3 = self.refine_albedo_u_3(cat_up(dx2, x1))
    #         albedo = to_01(self.refine_albedo_final(dx3))
    #         return albedo


class AngularTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_head, depth, out_dim, out_hidden, dropout=0.,
                 attn_type='mask', target_embed=True, occ_th=0.05, mlp_ratio=4):
        super().__init__()
        self.depth = depth
        self.attn_type = attn_type
        self.occ_th = occ_th
        # from BERT
        self.target_embed = nn.Parameter(torch.empty(1, 1, input_dim).normal_(std=0.02)) if target_embed else None

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(input_dim, num_heads, dim_head, dropout, mlp_ratio, attn_type) for _ in
             range(depth)]
        )
        self.proj_out = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, out_hidden), nn.GELU(),
            nn.Linear(out_hidden, out_hidden), nn.GELU(),
            nn.Linear(out_hidden, out_dim),
        )

    def forward(self, x, proj_err):
        b, h, w, v, c = x.shape
        x = x.reshape([b * h * w, v, -1])
        if self.target_embed is not None:
            x = torch.cat([self.target_embed.expand(b * h * w, -1, -1), x], dim=1)
            v += 1

        if self.attn_type == 'weight':
            weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
            if self.target_embed is not None:
                weight = torch.cat([weight[..., :1, :], weight], dim=-2)
            # weight = F.normalize(weight, dim=-2, p=1.0)
            weight = weight.reshape([b * h * w, v])
            mask = None
        elif self.attn_type == 'mask':
            weight = None
            mask = (torch.abs(proj_err[..., 1:, :]) < self.occ_th).squeeze(-1)
            if self.target_embed is not None:
                tmp = torch.ones([b, h, w, 2], dtype=torch.bool, device=mask.device)
            else:
                tmp = torch.ones([b, h, w, 1], dtype=torch.bool, device=mask.device)
            mask = torch.cat([tmp, mask], dim=-1)
            mask = mask.reshape([b * h * w, v])
        else:
            weight = None
            mask = None

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, weight=weight, mask=mask, is_last=i == self.depth - 1)
        # x = x[:, 0]
        return self.proj_out(x).reshape([b, h, w, -1])


class BasicTransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_head, dropout=0.1, mlp_ratio=4, attn_type='mask'):
        super().__init__()
        self.norm_0 = nn.LayerNorm(input_dim)
        self.norm_1 = nn.LayerNorm(input_dim)
        self.attn = CrossAttention(input_dim=input_dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout,
                                   attn_type=attn_type)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(input_dim * mlp_ratio), input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_input, weight=None, mask=None, is_last=False):
        x = self.attn(self.norm_0(x_input), weight=weight, mask=mask)
        x = self.dropout(x) + x_input
        # x = self.mlp(self.norm_1(x)) + x
        if is_last:
            x = self.mlp(self.norm_1(x[..., 0, :])) + x[..., 0, :]
        else:
            x = self.mlp(self.norm_1(x)) + x
        return x


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
                if hasattr(cfg_all, 'VSGDecoder'):
                    if cfg_all.VSGDecoder.use and cfg_all.VSGDecoder.vsg_rgb:
                        vsg_out_ch += 3
            elif self.parameterize == 'feat':
                vsg_out_ch = 33
            self.unet = Unet3D(n_blocks, cfg.norm, filters, vsg_out_ch, use_exitant, exi_dim)

        elif self.vsg_type == 'triplane':
            input_ch = cfg_all.InLightEncoder.dim + 17
            patch_size = cfg.patch_size
            image_size_org = [cfg_all.envCols, cfg_all.envRows]
            self.image_size = image_size_org

            def make_divisible_by_p(x, p):
                remainder = x % p
                return 0 if remainder == 0 else p - remainder

            self.image_size[0] += make_divisible_by_p(self.image_size[0], patch_size)
            self.image_size[1] += make_divisible_by_p(self.image_size[1], patch_size)

            # vit_in_ch = int(cfg.kv_dim * self.image_size[0] // patch_size * self.image_size[1] // patch_size /
            #                 (image_size_org[0] * image_size_org[1]))
            vit_in_ch = input_ch
            self.pre_net = None
            if cfg.use_pre_net:
                vit_in_ch = 64
                self.pre_net = (nn.Sequential(
                    nn.Conv2d(input_ch, int(input_ch * 4.0), kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    # nn.Conv2d(int(input_ch * 4.0), int(input_ch * 4.0), kernel_size=1, stride=1, padding=0),
                    # nn.ReLU(inplace=True),
                    nn.Conv2d(int(input_ch * 4.0), vit_in_ch, kernel_size=1, stride=1, padding=0),
                ))

            self.enc = VisionTransformer(
                input_ch=vit_in_ch,
                image_size=self.image_size,
                patch_size=patch_size,
                num_layers=cfg.enc_num_layers,
                num_heads=cfg.enc_num_heads,
                hidden_dim=cfg.kv_dim,
                mlp_dim=cfg.mlp_dim,
            )

            self.dec = TriplaneTransformer(
                inner_dim=cfg.q_dim,
                image_feat_dim=cfg.kv_dim,
                triplane_low_res=cfg.triplane_low_res,
                num_heads=cfg.dec_num_heads,
                tri_up=cfg.tri_up,
                triplane_dim=cfg.triplane_dim,
                num_layers=cfg.dec_num_layers,
            )

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

            elif self.vsg_type == 'triplane':
                if self.pre_net is not None:
                    x = self.pre_net(x)

                n, c, h, w = x.shape
                if h != self.image_size[1]:
                    x = torch.nn.functional.pad(x, [0, 0, 0, self.image_size[1] - h, 0, 0])
                if w != self.image_size[0]:
                    x = torch.nn.functional.pad(x, [0, self.image_size[0] - w, 0, 0, 0, 0])

                scene_feature = self.enc(x)
                triplane = self.dec(scene_feature)
                return triplane


class VSGDecoder(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        # renderings
        self.use_checkpoint = cfg_all.use_checkpoint
        self.renderer = ImportanceRenderer(cfg.depth_resolution, cfg.closest_threshold,
                                           cfg.importance_sampling, cfg.vsg_rgb, cfg.anti_aliasing,
                                           cfg_all.VSGEncoder.vsg_type, cfg.geo_type, cfg.sg_order)

        self.decoder = None
        if cfg_all.VSGEncoder.vsg_type == 'triplane' or (
                cfg_all.VSGEncoder.vsg_type == 'voxel' and cfg_all.VSGEncoder.parameterize == 'feat'):
            out_ch = 8
            if cfg.vsg_rgb:
                out_ch += 3
            self.decoder = OSGDecoder(cfg_all.VSGEncoder.triplane_dim, cfg.hidden, out_ch)

    def forward(self, planes, ray_o, ray_d, bb, radii):
        return checkpoint(self._forward, (planes, ray_o, ray_d, bb, radii,), self.parameters(), self.use_checkpoint)

    def _forward(self, planes, ray_o, ray_d, bb, radii):
        with torch.set_grad_enabled(self.is_train):
            return self.renderer(planes, self.decoder, ray_o, ray_d, bb, radii)


# https://github.com/3DTopia/OpenLRM/tree/main/lrm
class TriplaneTransformer(nn.Module):
    """
    Transformer with condition and modulation that generates a triplane representation.

    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """

    def __init__(self, inner_dim: int, image_feat_dim: int,
                 triplane_low_res: int, tri_up: int, triplane_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_dim = triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3 * triplane_low_res ** 2, inner_dim) * (1. / inner_dim) ** 0.5)
        self.layers = nn.ModuleList([
            ConditionModulationBlock(inner_dim=inner_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=tri_up, stride=tri_up, padding=0)

    def forward(self, image_feats):
        # image_feats: [N, L_cond, D_cond]
        # camera_embeddings: [N, D_mod]

        N = image_feats.shape[0]
        H = W = self.triplane_low_res

        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        for layer in self.layers:
            x = layer(x, image_feats)
        x = self.norm(x)

        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3 * N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x


class ConditionModulationBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """

    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
                                                dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=num_heads,
                                               dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa)[0]
        x = x + self.mlp(self.norm3(x))
        return x


import math

# kernel, stride, padding
convnet = [[3, 1, 1], [3, 1, 1], [2, 2, 1],
           [3, 1, 1], [3, 1, 1], [2, 2, 1],
           [3, 1, 1], [3, 1, 1], [2, 2, 1],
           [3, 1, 1], [3, 1, 1], [2, 2, 1],
           [3, 1, 1], [3, 1, 1], ]
# layer_names = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'final']
# convnet = [[7, 2, 3], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], ]
# layer_names = ['conv1', 'pool1', 'layer1-1', 'layer1-2', 'layer1-3', 'layer1-4', 'layer2-1', 'layer2-2', 'layer2-3', 'layer2-4', 'layer3-1', 'layer3-2', 'layer3-3', 'layer3-4', 'layer4-1', 'layer4-2', 'layer4-3', 'layer4-4',]

imsize = 128


def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print(
        "\t size: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))


layerInfos = []
if __name__ == '__main__':
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        # printLayer(currentLayer, layer_names[i])
        printLayer(currentLayer, 'a')
    print("------------------------")
