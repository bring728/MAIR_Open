from utils import *
from torch import nn
from .net_custom import AngularTransformer
from .net_backbone import RNN, CrossAttention
from .network_utils import checkpoint, init_seq


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def fused_mean_variance(x, weight, dim):
    mean = torch.sum(x * weight, dim=dim, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=dim, keepdim=True)
    return mean, var


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


# CVPR Version
class MVANet(nn.Module):
    def __init__(self, dim, dim_head, depth, mlp_hidden, out_hidden, dim_out, attn_type):
        super().__init__()
        self.act = nn.GELU()
        self.attn_type = attn_type

        self.to_v = nn.ModuleList()
        self.to_out = nn.ModuleList()
        self.net = nn.ModuleList()

        self.depth = depth
        for i in range(depth):
            self.to_v.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head, bias=False)
            ))
            self.to_out.append(nn.Linear(dim_head * 3, dim))
            self.net.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_hidden), self.act,
                nn.Linear(mlp_hidden, dim),
            ))

        self.mlp_brdf = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_hidden), self.act,
            nn.Linear(out_hidden, out_hidden), self.act,
            nn.Linear(out_hidden, dim_out),
        )
        # self.mlp_brdf.apply(weights_init)
        # for i in range(self.depth):
        #     self.to_v[i].apply(weights_init)
        #     self.to_out[i].apply(weights_init)
        #     self.net[i].apply(weights_init)

    def forward(self, x_input, proj_err):
        if self.attn_type == 'None':
            weight = torch.ones_like(proj_err) / proj_err.shape[-2]
        else:
            weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
            weight = F.normalize(weight, dim=-2, p=1.0)

        b, h, w, n, _ = x_input.shape
        x = x_input
        for i in range(self.depth):
            v = self.to_v[i](x)

            mean, var = fused_mean_variance(v, weight, dim=-2)
            x = torch.cat([v, torch.cat([mean, var], dim=-1).expand(-1, -1, -1, n, -1)], dim=-1)

            if not i == self.depth - 1:
                x = self.to_out[i](x)
                x = x + x_input
                x = self.net[i](x)
                x = x + x_input
            else:
                x = self.to_out[i](x[..., 0, :])
                x = x + x_input[..., 0, :]
                x = self.net[i](x)
                x = x + x_input[..., 0, :]

        brdf_feature = self.mlp_brdf(x)
        return brdf_feature


class MVANet_attn(nn.Module):
    def __init__(self, dim, dim_head, depth, mlp_hidden, out_hidden, dim_out, target_embed,
                 attn_dim_head, attn_n_head, attn_type):
        super().__init__()
        dropout = 0.0
        self.act = nn.GELU()
        self.to_v = nn.ModuleList()
        self.to_out = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_norm = nn.ModuleList()
        self.attn_mlp = nn.ModuleList()
        self.net = nn.ModuleList()
        if target_embed:
            self.target_embed = nn.Parameter(torch.empty(1, 1, dim).normal_(std=0.02))
        else:
            self.target_embed = None

        self.depth = depth
        for i in range(depth):
            self.to_v.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim_head, bias=False)))
            self.to_out.append(nn.Linear(dim_head * 3, dim))

            self.attn_norm.append(nn.LayerNorm(dim))
            self.attn.append(CrossAttention(input_dim=dim, num_heads=attn_n_head,
                                            dim_head=attn_dim_head, dropout=dropout, attn_type=attn_type))
            self.attn_norm.append(nn.LayerNorm(dim))
            self.attn_mlp.append(nn.Sequential(
                nn.Linear(dim, int(dim * 4.0)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(dim * 4.0), dim),
                nn.Dropout(dropout),
            ))
            self.attn_norm.append(nn.LayerNorm(dim))
            self.attn.append(CrossAttention(input_dim=dim, num_heads=attn_n_head,
                                            dim_head=attn_dim_head, dropout=dropout, attn_type=attn_type))

            self.net.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_hidden), self.act,
                nn.Linear(mlp_hidden, dim),
            ))

        self.mlp_brdf = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_hidden), self.act,
            nn.Linear(out_hidden, out_hidden), self.act,
            nn.Linear(out_hidden, dim_out),
        )
        self.mlp_brdf.apply(weights_init)
        for i in range(self.depth):
            self.to_v[i].apply(weights_init)
            self.to_out[i].apply(weights_init)
            self.net[i].apply(weights_init)

    def forward(self, x_input, proj_err):
        b, h, w, n, c = x_input.shape
        x_input = x_input.reshape([b * h * w, n, -1])
        proj_err = proj_err.reshape([b * h * w, n, 1])

        weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
        weight = F.normalize(weight, dim=-2, p=1.0)

        mask = (torch.abs(proj_err[..., 1:, :]) < 0.05).squeeze(-1)
        tmp = torch.ones([b * h * w, 1], dtype=torch.bool, device=mask.device)
        mask = torch.cat([tmp, mask], dim=1)
        target_embed = None
        if self.target_embed is not None:
            mask = torch.cat([tmp, mask], dim=1)
            target_embed = self.target_embed.expand(b * h * w, -1, -1)

        x = x_input
        for i in range(self.depth):
            x_mva = self.to_v[i](x[:, -n:])
            mean, var = fused_mean_variance(x_mva, weight, dim=1)
            x_mva = torch.cat([x_mva, torch.cat([mean, var], dim=2).expand(-1, n, -1)], dim=2)
            x_mva = self.to_out[i](x_mva) + x_input

            if target_embed is not None:
                # if target_embed is not None and i == self.depth - 1:
                #     mask = torch.cat([tmp, mask], dim=1)
                x = torch.cat([target_embed, x], dim=1)
                # x[:, :1] += self.target_embed.expand(b * h * w, -1, -1)

            x_attn = self.attn[2 * i](self.attn_norm[3 * i](x), mask=mask, weight=weight) + x
            x_attn = self.attn_mlp[i](self.attn_norm[3 * i + 1](x_attn)) + x_attn
            x_attn = self.attn[2 * i + 1](self.attn_norm[3 * i + 2](x_attn), mask=mask, weight=weight) + x_attn
            if not i == self.depth - 1:
                if target_embed is not None:
                    target_embed = x_attn[:, :1]
                    x_attn = x_attn[:, 1:]
                x = self.net[i](x_attn + x_mva) + x_input
            else:
                x = self.net[i](x_attn[:, 0] + x_mva[:, 0]) + x_input[:, 0]
        return self.mlp_brdf(x).reshape([b, h, w, -1])


class AggregationNet(nn.Module):
    def __init__(self, cfg_all, cfg):
        super().__init__()
        self.cfg = cfg
        self.out_dim = cfg.out_dim
        context_dim = cfg_all.ContextNet.dim if hasattr(cfg_all, 'ContextNet') and cfg_all.ContextNet.use else 0

        if hasattr(cfg_all, 'InLightSG'):
            self.aggregation_type = 'SG'
            self.SGNum = cfg_all.InLightSG.SGNum
            self.DL_type = cfg.DL_type

            self.embed_dim = context_dim + 3 + cfg.light_dim
            act = nn.ELU(inplace=True)
            if hasattr(cfg_all.RefineNet, 'use') and not cfg_all.RefineNet.use:
                self.out_dim = 4

            if self.DL_type == 'none':
                self.pbr_mlp = None
                self.embed_dim = context_dim + 3
            else:
                if self.DL_type == 'raw':
                    input_ch = 13
                elif self.DL_type == 'microfacet':
                    input_ch = 8  # intensity, sharp, fresnel, dot, dot, dot
                self.pbr_mlp = nn.Sequential(nn.Linear(input_ch, cfg.light_hidden), act,
                                             nn.Linear(cfg.light_hidden, cfg.light_hidden), act,
                                             nn.Linear(cfg.light_hidden, cfg.light_dim))
        else:
            self.light_concat = False
            if hasattr(cfg, 'light_concat'):
                self.light_concat = cfg.light_concat

            self.aggregation_type = 'feat'
            self.spec_type = cfg.spec_type
            self.ref_ndotv_dim = 28
            if hasattr(cfg, 'embed_dim'):
                self.fusion_new = True
                self.embed_dim = cfg.embed_dim
                cfg.light_dim = self.embed_dim - context_dim - 3
                self.pos_embed = nn.Linear(self.ref_ndotv_dim, self.embed_dim, bias=True)
            else:
                self.fusion_new = False
                self.refdir_in = cfg.refdir_in
                self.embed_dim = cfg.light_dim
                if self.refdir_in:
                    self.rgb_embed = nn.Linear(3 + self.ref_ndotv_dim, cfg.light_dim - context_dim, bias=True)
                else:
                    if cfg.light_dim - context_dim == 3:
                        self.rgb_embed = DummyModule()
                    elif cfg.light_dim - context_dim > 3:
                        self.rgb_embed = nn.Linear(3, cfg.light_dim - context_dim, bias=True)
                    else:
                        raise Exception('sdf')

                if self.light_concat:
                    self.rgb_embed = DummyModule()
                    cfg.light_dim = cfg.light_dim // 2
                    self.embed_dim = cfg.light_dim + context_dim + 3

            light_net = []
            self.net_act = nn.ReLU(inplace=True)
            if cfg.spec_type == 'spec_in':
                in_dim = cfg_all.InLightEncoder.dim + self.ref_ndotv_dim
            elif cfg.spec_type == 'spec_feat':
                in_dim = cfg_all.SpecularNet.hidden
            self.net_act = nn.ReLU(inplace=True)
            self.spec_num_layers = 2
            light_net.append(nn.Linear(in_dim, cfg.light_hidden, bias=True))
            light_net.append(nn.Linear(cfg.light_hidden, cfg.light_hidden, bias=True))
            light_net.append(nn.Linear(cfg.light_hidden + 3, cfg.light_dim, bias=True))
            self.light_net = nn.ModuleList(light_net)
            init_seq(self.light_net, 'xavier_uniform', self.net_act)

        if cfg.type == 'attn':
            self.transformer = AngularTransformer(self.embed_dim, cfg.num_heads, cfg.dim_head, cfg.depth, self.out_dim,
                                                  cfg.out_hidden, cfg.dropout, cfg.attn_type, cfg.target_embed,
                                                  cfg.occ_th, cfg.mlp_ratio)
        elif cfg.type == 'mva':
            if not hasattr(cfg, 'attn_type'):
                cfg.attn_type = 'weight'
            self.transformer = MVANet(self.embed_dim, cfg.dim_head, cfg.depth, cfg.mlp_hidden, cfg.out_hidden,
                                      self.out_dim, cfg.attn_type)
        elif cfg.type == 'mva_attn':
            if not hasattr(cfg, 'attn_type'):
                cfg.attn_type = 'mask'
            self.transformer = MVANet_attn(self.embed_dim, cfg.dim_head, cfg.depth, cfg.mlp_hidden, cfg.out_hidden,
                                           self.out_dim, cfg.target_embed, cfg.attn_dim_head, cfg.attn_n_head,
                                           cfg.attn_type)
        elif cfg.type == 'rnn':
            self.transformer = RNN(self.embed_dim, cfg.hidden, cfg.num_layers, cfg.bias, self.out_dim, cfg.act, cfg.gru)

    def forward(self, rgb, shading, spec_feat, spec_in, proj_err, feat, view_dir, normal, IL):
        with torch.set_grad_enabled(self.is_train):
            bn, H, W, V, _ = rgb.shape
            if not feat.numel() == 0:
                feat = feat.permute(0, 2, 3, 1)[:, :, :, None]
                feat = feat.expand(-1, -1, -1, V, -1)

            if self.aggregation_type == 'SG':
                if self.DL_type == 'none':
                    # light_embed = torch.cat([normal.expand(-1, -1, -1, V, -1), view_dir], dim=-1)
                    light_embed = torch.tensor([], device=rgb.device)
                else:
                    normal = normal[:, :, :, None].expand(-1, -1, -1, V, self.SGNum, -1)
                    view_dir = view_dir.unsqueeze(-2).expand(-1, -1, -1, -1, self.SGNum, -1)
                    IL = IL.expand(-1, -1, -1, V, -1, -1)
                    IL_axis = IL[..., :3]
                    IL_intensity = torch.tan(np.pi / 2 * 0.999 * IL[..., 4:])
                    if self.DL_type == 'raw':
                        IL_sharp = torch.tan(np.pi / 2 * 0.999 * IL[..., 3:4])
                        pbr_feature_perSG = self.pbr_mlp(
                            torch.cat([IL_axis, IL_sharp, IL_intensity, normal, view_dir], dim=-1))
                        light_embed = torch.sum(pbr_feature_perSG, dim=-2)

                    elif self.DL_type == 'microfacet':
                        IL_sharp = 1 / (torch.tan(np.pi / 2 * 0.999 * IL[..., 3:4]) + 0.1)
                        h = IL_axis + view_dir
                        h = F.normalize(h, p=2.0, dim=-1)
                        NdotL = torch.sum(IL_axis * normal, dim=-1, keepdim=True)
                        NdotV = torch.sum(view_dir * normal, dim=-1, keepdim=True).expand(-1, -1, -1, -1, self.SGNum,
                                                                                          -1)
                        NdotH_2 = torch.pow(torch.sum(normal * h, dim=-1, keepdim=True), 2.0)
                        hdotV = torch.sum(h * view_dir, dim=-1, keepdim=True)
                        fresnel = 0.95 * torch.pow(2.0, (-5.55472 * hdotV - 6.98316) * hdotV) + 0.05
                        pbr_feature_perSG = self.pbr_mlp(
                            torch.cat([NdotL, IL_sharp, IL_intensity, NdotH_2, NdotV, fresnel], dim=-1))

                        valid_mask = ((NdotL * torch.sum(IL_intensity, dim=-1, keepdim=True)) > 0).float()
                        light_embed = torch.sum(pbr_feature_perSG * valid_mask, dim=-2)
                embed_in = torch.cat([rgb, light_embed, feat], dim=-1)

            else:
                ref_ndotv = spec_in[..., -self.ref_ndotv_dim:]
                if self.spec_type == 'spec_in':
                    light_embed = spec_in
                elif self.spec_type == 'spec_feat':
                    light_embed = spec_feat

                light_embed = self.light_net[0](light_embed)
                light_embed = self.net_act(light_embed)
                light_embed = self.light_net[1](light_embed)
                light_embed = self.net_act(light_embed)
                light_embed = torch.cat([light_embed, shading.expand(-1, -1, -1, V, -1)], dim=-1)
                light_embed = self.light_net[2](light_embed)
                if self.fusion_new:
                    embed_in = torch.cat([rgb, feat, light_embed], dim=-1)
                    pos_embed = self.pos_embed(ref_ndotv)
                    embed_in += pos_embed
                else:
                    if self.refdir_in:
                        rgb_embed = self.rgb_embed(torch.cat([rgb, ref_ndotv], dim=-1))
                    else:
                        rgb_embed = self.rgb_embed(rgb)

                    if self.light_concat:
                        embed_in = torch.cat([rgb_embed, feat, light_embed], dim=-1)
                    else:
                        embed_in = torch.cat([rgb_embed, feat], dim=-1) + light_embed

            return self.transformer(embed_in, proj_err).permute(0, 3, 1, 2)
