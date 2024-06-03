from utils import *
from torch import nn


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

        if not hasattr(cfg, 'attn_type'):
            cfg.attn_type = 'weight'
        self.transformer = MVANet(self.embed_dim, cfg.dim_head, cfg.depth, cfg.mlp_hidden, cfg.out_hidden,
                                  self.out_dim, cfg.attn_type)

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

            return self.transformer(embed_in, proj_err).permute(0, 3, 1, 2)
