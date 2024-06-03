import torch
from utils import cv2fromtorch
from functools import reduce
from torch import nn
from utils import LSregress


def get_confidence_mask(mask, conf, thresholds):
    for thresh in thresholds:
        conf_mask = mask * (conf > thresh).float()
        if conf_mask.sum() >= 1000:
            if thresh != 0.9:
                print(thresh)
            break
        else:
            print(conf_mask.sum())
    return conf_mask


class RecLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.mode
        self.eps = 1e-6  # float32 only has 7 decimal digits precision
        losskey, loss_type, loss_w, lossmask = cfg.losskey, cfg.losstype, cfg.weight, cfg.lossmask

        self.si_max = 1000
        if hasattr(cfg, 'si_max'):
            self.si_max = cfg.si_max

        si_flag_list = [0, ] * len(losskey)
        if hasattr(cfg, 'si_ramp'):
            self.si_ramp = cfg.si_ramp
            # when si_ramp is used, the first loss key should be si-loss and the second one should be loss
            assert loss_type[0].startswith('si') and loss_type[0][2:] == loss_type[1] and losskey[0] == losskey[1]
            si_flag_list[0] = 1
            si_flag_list[1] = -1

        self.l1 = lambda a, b, m: torch.abs(a.contiguous() - b.contiguous()) * m
        self.l2 = lambda a, b, m: (a.contiguous() - b.contiguous()) ** 2.0 * m
        self.logl1 = lambda a, b, m: torch.abs(torch.log(a.contiguous() + 1.0) - torch.log(b.contiguous() + 1.0)) * m
        self.logl2 = lambda a, b, m: (torch.log(a.contiguous() + 1.0) - torch.log(b.contiguous() + 1.0)) ** 2 * m
        self.ang = lambda a, b, m: torch.acos(torch.clamp(
            torch.sum(a.contiguous() * b.contiguous(), dim=1, keepdim=True), min=-1 + self.eps,
            max=1 - self.eps)) * m
        self.beta = lambda vis, _, m: -vis * torch.log(vis.contiguous() + self.eps) * m
        self.nonzero = lambda x, _, m: -torch.log10(x.contiguous() + self.eps) * m
        # img is B V C H W
        self.wl2 = lambda a, b, m, w: torch.sum((a - b) ** 2 * w, dim=1) * m
        self.losslist = []
        for si_flag, k, t, w, m in zip(si_flag_list, losskey, loss_type, loss_w, lossmask):
            loss_func = getattr(self, t[2:]) if t.startswith('si') else getattr(self, t)
            self.losslist.append((si_flag, k, t, loss_func, w, m))

    def forward(self, data, pred, gt, mask, split, step):
        log = {}
        total_loss = 0.0
        for si_flag, k, t, loss, w, m in self.losslist:
            if k not in gt:
                gt[k] = None

            if m == 'nomask':
                curr_mask = 1.0
                curr_scalar = 1.0
            else:
                # b c h w
                if pred[k].ndim == 4 and (pred[k].shape[1] == 3 or pred[k].shape[1] == 1):
                    curr_mask = mask['default'][0]
                    curr_scalar = mask['default'][1]
                # b h w l c
                elif pred[k].ndim == 5 and pred[k].shape[3] >= 128:
                    curr_mask = mask['env'][0]
                    curr_scalar = mask['env'][1]
                # b h w v c
                elif pred[k].ndim == 5 and pred[k].shape[3] <= 9:
                    curr_mask = mask['all_view'][0]
                    curr_scalar = mask['all_view'][1]
                else:
                    raise Exception('mask error')

            if t.startswith('si'):
                pred['si_' + k], sc = LSregress(pred[k].detach(), gt[k], pred[k], curr_mask, self.si_max)
                log[f'{split}/{k}_{t}_sc'] = sc.mean().item()
                loss_current = torch.mean(loss(pred['si_' + k], gt[k], curr_mask) * curr_scalar)
            else:
                loss_current = torch.mean(loss(pred[k], gt[k], curr_mask) * curr_scalar)
            log[f'{split}/{k}_{t}_err'] = loss_current.item()
            # elif m == 'weighted_mask':
            #     loss_current = torch.mean(loss(pred[k], gt[k], curr_mask[..., 0, 0], pred['w']) * scalar)
            if si_flag == 1:
                assert t == 'silogl2'
                ramp = min(step / self.si_ramp, 1.0)
                w *= ramp
                log[f'{split}/si_rampup'] = ramp
            elif si_flag == -1:
                assert t == 'logl2'
                w *= (1 - ramp)
            total_loss += (loss_current * w)
        # mode MG
        if self.mode == 'MG' and split == 'test':
            scalar = mask['default'][1]
            mask = mask['default'][0]

            cds_l1 = torch.mean(self.l1(data['cds_dn'], gt['d'], mask) * scalar)

            d_gt_scale, sc = LSregress(pred['d'], gt['d'], pred['d'], mask, self.si_max)
            depth_l1_gt = torch.mean(self.l1(d_gt_scale, gt['d'], mask) * scalar)

            thresholds = [round(0.9 - 0.1 * i, 1) for i in range(10)]
            conf_mask = get_confidence_mask(mask, data['cds_conf'], thresholds)
            d_cds_scale, sc = LSregress(pred['d'], data['cds_dn'], pred['d'], conf_mask)
            depth_l1_cds = torch.mean(self.l1(d_cds_scale, gt['d'], mask) * scalar)

            log[f'{split}/cds_l1_err'] = cds_l1.item()
            log[f'{split}/depth_l1_gt_err'] = depth_l1_gt.item()
            log[f'{split}/depth_l1_cds_err'] = depth_l1_cds.item()
        # for i in torch.arange(0, 1, 0.1):
        #     conf_mask = mask * (data['cds_conf'] > i).float()
        #     depth_si, sc = LSregress(pred['d'].detach(), data['cds_dn'], pred['d'], conf_mask, self.si_max)
        #     loss_val = torch.mean(self.l2(depth_si, gt['d'], mask) * scalar)
        #     log[f'{split}/{i}_depth_err'] = loss_val.item()
        return total_loss, log
