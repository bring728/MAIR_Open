import torch

from utils import *
from tqdm import tqdm
import time
import datetime
from termcolor import colored
import torch.nn.functional as F
import wandb
import torch.distributed as dist
from cfgnode import CfgNode
import yaml
from einops import rearrange

colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']


def want_save_now(step):
    # if step <= 200:
    #     return step % 50 == 0
    if step <= 500:
        return step % 50 == 0
    elif step <= 3000:
        return step % 500 == 0
    elif step <= 10000:
        return step % 2000 == 0
    else:
        return step % 5000 == 0


def print_state(name, start_time, prev_time, max_it, global_step, print_period, total_loss, print_color=True):
    current_time = time.time()
    elapsed_sec = current_time - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    total_sec = elapsed_sec * max_it / global_step
    total_time = str(datetime.timedelta(seconds=total_sec)).split('.')[0]
    prog = global_step / max_it * 100
    sec_it = (current_time - prev_time) / print_period
    print_str = (f'{name} - step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, total:{total_time}, '
                 f'sec/it: {sec_it:.3f}, loss: {total_loss:.3f}')
    if print_color:
        print(colored(print_str, 'green'), end="\r")
    else:
        print(colored(print_str, 'yellow'))
    return current_time, sec_it


def extract_single_batch(data):
    out = {}
    for k in data:
        if data[k] is not None:
            if isinstance(data[k], tuple):
                out[k] = tuple([a[0] for a in data[k]])
            else:
                out[k] = data[k][0]
    return out


# data is BCHW
def vis_img(data, is_single=False, is_hdr=False, is_normal=False, normalize=False, is_env=False, size=None):
    if data.ndim > 3:
        data = data[:1]

    if is_env:
        data = torch.clamp(hdr2ldr(data), 0.0, 1.0)
        data = data / (torch.max(data) + TINY_NUMBER)
        _, r, c, l, ch = data.shape
        env_h = 8
        env_w = 16
        data = rearrange(data, 'b r c (h w) C -> b C (r h) (c w)', h=env_h, w=env_w)
    else:
        if is_hdr:
            data = torch.clamp(hdr2ldr(data), 0, 1.0)
        if is_single:
            data = data.expand([-1, 3, -1, -1])
        if is_normal:
            data = 0.5 * (data + 1)
        if normalize:
            data = data / data.max()

    if size is not None:
        data = F.interpolate(data, size=size)
    data = data[0]
    return data


def eval_model(model, loader, gpu, cfg, num_gpu, calculator, phase, step, is_training=False):
    flag = 'train' if is_training else 'test'
    model.switch_to_eval()
    scalars_to_log_all = {}
    scalars_to_log_mean = {}
    total_sample_num = 0
    with torch.no_grad():
        for i, val_data in enumerate(tqdm(loader)):
            names = val_data['name']
            bn = len(val_data['name'])
            val_data = tocuda(val_data, gpu, False)
            pred, gt, mask = model.forward(val_data, cfg, flag)
            _, scalars_to_log = calculator(val_data, pred, gt, mask, phase, step)
            for k in scalars_to_log:
                if k in scalars_to_log_all:
                    scalars_to_log_all[k].append(bn * scalars_to_log[k])
                else:
                    scalars_to_log_all[k] = [bn * scalars_to_log[k], ]
            total_sample_num += bn

    for val_k in scalars_to_log_all:
        scalars_to_log_mean[val_k] = sum(scalars_to_log_all[val_k]) / total_sample_num

    if num_gpu > 1:
        device = torch.device('cuda:{}'.format(gpu))
        tensor_list = [torch.zeros(len(scalars_to_log_mean), dtype=torch.float, device=device) for _ in range(num_gpu)]
        tensor_here = torch.zeros(len(scalars_to_log_mean), dtype=torch.float, device=device)

        scalars_to_log_ddp = sorted(scalars_to_log_mean.items())
        for i, scalar in enumerate(scalars_to_log_ddp):
            tensor_here[i] = scalar[1]

        dist.all_gather(tensor_list, tensor_here)
        tensor_mean = torch.mean(torch.stack(tensor_list, dim=0), dim=0)

        scalars_to_log_mean = {}
        for i, scalar in enumerate(scalars_to_log_ddp):
            scalars_to_log_mean[scalar[0]] = round(tensor_mean[i].item(), 6)

    model.switch_to_train()
    return scalars_to_log_mean, scalars_to_log_all


@torch.no_grad()
def output_model(model, loader, gpu, cfg, phase):
    if phase == 'custom_MG':
        thresholds = [round(0.9 - 0.1 * i, 1) for i in range(10)]

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

        for i, data in enumerate(tqdm(loader)):
            names = data['name']
            data = tocuda(data, gpu, False)
            pred, gt, masks = model.forward(data, cfg, phase)
            masks = masks['default'][0]
            for name, mask, depth, cds_depth, cds_conf in zip(names, masks, pred['d'], data['cds_depth'],
                                                              data['cds_conf']):
                conf_mask = get_confidence_mask(mask, cds_conf, thresholds)
                si_depth, sc = LSregress(depth, cds_depth, depth, conf_mask)
                # print(sc)
                saveBinary(name, si_depth.cpu().numpy()[0])

    else:
        for i, data in enumerate(tqdm(loader)):
            # names = data['name']
            data = tocuda(data, gpu, False)
            pred, gt = model.forward(data, cfg, 'output')
            outname = data['outname'][0]
            print(outname)

            np.save(osp.join(outname, 'cam_hwf.npy'), data['hwf'][0].cpu().numpy())
            np.savez_compressed(osp.join(outname, 'vsg'),
                                vsg=np.ascontiguousarray(pred['vsg'][0].float().data.cpu().numpy()),
                                scale=pred['vsg'][1])

            saveImage(osp.join(outname, 'albedo.png'), pred['a'], is_hdr=True)
            saveImage(osp.join(outname, 'normal.png'), 0.5 * (pred['n'] + 1), is_hdr=False)
            saveImage(osp.join(outname, 'rough.png'), pred['r'], is_hdr=False, is_single=True)
            saveImage(osp.join(outname, 'depth.png'), pred['d'], is_hdr=False, is_single=True)
            saveImage(osp.join(outname, 'color_gt.png'), data['i'], is_hdr=True)
            saveImage(osp.join(outname, 'color_pred.png'), pred['rgb_vsg'], is_hdr=True)
            saveImage(osp.join(outname, 'diffScaled.png'), pred['diff_vsg'], is_hdr=True)
            saveImage(osp.join(outname, 'specScaled.png'), pred['spec_vsg'], is_hdr=True)
            envmapsPredImage = pred['e_vsg'][0].float().data.cpu().numpy()
            envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0])
            np.savez_compressed(osp.join(outname, 'env'),
                                env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
            writeEnvToFile(pred['e_vsg'].float(), 0, osp.join(outname, 'envmaps.png'), nrows=12, ncols=8)

            if cfg.version == 'MAIR++':
                saveImage(osp.join(outname, 'albedo_single.png'), pred['a_s'], is_hdr=True)
                saveImage(osp.join(outname, 'rough_single.png'), pred['r_s'], is_hdr=False, is_single=True)
                saveImage(osp.join(outname, 'color_pred_ilr.png'), pred['rgb_ilr'], is_hdr=True)
                saveImage(osp.join(outname, 'diffScaled_ilr.png'), pred['diff_ilr'], is_hdr=True)
                saveImage(osp.join(outname, 'specScaled_ilr.png'), pred['spec_ilr'], is_hdr=True)

                np.savez_compressed(osp.join(outname, 'ilr'),
                                    ilr=np.ascontiguousarray(pred['ilr'][0].float().data.cpu().numpy()))
                envmapsPredImage = pred['e_ilr'][0].float().data.cpu().numpy()
                envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0])
                np.savez_compressed(osp.join(outname, 'env_ilr'),
                                    env=np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1]))
                writeEnvToFile(pred['e_ilr'].float(), 0, osp.join(outname, 'envmaps_ilr.png'), nrows=12, ncols=8)


@torch.no_grad()
def mat_edit(model, loader, gpu, cfg, debug=False):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    # pip install git+https://github.com/facebookresearch/segment-anything.git
    from ObjectInsertion.Zhu.network.lightnet import LightNet, MODE_MIX
    from ObjectInsertion.Zhu.render.layer import RenderLayerClip

    sam = sam_model_registry["default"](checkpoint="/home/vig-titan/pretrained/sam_vit_h_4b8939.pth")
    device = "cuda"
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # a_edit = 'inversion'
    a_edit = 'original'
    # r_edit = 'specular'

    # weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # seg_model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False).to(gpu)
    # seg_model = seg_model.eval()
    click_handler = MouseHandler()
    for i, data in enumerate(tqdm(loader)):
        name = data['outname']
        data = tocuda(data, gpu, False)
        img_cv = cv2fromtorch(hdr2ldr(data['i']))
        if debug:
            key = ord('o')
        else:
            img_cv_show = cv2.resize(img_cv[..., ::-1], (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('rgb', img_cv_show)
            cv2.setMouseCallback('rgb', click_handler.handle_click)
            key = cv2.waitKey()
        if key == ord('q'):
            break
        elif key == ord('z'):
            with open('mat_edit_pass.txt', 'a') as f_error:
                f_error.write(osp.basename(name[0]) + '\n')
            continue
        else:
            clicked_point = click_handler.get_clicked_point()
            clicked_point = (clicked_point[0] // 2, clicked_point[1] // 2)
            masks = mask_generator.generate(img_cv)
            bool_masks = [m['segmentation'] for m in masks if m['segmentation'][clicked_point[1], clicked_point[0]]]
            masked_images = [(m, m.astype(np.uint8)[..., None] * img_cv) for m in bool_masks]
            # output = seg_model(hdr2ldr(data['i']))
            # masks = output[0]['masks']
            # proba_threshold = 0.5
            # bool_masks = masks > proba_threshold
            # bool_masks = [m for m in bool_masks if m[0, clicked_point[1], clicked_point[0]]]
            # masked_images = [(m, m * hdr2ldr(data['i'])) for m in bool_masks]
            print(len(masked_images))
            for mask, masked_image in masked_images:
                # if debug:
                #   key = ord('o')
                # else:
                #     cv2.imshow('rgb', cv2.hconcat([img_cv, masked_image])[..., ::-1])
                #     key = cv2.waitKey()

                # if key == ord('o'):
                mask = torch.from_numpy(mask).to(device)[None, None]

                r_edit = 0.1
                a_scale = torch.tensor([1.0, 0.3, 0.3], dtype=torch.float32, device=device)[None, :, None, None]
                r = torch.ones_like(data['i'])[:, :1] * r_edit

                func1 = lambda tmp: tmp[:, [2, 0, 1]]
                func2 = lambda tmp: tmp[:, [1, 2, 0]]
                func3 = lambda tmp: tmp * a_scale
                func_list = [func1, func2, func3]

                rgb_output_list = []
                for func in func_list:
                    rgb_output = [data['i'], ]
                    for model_name in cfg.models:
                        a = func(data[f'{model_name}_a'])
                        if model_name == 'MAIR++':
                            rgb_pred = model.mat_edit_func(data[f'{model_name}_cam'], data['i'],
                                                           data[f'{model_name}_n'],
                                                           a, r, data[f'{model_name}_ilr'], True)
                        else:
                            rgb_pred = model.mat_edit_func(data[f'{model_name}_cam'], data['i'],
                                                           data[f'{model_name}_n'],
                                                           a, r, data[f'{model_name}_e'], False)

                        rgb_edited = ~mask * data['i'] + mask * rgb_pred
                        rgb_output.append(rgb_edited)
                    rgb_output_list.append(hdr2ldr(torch.cat(rgb_output, dim=-1))[0])
                rgb = torch.cat(rgb_output_list, dim=1)
                rgb = cv2fromtorch(rgb)[..., ::-1]
                cv2.imwrite(name[0] + '.png', rgb)
                with open('mat_edit_pass.txt', 'a') as f_error:
                    f_error.write(osp.basename(name[0]) + '\n')
                break
                # cv2.imshow('rgb', cv2.resize(rgb, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR))
                # key = cv2.waitKey()
                # if key == ord('z'):
                #     continue


def mat_edit_single(model, gpu, models_name, config):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    # pip install git+https://github.com/facebookresearch/segment-anything.git
    from ObjectInsertion.Zhu.network.lightnet import LightNet, MODE_MIX
    from ObjectInsertion.Zhu.render.layer import RenderLayerClip

    generate_video = True

    sam = sam_model_registry["default"](checkpoint='/home/vig-titan-118/Downloads/sam_vit_h_4b8939.pth').to(gpu)
    # sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth").to(gpu)
    mask_generator = SamAutomaticMaskGenerator(sam)

    cfg = loadcfg(f'config/{config}.yml')
    name = cfg.scene_name
    outname = name.format('mat_edit')
    size = (320, 240)
    data = {}
    if 'Zhu' in models_name:
        zhu_cfg = loadcfg('ObjectInsertion/Zhu/config_zhu_light.yml')
        zhu_model = LightNet(zhu_cfg.model, MODE_MIX)
        checkpoint = torch.load('/media/vig-titan-118/Samsung_T5/Zhu_output/pretrained/lightnet/39.pth', map_location='cpu')
        # checkpoint = torch.load('E:/Zhu_output/pretrained/lightnet/39.pth', map_location='cpu')
        zhu_model.load_state_dict(checkpoint)
        zhu_model.eval().to(gpu)
        for param in zhu_model.parameters():
            param.requires_grad = False

    for model_str in models_name:
        scene = name.format(model_str)
        if not osp.exists(osp.join(scene, 'cam_hwf.npy')):
            h, w, f = np.load(osp.join(name.format('MAIR++'), 'cam_hwf.npy'))
        else:
            h, w, f = np.load(osp.join(scene, 'cam_hwf.npy'))
        intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float).astype(np.float32)
        data[f'{model_str}_cam'] = intrinsic
        n_name = osp.join(scene, 'normal.png')
        d_name = osp.join(scene, 'depth.png')
        a_name = osp.join(scene, 'albedo.png')
        r_name = osp.join(scene, 'rough.png')
        n = loadImage(n_name, 'n', size, normalize=True)
        d = loadImage(d_name, 'r', size, normalize=True)
        a = loadImage(a_name, 'a', size, normalize=True)
        r = loadImage(r_name, 'r', size, normalize=True)
        data[f'{model_str}_n'] = n.transpose([2, 0, 1])
        data[f'{model_str}_d'] = d[None]
        data[f'{model_str}_a'] = a.transpose([2, 0, 1])
        data[f'{model_str}_r'] = r[None]

        if model_str == 'MAIR++':
            env_name = osp.join(scene, 'ilr.npz')
            e = np.load(env_name)['ilr'].copy()
            data[f'{model_str}_ilr'] = e
        else:
            env_name = osp.join(scene, 'env.npz')
            e = np.load(env_name)['env'][..., ::-1].copy().transpose([4, 0, 1, 2, 3])
            data[f'{model_str}_e'] = e

    name = osp.join(name.format('MAIR++'), 'color_gt.png')
    im = cv2.imread(name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[..., ::-1]
    im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    data['i'] = ldr2hdr(im).transpose([2, 0, 1])

    for k in data.keys():
        data[k] = torch.from_numpy(data[k])[None]
    data = tocuda(data, gpu, False)
    click_handler = MouseHandler()
    img_cv = cv2fromtorch(hdr2ldr(data['i']))
    img_cv_show = cv2.resize(img_cv, (640, 480), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('rgb', img_cv_show)
    cv2.setMouseCallback('rgb', click_handler.handle_click)
    cv2.waitKey()
    cv2.destroyWindow('rgb')
    clicked_points = click_handler.get_clicked_point()
    clicked_points = [(c[0] // 2, c[1] // 2) for c in clicked_points]

    masks = mask_generator.generate(img_cv)
    bool_masks = []
    for clicked_point in clicked_points:
        bool_masks.append([m['segmentation'] for m in masks if m['segmentation'][clicked_point[1], clicked_point[0]]])

    mask_all = np.zeros_like(bool_masks[0][0])
    for mask in bool_masks:
        mask_all = mask_all | mask[0]
    mask = torch.from_numpy(mask_all).to(gpu)[None, None]
    idx = 0
    while True:
        inv_mask_float = (~mask).float()
        idx += 1
        cfg = loadcfg(f'config/{config}.yml')

        a_scale = torch.tensor(cfg.a_scale, dtype=torch.float32).to(gpu)[None, :, None, None]
        a_offset = torch.tensor(cfg.a_offset, dtype=torch.float32).to(gpu)[None, :, None, None]

        # for video generation
        if generate_video:
            interp_num = 100
        else:
            interp_num = 1
        rgb_all_frame = []
        for j in range(interp_num):
            rgb_output = [data['i'], ]
            for model_name in cfg.models:
                r_edit = cfg.r_edit
                if r_edit == 'None':
                    r_new = data[f'{model_name}_r']
                else:
                    r_new = torch.ones_like(data['i'])[:, :1] * r_edit

                a_new = data[f'{model_name}_a'][:, cfg.a_permute]
                a_new = a_new * a_scale
                a_new = a_new + a_offset
                a_new = torch.clamp(a_new, 0.0, 1.0)

                if generate_video:
                    a_new = (j / interp_num) * a_new + (1 - (j / interp_num)) * data[f'{model_name}_a']
                    r_new = (j / interp_num) * r_new + (1 - (j / interp_num)) * data[f'{model_name}_r']

                a_new = data[f'{model_name}_a'] * inv_mask_float + a_new * mask.float()
                r_new = data[f'{model_name}_r'] * inv_mask_float + r_new * mask.float()

                if model_name == 'MAIR++':
                    rgb_pred = model.mat_edit_func(inv_mask_float, data[f'{model_name}_cam'], data['i'],
                                                   data[f'{model_name}_n'],
                                                   a_new, r_new, data[f'{model_name}_ilr'], True)
                elif model_name == 'Zhu':
                    fov = math.degrees(
                        2 * math.atan(data[f'{model_name}_cam'][0, 0, 2] / data[f'{model_name}_cam'][0, 0, 0]))
                    zhu_renderer = RenderLayerClip(fov=fov, spp=zhu_cfg.eval.spp, imWidth=size[0], imHeight=size[1],
                                                   chunk=zhu_cfg.render.chunk,
                                                   uncertainty_boundary=zhu_cfg.render.uncertainty_boundary).to(gpu)
                    rgb_pred = zhu_mat_edit(zhu_renderer, zhu_model, mask, fov, data['i'].clone(),
                                            data[f'{model_name}_n'],
                                            data[f'{model_name}_d'], a_new, r_new, size)
                else:
                    rgb_pred = model.mat_edit_func(inv_mask_float, data[f'{model_name}_cam'], data['i'],
                                                   data[f'{model_name}_n'],
                                                   a_new, r_new, data[f'{model_name}_e'], False)

                rgb_edited = ~mask * data['i'] + mask * rgb_pred
                rgb_output.append(rgb_edited)

            rgb_all_frame.append(rgb_output)

        # for video generation
        if generate_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 정의 (예: 'XVID', 'MJPG', 'X264')
            fps = 30
            # out = cv2.VideoWriter('video_orange.avi', fourcc, fps, (size[0] * 5, size[1]))
            out = cv2.VideoWriter('video_orange.avi', fourcc, fps, (size[0] * 5, size[1]))
            # out_li = cv2.VideoWriter('Li.avi', fourcc, fps, (size[0], size[1]))
            # out_zhu = cv2.VideoWriter('Zhu.avi', fourcc, fps, (size[0], size[1]))
            # out_mair = cv2.VideoWriter('MAIR.avi', fourcc, fps, (size[0], size[1]))
            # out_mairplus = cv2.VideoWriter('MAIR++.avi', fourcc, fps, (size[0], size[1]))
            for rgb_all in rgb_all_frame:
                rgb = hdr2ldr(torch.cat(rgb_all, dim=-1))[0]
                rgb = cv2fromtorch(rgb)
                out.write(rgb)
                # rgb = hdr2ldr(rgb_all[1])[0]
                # rgb = cv2fromtorch(rgb)
                # out_li.write(rgb)
                # rgb = hdr2ldr(rgb_all[2])[0]
                # rgb = cv2fromtorch(rgb)
                # out_mair.write(rgb)
                # rgb = hdr2ldr(rgb_all[3])[0]
                # rgb = cv2fromtorch(rgb)
                # out_mairplus.write(rgb)
            for rgb_all in reversed(rgb_all_frame):
                rgb = hdr2ldr(torch.cat(rgb_all, dim=-1))[0]
                rgb = cv2fromtorch(rgb)
                out.write(rgb)
                # rgb = hdr2ldr(rgb_all[1])[0]
                # rgb = cv2fromtorch(rgb)
                # out_li.write(rgb)
                # rgb = hdr2ldr(rgb_all[2])[0]
                # rgb = cv2fromtorch(rgb)
                # out_mair.write(rgb)
                # rgb = hdr2ldr(rgb_all[3])[0]
                # rgb = cv2fromtorch(rgb)
                # out_mairplus.write(rgb)
            out.release()
            # out_li.release()
            # out_mair.release()
            # out_mairplus.release()
            break
        else:
            rgb = hdr2ldr(torch.cat(rgb_all_frame[0], dim=-1))[0]
            rgb = cv2fromtorch(rgb)
            cv2.imshow('rgb', cv2.resize(rgb, (0, 0), fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR))
            key = cv2.waitKey()
            cv2.destroyAllWindows()
            if key == ord('s'):
                # cv2.imwrite(f'{outname}_{idx}.png', rgb)
                cv2.imwrite(f'{outname}_{idx}_org.png', rgb[:, :size[0]])
                for i, model_str in enumerate(models_name):
                    cv2.imwrite(f'{outname}_{idx}_{model_str}.png', rgb[:, size[0] * (i + 1):size[0] * (i + 2)])

                with open(f'{outname}_{idx}.txt', 'w') as f:
                    f.write(osp.basename(outname))
                    f.write(f"\na_offset : ")
                    for number in cfg.a_offset:
                        f.write(f"{number} ")
                    f.write(f"\na_permute : ")
                    for number in cfg.a_permute:
                        f.write(f"{number} ")
                    f.write(f"\na_scale : ")
                    for number in cfg.a_scale:
                        f.write(f"{number} ")
                    f.write(f"\nr_edit : ")
                    f.write(str(cfg.r_edit))
            elif key == ord('q'):
                break


def zhu_mat_edit(zhu_renderer, zhu_model, mask, fov, im, n, d, a, r, size_fg):
    from ObjectInsertion.Zhu.zhu_utils import depth_to_vpos

    mask = mask[0, 0]
    d = d[0, 0]
    box_all, boxes = bounding_box(mask.cpu().numpy(), chunk=30000)
    m = torch.zeros_like(r)
    vpos = depth_to_vpos(d, fov, True)
    vpos = vpos.unsqueeze_(0).to(d.device)
    diff_list = []
    spec_list = []
    for box in boxes:
        r_diff, r_spec, _ = zhu_renderer(box, zhu_model, im, a, n, r, m, vpos)
        diff_list.append(r_diff)
        spec_list.append(r_spec)
    radiance = torch.cat(diff_list, dim=2) + torch.cat(spec_list, dim=2)
    radiance = torch.where(torch.isfinite(radiance), radiance,
                           torch.zeros_like(radiance))[0]
    radiance = torch.clamp(radiance, 0, 1.0)

    min_cx = max(box_all[0], 0)
    max_cx = min(box_all[1], size_fg[0])
    min_cy = max(box_all[2], 0)
    max_cy = min(box_all[3], size_fg[1])

    im_clip = im[0][:, min_cy:max_cy, min_cx:max_cx]
    mask_clip = mask[min_cy:max_cy, min_cx:max_cx][None].float()
    im[0, :, min_cy:max_cy, min_cx:max_cx] = im_clip * (1 - mask_clip) + radiance * mask_clip
    return im


def calculate_statistics(loader, gpu):
    from model import compute_projection
    u, v = np.meshgrid(np.arange(320), np.arange(240), indexing='xy')
    u = u.astype(dtype=np.float32) + 0.5  # add half pixel
    v = v.astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    pixels = torch.from_numpy(pixels)
    pixels = pixels.to(gpu)[None, :, :, :, None]

    def load(name, gpu, is_ldr=False):
        img = cv2.imread(name) / 255.0
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        if is_ldr:
            img = img ** 2.2
        return torch.from_numpy(img.astype(np.float32)).to(gpu)

    for method in ['MAIR', 'MAIR++', 'Li21', 'Zhu']:
        variance_albedo_list = []
        variance_rough_list = []
        for i, data in enumerate(tqdm(loader)):
            data = tocuda(data, gpu, False)
            scene = data['outname'][0]
            src_idx = data['src_idx']
            albedos = []
            roughs = []
            for i in src_idx:
                name = scene.replace('MAIR++', f'{method}')[:-4] + f'_{int(i[0]):03d}'
                albedos.append(load(osp.join(name, 'albedo.png'), gpu, is_ldr=True))
                roughs.append(load(osp.join(name, 'rough.png'), gpu))
            albedos = torch.stack(albedos).permute([0, 3, 1, 2])[None]
            albedos, viewdir, proj_err, _ = compute_projection(pixels, data['all_cam'], data['c2w'],
                                                               data['all_depth'], albedos)
            albedos = albedos[0]
            mask = (torch.abs(proj_err[..., 1:, :]) < 0.05)[0]
            tmp = torch.ones([120, 160, 1, 1], dtype=torch.bool, device=mask.device)
            mask = torch.cat([tmp, mask], dim=2)
            sum_mask = mask.sum(dim=2, keepdim=True)  # (1, H, W)

            masked_images = albedos * mask
            mean_map = masked_images.sum(dim=2, keepdim=True) / sum_mask  # (1, H, W)
            squared_diff = ((albedos - mean_map) ** 2) * mask  # (9, H, W)
            variance_map = squared_diff.sum(dim=2, keepdim=True) / sum_mask  # (1, H, W)
            variance_albedo_list.append(variance_map.mean().item())

            roughs = torch.stack(roughs).permute([0, 3, 1, 2])[None]
            roughs, viewdir, proj_err, _ = compute_projection(pixels, data['all_cam'], data['c2w'],
                                                              data['all_depth'], roughs)
            roughs = roughs[0]
            mask = (torch.abs(proj_err[..., 1:, :]) < 0.05)[0]
            tmp = torch.ones([120, 160, 1, 1], dtype=torch.bool, device=mask.device)
            mask = torch.cat([tmp, mask], dim=2)
            sum_mask = mask.sum(dim=2, keepdim=True)  # (1, H, W)

            masked_images = roughs * mask
            mean_map = masked_images.sum(dim=2, keepdim=True) / sum_mask  # (1, H, W)
            squared_diff = ((roughs - mean_map) ** 2) * mask  # (9, H, W)
            variance_map = squared_diff.sum(dim=2, keepdim=True) / sum_mask  # (1, H, W)
            variance_rough_list.append(variance_map.mean().item())

        albedo_var = sum(variance_albedo_list) / len(variance_albedo_list)
        rough_var = sum(variance_rough_list) / len(variance_rough_list)
        print(albedo_var, rough_var, method)
    print()
