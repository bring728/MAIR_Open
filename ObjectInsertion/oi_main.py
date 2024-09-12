from tqdm import tqdm
import argparse
from oi_utils import *
from oi_func import mair_object_rendering, zhu_object_rendering
import imageio
from utils import loadcfg
from datetime import datetime
# from Zhu.network.lightnet import LightNet, MODE_MIX


def oi_main(config, gpu, model_root, out_root):
    now = datetime.now()
    out_root = osp.join(out_root, now.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_root, exist_ok=True)
    os.system(f'cp {config} {out_root}')
    print(out_root)

    cfg = loadcfg(config)
    cfg_flag = cfg.insert_option
    cfg_objects = cfg.objects
    model_type = cfg.model_type
    model_root = model_root.format(model_type)

    cfg_flag.gpu = gpu
    cfg_flag.model_type = model_type
    if cfg_flag.fg_oversampling:
        cfg_flag.size_fg = [cfg_flag.size[0] * 2, cfg_flag.size[1] * 2]
    else:
        cfg_flag.size_fg = [cfg_flag.size[0], cfg_flag.size[1]]
    if cfg_flag.env_angular_large:
        cfg_flag.size[2] *= 2
        cfg_flag.size[3] *= 2

    if cfg_flag.model_type == 'Zhu':
        zhu_model = LightNet(cfg.model, MODE_MIX)
        checkpoint = torch.load('/media/vig-titan/Samsung_T5/Zhu_output/pretrained/lightnet/39.pth', map_location='cpu')
        zhu_model.load_state_dict(checkpoint)
        zhu_model.eval().to(gpu)
        for param in zhu_model.parameters():
            param.requires_grad = False

    for obj in tqdm(cfg_objects):
        mesh, self_occ, mesh_init_mat = mesh_init(obj['type'])
        cfg_flag.self_occ = self_occ
        cfg_flag.object_type = obj['type']
        tr_mat_prev = np.identity(4)
        scene = obj['scene']
        outdir = osp.join(out_root, scene)
        os.makedirs(outdir, exist_ok=True)
        scene = osp.join(model_root, scene)
        print(scene)

        common_dict, cam_dict, bg_dict = ready_OI(scene, cfg_flag)
        imageio.imwrite(osp.join(outdir, f'color_gt.png'), (common_dict['img'] * 255.0).astype(np.uint8))
        if cfg_flag.model_type == 'Zhu':
            common_dict['zhu_model'] = zhu_model

        bg_image = common_dict['img']
        for i, (coord, scale, trans) in enumerate(zip(obj['coord'], obj['scale'], obj['trans'])):
            outname = osp.join(outdir, f'{model_type}_{i}.png')
            # if osp.exists(outname):
            #     bg_image = np.array(Image.open(outname)) / 255.0
            #     continue
            tr_mat = get_tr(common_dict, coord, scale, trans)
            mesh.apply_transform(tr_mat @ np.linalg.inv(tr_mat_prev))
            tr_mat_prev = tr_mat
            bg_image = oi_run(mesh, bg_image, common_dict, cam_dict, bg_dict, cfg_flag)
            imageio.imwrite(outname, (bg_image * 255.0).astype(np.uint8))


def oi_run(mesh, bg_image, common_dict, cam_dict, bg_dict, flags):
    gpu, size, size_fg = flags['gpu'], flags['size'], flags['size_fg']
    common_dict['albedo'] = torch.tensor(flags.albedo).to(gpu)
    common_dict['rough'] = flags.rough
    model_type = flags.model_type
    self_occ, object_type = flags.self_occ, flags.object_type

    if model_type == 'MAIR' or model_type == 'MAIR++':
        index_ray, fg_radiance = mair_object_rendering(mesh, common_dict, cam_dict, gpu, flags, self_occ, object_type)

        pixel_ray = common_dict['pixels'][index_ray]
        mask = np.zeros([size_fg[1], size_fg[0], 3], dtype=np.float32)
        mask[pixel_ray[:, 1], pixel_ray[:, 0]] = 1.0
        fg = np.zeros([size_fg[1], size_fg[0], 3], dtype=np.float32)
        fg[pixel_ray[:, 1], pixel_ray[:, 0]] = fg_radiance.cpu().numpy()
        fg_ldr = hdr2ldr(fg)
        if flags.apply_LPF:
            mask = cv2.erode(mask, np.ones((3, 3), dtype=np.float32))
            fg_ldr = cv2.medianBlur(fg_ldr, 3)
        if flags.fg_oversampling:
            fg_ldr = cv2.resize(fg_ldr, (size_fg[0] // 2, size_fg[1] // 2), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (size_fg[0] // 2, size_fg[1] // 2), interpolation=cv2.INTER_AREA)
        if flags.only_fg:
            final_image = (1.0 - mask) * bg_image + mask * fg_ldr
            return final_image

        index_ray, bounced_radiance = mair_object_rendering(mesh, common_dict, bg_dict,
                                                            gpu, flags, False, 'shadow')
        bounced_envmaps = bg_dict['envmaps'].clone()
        bounced_envmaps[index_ray] = bounced_radiance * 0.1
        if flags.env_spatial_large:
            bounced_envmaps = bounced_envmaps.reshape([1, size[1], size[0], size[3], size[2], 3]).permute(
                [0, 5, 1, 2, 3, 4])
        else:
            bounced_envmaps = bounced_envmaps.reshape(
                [1, size[1] // 4, size[0] // 4, size[3], size[2], 3]).permute(
                [0, 5, 1, 2, 3, 4])
        bounced_shading = envmapToShading(common_dict['shadWeight'], bounced_envmaps)

        shade_factor = torch.clamp(bounced_shading / bg_dict['shading'], 0.0, 1.0)
        shade_factor = np.transpose(shade_factor[0].cpu().numpy(), [1, 2, 0])
        # shade_factor = shade_factor ** (1.0 / 2.2)
        shade_factor_ldr = cv2.GaussianBlur(cv2.dilate(shade_factor, np.ones((3, 3), dtype=np.float32)),
                                            (3, 3), 0)
        if not flags.env_spatial_large:
            shade_factor_ldr = cv2.resize(shade_factor_ldr, (size[0], size[1]),
                                          interpolation=cv2.INTER_LINEAR)

        bg_shaded = (bg_image * shade_factor_ldr)
        final_image = (1.0 - mask) * bg_shaded + mask * fg_ldr
        return final_image

    elif model_type == 'Zhu':
        final_image = zhu_object_rendering(mesh, bg_image, common_dict, cam_dict, gpu, size_fg, object_type)
        final_image = hdr2ldr(final_image)
        return final_image[0].permute([1, 2, 0]).cpu().numpy()


if __name__ == "__main__":
    # config = 'oi_coord_Zhu.yml'
    config = 'oi_coord_mair.yml'
    # config = 'oi_coord_mair++.yml'
    gpu = 0
    model_root = '/home/vig-titan-118/PycharmProjects/MAIR_Open/Examples/output/{}'
    out_root = '/home/vig-titan-118/PycharmProjects/MAIR_Open/Examples/object_insertion'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=config, help='stage mode')
    parser.add_argument('--gpu', type=int, default=gpu, help='stage mode')
    parser.add_argument('--model_root', type=str, default=model_root, help='stage mode')
    parser.add_argument('--out_root', type=str, default=out_root, help='stage mode')
    args = parser.parse_args()

    oi_main(args.config, args.gpu, args.model_root, args.out_root)
