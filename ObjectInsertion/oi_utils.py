import glob
import math
import scipy.io as io
import os
import trimesh
import trimesh.transformations as tr
import numpy as np
import cv2
import os.path as osp
import torch
from PIL import Image
import torch.nn.functional as F
# from Zhu.render.layer import RenderLayerClip
# from Zhu.zhu_utils import depth_to_vpos
from einops import rearrange
import sys

sys.path.append('../')
from model import pbr, envmapfromVSG
from utils import loadImage, get_N2C, cv2fromtorch, envmapToShading, hdr2ldr, ldr2hdr, bounding_box

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
point_list = []


# mouse callback function
def line_drawing(event, x, y, flags, img):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
        point_list.append(np.array([x, y]))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)
            point_list.append(np.array([x, y]))
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)
        point_list.append(np.array([x, y]))


def path_from_mouse(img, dist_th=6.0):
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw', line_drawing, img)
    while True:
        cv2.imshow('test draw', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    path = []
    for i in range(len(point_list) - 1):
        path.append(np.round(np.linspace(point_list[i], point_list[i + 1], 10)))

    path = np.concatenate(path, axis=0)
    index = sorted(np.unique(path, axis=0, return_index=True)[1])
    path_list = path[index].astype(np.int32)
    num_path = len(path_list)
    idx_list = [0, ]
    dist_before = 0.0
    for i in range(num_path - 1):
        dist_curr = np.linalg.norm(path_list[i] - path_list[i + 1])
        distance = dist_curr + dist_before
        if distance > dist_th:
            idx_list.append(i + 1)
            dist_before = 0.0
        else:
            dist_before += dist_curr
    return path_list[idx_list]


def ready_OI(scene, flags):
    xy_offset = 1.3
    gpu, size = flags['gpu'], flags['size']
    env_spatial_large, env_angular_large = flags['env_spatial_large'], flags['env_angular_large']
    fg_oversampling = flags['fg_oversampling']

    name = osp.join(scene.format(flags.model_type), '{}')
    bg_org = np.array(Image.open(name.format('color_gt.png'))) / 255.0
    h, w, _ = bg_org.shape

    bg_normal_rub = loadImage(name.format('normal.png'), 'n', normalize=True)
    bg_normal_rub = torch.from_numpy(bg_normal_rub[None]).to(gpu)
    bg_normal_org = bg_normal_rub[0].permute(2, 0, 1).cpu().numpy()

    # for the same scale, we use MAIR++ depth map.
    # bg_depth = loadImage(name.replace(flags.model_type, 'MAIR++').format('depth.png'), 'r')
    bg_depth = loadImage(name.format('depth.png'), 'r')
    bg_depth_norm = torch.from_numpy(bg_depth).to(gpu)[None]
    h_cam, w_cam, f_cam = np.load(name.format('cam_hwf.npy'))
    f = f_cam * (bg_depth_norm.shape[1] / h_cam)
    fov = math.degrees(2 * math.atan(w_cam / (2 * f_cam)))

    (pixels, norm_coord_rub, cam_viewdir_rub, cam_origins, cam_ray_rub, ls, up,
     rub_to_rdf, envWeight_ndotl, bb, shadWeight, ndotl) = get_cam_vector(gpu, size, f, f, fg_oversampling, xy_offset)

    bg_coord_rub = norm_coord_rub.reshape([size[1], size[0], 3]).permute([2, 0, 1]) * bg_depth_norm
    bg_coord_rub_org = bg_coord_rub.cpu().numpy()

    common_dict = {}
    if flags.model_type == 'MAIR' or flags.model_type == 'MAIR++':
        npz = np.load(name.format('vsg.npz'))
        VSG, light_scale = npz['vsg'], npz['scale'].item()
        VSG = torch.from_numpy(VSG).to(gpu)
        common_dict = {'VSG': VSG, 'light_scale': light_scale}

    elif flags.model_type == 'Zhu':
        albedo = loadImage(name.format('albedo.png'), 'a', normalize=True)
        common_dict['zhu_a'] = torch.from_numpy(albedo[None]).to(gpu).permute([0, 3, 1, 2])
        rough = loadImage(name.format('rough.png'), 'r', normalize=True)
        common_dict['zhu_r'] = torch.from_numpy(rough[None, None]).to(gpu)
        common_dict['zhu_m'] = torch.zeros_like(common_dict['zhu_r'])
        common_dict['zhu_n'] = bg_normal_rub.permute([0, 3, 1, 2])
        common_dict['zhu_d'] = bg_depth_norm[0]

        renderer = RenderLayerClip(fov=fov, spp=flags.spp, imWidth=size[0], imHeight=size[1], chunk=flags.chunk,
                                   uncertainty_boundary=flags.uncertainty_boundary)
        renderer.to(gpu)
        common_dict['zhu_renderer'] = renderer
        common_dict['fov'] = fov

    if not env_spatial_large:
        bg_coord_rub = F.adaptive_avg_pool2d(bg_coord_rub[None], (h // 4, w // 4))[0]
        bg_normal_rub = F.adaptive_avg_pool2d(bg_normal_rub.permute([0, 3, 1, 2]), (h // 4, w // 4)).permute(
            [0, 2, 3, 1])
        bg_normal_rub = F.normalize(bg_normal_rub, p=2.0, dim=-1)

    N2C = get_N2C(bg_normal_rub, up)
    bg_ray_rub = (N2C.unsqueeze(-3) @ ls[None, None, ..., None]).squeeze(-1)
    bg_origins = bg_coord_rub[None].permute([0, 2, 3, 1])
    bg_origins = bg_origins[:, :, :, None, :].expand_as(bg_ray_rub).reshape([-1, 3]).cpu().numpy()
    bg_ray_rub = bg_ray_rub.reshape([-1, 3])
    bg_viewdir_rub = -bg_ray_rub
    bg_ray_rub = bg_ray_rub.cpu().numpy()

    bg_envmaps = np.load(name.format('env.npz'))['env'][..., ::-1].copy().transpose([4, 0, 1, 2, 3])
    bg_envmaps = torch.from_numpy(bg_envmaps[None]).to(gpu)

    if env_angular_large:
        b_, c_, H_, W_, h_, w_ = bg_envmaps.shape
        bg_envmaps = rearrange(bg_envmaps, 'b c H W h w -> (b H W) c h w')
        bg_envmaps = F.interpolate(bg_envmaps, [size[3], size[2]], mode='bilinear')
        bg_envmaps = rearrange(bg_envmaps, '(b H W) c h w -> b c H W h w', b=1, H=H_, W=W_)

    if env_spatial_large:
        bg_envmaps = bg_envmaps.permute([0, 1, 4, 5, 2, 3]).reshape([1, 3 * size[2] * size[3], h // 4, w // 4])
        bg_envmaps = F.interpolate(bg_envmaps, [h, w], mode='bilinear').reshape([1, 3, 8, 16, h, w]).permute(
            [0, 1, 4, 5, 2, 3])

    bg_shading = envmapToShading(shadWeight, bg_envmaps)
    bg_envmaps = bg_envmaps.permute([0, 2, 3, 4, 5, 1]).reshape([-1, 3])

    cam_dict = {'origin': cam_origins, 'ray': cam_ray_rub, 'viewdir': cam_viewdir_rub}
    bg_dict = {'origin': bg_origins, 'ray': bg_ray_rub, 'viewdir': bg_viewdir_rub, 'ldr': bg_org, 'shading': bg_shading,
               'envmaps': bg_envmaps}

    r_res = 64
    r_dist = (np.arange(r_res) + 0.5) / r_res
    r_dist = torch.from_numpy(r_dist.astype(np.float32)).to(gpu)[None, None, None, None]
    common_dict.update({'pixels': pixels, 'ls': ls, 'up': up, 'img': bg_org,
                        'rub_to_rdf': rub_to_rdf, 'envWeight_ndotl': envWeight_ndotl, 'ndotl': ndotl,
                        'bb': bb, 'r_dist': r_dist, 'shadWeight': shadWeight,
                        'size': size, 'coord': bg_coord_rub_org, 'normal': bg_normal_org})
    return common_dict, cam_dict, bg_dict


def get_cam_vector(gpu, size, fx, fy, oversampling, xy_offset):
    u, v = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='xy')
    u = u.astype(dtype=np.float32) + 0.5  # add half pixel
    v = v.astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    pixels = torch.from_numpy(pixels)
    pixels_org = pixels.to(gpu)[None, :, :, :3, None]

    fov_x = size[0] / (2.0 * fx)
    fov_y = size[1] / (2.0 * fy)
    bb = np.array([xy_offset * fov_x, xy_offset * fov_y, 1.05], dtype=np.float32)[None]
    bb = torch.from_numpy(bb).to(gpu)

    # azimuth, elevation
    Az_org = ((np.arange(size[2]) + 0.5) / size[2] - 0.5) * 2 * np.pi  # -pi ~ pi, theta
    El_org = ((np.arange(size[3]) + 0.5) / size[3]) * np.pi / 2.0  # 0 ~ pi/2, phi
    Az, El = np.meshgrid(Az_org, El_org)
    Az_flat = Az.reshape(-1, 1)
    El_flat = El.reshape(-1, 1)
    lx_dir = np.sin(El_flat) * np.cos(Az_flat)
    ly_dir = np.sin(El_flat) * np.sin(Az_flat)
    lz_dir = np.cos(El_flat)
    ls = torch.from_numpy(np.concatenate((lx_dir, ly_dir, lz_dir), axis=-1).astype(np.float32))
    ls = ls.to(gpu)[None]

    shadWeight = np.cos(El) * np.sin(El)
    shadWeight = shadWeight[None, None, None, None]
    shadWeight = torch.from_numpy(shadWeight.astype(np.float32))
    shadWeight = shadWeight.to(gpu)

    up = torch.from_numpy(np.array([0, 1.0, 0], dtype=np.float32)).to(gpu)
    rub_to_rdf = torch.from_numpy(np.array([1.0, -1.0, -1.0], dtype=np.float32)).to(gpu)

    envWeight = np.sin(El_flat) * np.pi * np.pi / size[2] / size[3]
    envWeight = torch.from_numpy(envWeight.astype(np.float32))[None, None, None, None].to(gpu)
    ndotl = torch.sum(ls[0] * torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=gpu), dim=1)
    ndotl = torch.clamp(ndotl, 0.0, 1.0)[None, None, None, None].unsqueeze(-1)
    envWeight_ndotl = ndotl * envWeight

    intrinsic = torch.from_numpy(
        np.array([[fx, 0, size[0] / 2], [0, fy, size[1] / 2], [0, 0, 1]], dtype=np.float32)).to(gpu)
    # rdf to rub ! ( rub_to_rdf = rdf_to_rub )
    norm_coord_rub_org = (torch.inverse(intrinsic) @ pixels_org)[..., 0].reshape([-1, 3]) * rub_to_rdf[None]

    if oversampling:
        u, v = np.meshgrid(np.arange(size[0] * 2), np.arange(size[1] * 2), indexing='xy')
        u = u.astype(dtype=np.float32) + 0.5  # add half pixel
        v = v.astype(dtype=np.float32) + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
        pixels = torch.from_numpy(pixels)
        pixels_large = pixels.to(gpu)[None, :, :, :3, None]
        intrinsic[:2] *= 2
        norm_coord_rub_large = (torch.inverse(intrinsic) @ pixels_large)[..., 0].reshape([-1, 3]) * rub_to_rdf[None]
        norm_coord = norm_coord_rub_large
        pixels = pixels_large
        origins = np.zeros([norm_coord.shape[0], 3], dtype=float)
    else:
        norm_coord = norm_coord_rub_org
        pixels = pixels_org
        origins = np.zeros([size[0] * size[1], 3], dtype=float)

    viewdir_rub = F.normalize(norm_coord, dim=-1)
    ray_vec = viewdir_rub.cpu().numpy()  # ray vector is camera to pixel
    pixels = (pixels[0, :, :, :2].reshape([-1, 2]) - 0.5).int().cpu().numpy()
    viewdir_rub = -viewdir_rub  # view dir is pixel -> camera
    return pixels, norm_coord_rub_org, viewdir_rub, origins, ray_vec, ls, up, rub_to_rdf, envWeight_ndotl, bb, shadWeight, ndotl


def rotateEnvmap(envmap, vn):
    up = np.array([0, 1, 0], dtype=np.float32)
    z = vn
    z = z / np.sqrt(np.sum(z * z))
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x))
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y))

    # x = np.asarray([x[2], x[0], x[1]], dtype = np.float32 )
    # y = np.asarray([y[2], y[0], y[1]], dtype = np.float32 )
    # z = np.asarray([z[2], z[0], z[1]], dtype = np.float32 )
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]

    R = np.concatenate([x, y, z], axis=0)
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]

    envmapRot = np.zeros(envmap.shape, dtype=np.float32)
    height, width = envmapRot.shape[0], envmapRot.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            theta = r / float(height - 1) * np.pi
            phi = (c / float(width) * np.pi * 2 - np.pi)
            z = np.sin(theta) * np.cos(phi)
            x = np.sin(theta) * np.sin(phi)
            y = np.cos(theta)
            coord = x * rx + y * ry + z * rz
            nx, ny, nz = coord[0], coord[1], coord[2]
            thetaNew = np.arccos(nz)
            nx = nx / (np.sqrt(1 - nz * nz) + 1e-12)
            ny = ny / (np.sqrt(1 - nz * nz) + 1e-12)
            nx = np.clip(nx, -1, 1)
            ny = np.clip(ny, -1, 1)
            nz = np.clip(nz, -1, 1)
            phiNew = np.arccos(nx)
            if ny < 0:
                phiNew = - phiNew
            u, v = angleToUV(thetaNew, phiNew)
            color = uvToEnvmap(envmap, u, v)
            envmapRot[r, c, :] = color
    return envmapRot


def angleToUV(theta, phi):
    u = (phi + np.pi) / 2 / np.pi
    v = 1 - theta / np.pi
    return u, v


def uvToEnvmap(envmap, u, v):
    height, width = envmap.shape[0], envmap.shape[1]
    c, r = u * (width - 1), (1 - v) * (height - 1)
    cs, rs = int(c), int(r)
    ce = min(width - 1, cs + 1)
    re = min(height - 1, rs + 1)
    wc, wr = c - cs, r - rs
    color1 = (1 - wc) * envmap[rs, cs, :] + wc * envmap[rs, ce, :]
    color2 = (1 - wc) * envmap[re, cs, :] + wc * envmap[re, ce, :]
    color = (1 - wr) * color1 + wr * color2
    return color


def preprocess_txt(txt_scene, txt_coord, txt_scale, txt_trans):
    txt_scene = txt_scene.strip().split(' ')[0]
    txt_coord = txt_coord.strip().split('coord_ins_start_list = ')[1].replace('[', '').replace(']', '').replace(' ',
                                                                                                                '').split(
        ',')
    txt_scale = txt_scale.strip().split('scale_list = ')[1].replace('[', '').replace(']', '').replace(' ',
                                                                                                      '').split(
        ',')
    txt_trans = txt_trans.strip().split('trans_list = ')[1].replace('[', '').replace(']', '').replace(' ',
                                                                                                      '').split(
        ',')
    num_obj = len(txt_coord) // 2

    coord = []
    scale = []
    trans = []
    for i in range(num_obj):
        coord.append([int(txt_coord[2 * i]), int(txt_coord[2 * i + 1])])
        scale.append(float(txt_scale[i]))
        trans.append([float(txt_trans[3 * i]), float(txt_trans[3 * i + 1]), float(txt_trans[3 * i + 2])])

    return txt_scene, coord, scale, trans


def txt_parsing(contents):
    content_idx = []
    content_list = []
    for i, content in enumerate(contents):
        if content[0] == '%':
            content_idx.append(i)
    num_content = len(content_idx)
    content_idx.append(len(contents) + 1)

    for i in range(num_content):
        dict_tmp = {}
        content_tmp = contents[content_idx[i]:content_idx[i + 1] - 1]
        dict_tmp['obj'] = content_tmp[0].strip().replace('%', '')
        dict_tmp['scenes'] = content_tmp[1::5]
        dict_tmp['coord'] = content_tmp[2::5]
        dict_tmp['scale'] = content_tmp[3::5]
        dict_tmp['trans'] = content_tmp[4::5]
        content_list.append(dict_tmp)
    return content_list


def get_tr(val_dict, coord, scale=1.0, trans=[0, 0, 0]):
    scale_mat = tr.scale_matrix(scale)
    mat2 = tr.translation_matrix(trans)
    mat1 = tr.translation_matrix(val_dict['coord'][:, coord[1], coord[0]])
    vn = val_dict['normal'][:, coord[1], coord[0]]
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rotateAxis = np.cross(up, vn)
    if np.sum(rotateAxis * rotateAxis) <= 1e-6:
        rotateAxis = None
        rotateAngle = None
    else:
        rotateAxis = rotateAxis / np.sqrt(np.sum(rotateAxis * rotateAxis))
        rotateAngle = np.arccos(np.sum(vn * up))
    mat = tr.rotation_matrix(rotateAngle, rotateAxis)
    transformation_matrix = mat2 @ mat1 @ mat @ scale_mat
    return transformation_matrix


def get_env(li_root, coord):
    env_org = np.load(osp.join(li_root, 'env.npz'))['env'][..., ::-1]
    envRow, envCol = env_org.shape[0], env_org.shape[1]
    cId, rId = coord[0] // 4, coord[1] // 4
    rId = np.clip(np.round(rId), 0, envRow - 1)
    cId = np.clip(np.round(cId), 0, envCol - 1)
    rId, cId = int(rId), int(cId)
    env = env_org[rId, cId, :, :, :]
    env = cv2.resize(env, (1024, 512), interpolation=cv2.INTER_LINEAR)
    # envBalck = np.zeros([256, 1024, 3], dtype=np.float32)
    # env = np.concatenate([env, envBalck], axis=0)
    # env = rotateEnvmap(env, vn)
    env = np.maximum(env, 0)
    return env


def oi_path(common_dict, path_method, testroot, li_root):
    if path_method == 'manual':
        outdir = osp.join(testroot, f'OI_path_manual')
        overlap = True
        mat_list = sorted(glob.glob(osp.join(li_root, '*new.mat')))
        for mat in mat_list:
            info = io.loadmat(mat)
            x = int(info['xObj'])
            y = int(info['yObj'])
            print(f'[{x}, {y}],', end=' ')
        tr_list = []
        coord_ins_start_list = [[221, 175], [207, 309]]
        scale_list = [1.5, 1.5, 1.4]
        trans_list = [[0., 0.1, 0.0], [0., 0.05, 0.0]]
        for coord, scale, trans in zip(coord_ins_start_list, scale_list, trans_list):
            tr_list.append(get_tr(common_dict, coord, scale, trans))

    if path_method == 'nerf':
        outdir = osp.join(testroot, f'OI_path_nerf')
        overlap = True
        tr_list = []
        coord_ins_start_list = [[221, 175], [207, 309]]
        scale_list = [1.5, 1.5, 1.4]
        trans_list = [[0., 0.1, 0.0], [0., 0.05, 0.0]]
        for coord, scale, trans in zip(coord_ins_start_list, scale_list, trans_list):
            tr_list.append(get_tr(common_dict, coord, scale, trans))

    elif path_method == 'mouse':
        overlap = False
        outdir = osp.join(testroot, f'OI_path_mouse')
        path_list = path_from_mouse(common_dict['img'][:, :, ::-1].astype(np.float32))
        tr_list = []
        for path in path_list:
            tr_list.append(get_tr(common_dict, path, scale=2.1))

    os.makedirs(outdir, exist_ok=True)
    return tr_list, overlap, outdir


def mesh_init(object_type):
    self_occ = True
    if object_type == 'bunny':
        mesh_scale = 0.3
        position = [0.0, -0.033, 0.0]
        rot_axis = [0.0, 1.0, -0.1]
        rot_axis = np.array(rot_axis, dtype=np.float32)
        rot_axis = rot_axis / np.sqrt(np.sum(rot_axis * rot_axis))
        mat2 = tr.rotation_matrix(np.pi, rot_axis)
        rot_axis = [0.0, 1.0, 0.0]
        mat3 = tr.rotation_matrix(np.pi, rot_axis)
        rot_mat = mat3 @ mat2
        mesh = trimesh.load('asset/bunny.ply')
    elif object_type == 'happy':
        mesh = trimesh.load('asset/happy.ply')
        mesh_scale = 1.0
        position = [0.0, -0.0511, 0.0]
        rot_mat = tr.identity_matrix()
    elif object_type == 'armadillo':
        mesh = trimesh.load('asset/armadillo.ply')
        mesh_scale = 0.001
        position = [0.0, 52.757, 0.0]
        rot_axis = [0.0, 1.0, 0.0]
        rot_mat = tr.rotation_matrix(np.pi, rot_axis)
    elif object_type == 'lucy':
        mesh = trimesh.load('asset/lucy.ply')
        mesh_scale = 0.00005
        position = [0.0, 780, 0.0]
        rot_axis = [0.0, 1.0, 0.0]
        rot_mat = tr.rotation_matrix(np.pi, rot_axis)
    elif object_type == 'dragon':
        mesh = trimesh.load('asset/dragon.ply')
        mesh_scale = 0.3
        position = [0.0, -0.053, 0.0]
        rot_axis = [0.0, 1.0, 0.0]
        rot_mat = tr.rotation_matrix(np.pi / 4, rot_axis)
    else:
        # chrome sphere
        self_occ = False
        mesh_scale = 0.03
        mesh = trimesh.load('asset/sphere.obj')
        position = [0.0, 1.0, 0.0]
        rot_mat = tr.identity_matrix()

    mat1 = tr.translation_matrix(position)
    mat = tr.scale_matrix(mesh_scale)
    mat_ret = mat @ rot_mat @ mat1
    mesh.apply_transform(mat_ret)
    return mesh, self_occ, mat_ret
