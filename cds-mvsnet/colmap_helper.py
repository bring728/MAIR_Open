import glob
import cv2
import numpy as np
import os
from subprocess import check_output, CalledProcessError
import colmap_read_model as read_model
import os.path as osp
import shutil


def exec_colmap(imagedir, outdir, single_camera, camera_model, match_type):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(outdir, 'sparse/0')):
        files_had = os.listdir(os.path.join(outdir, 'sparse/0'))
    else:
        files_had = []
    if all([f in files_had for f in files_needed]):
        print(f'Don\'t need to run COLMAP on {outdir}')
        return True
    else:
        print('Need to run COLMAP')
        if run_feature_extractor(imagedir, outdir, camera_model, single_camera):
            if run_feature_matching(outdir, match_type):
                if run_sparse(imagedir, outdir):
                    return True
        return False


def run_feature_extractor(imagedir, outdir, camera_model, single_camera):
    logfile_name = os.path.join(outdir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a+')

    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(outdir, 'database.db'),
        '--image_path', imagedir,
        '--ImageReader.camera_model', camera_model,
        '--ImageReader.single_camera', single_camera,
    ]
    try:
        feat_output = (check_output(feature_extractor_args, universal_newlines=True))
        logfile.write(feat_output)
        logfile.close()
        print('Features extracted')
        return True
    except CalledProcessError as exc:
        logfile.write(exc.output)
        logfile.close()
        print('Features extract failed')
        return False


def run_feature_matching(outdir, match_type):
    logfile_name = os.path.join(outdir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a+')

    exhaustive_matcher_args = ['colmap', match_type, '--database_path', os.path.join(outdir, 'database.db'),
                               '--SiftMatching.guided_matching', '1', '--SiftMatching.max_num_matches', '29000']
    try:
        match_output = (check_output(exhaustive_matcher_args, universal_newlines=True))
        logfile.write(match_output)
        logfile.close()
        print('Features matched')
        return True
    except CalledProcessError as exc:
        logfile.write(exc.output)
        logfile.close()
        print('Features match failed')
        return False


def run_sparse(imagedir, outdir):
    logfile_name = os.path.join(outdir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a+')

    p = os.path.join(outdir, 'sparse/0')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(outdir, 'database.db'),
        '--image_path', imagedir,
        '--output_path', os.path.join(outdir, 'sparse'),  # --export_path changed to --output_path in colmap 3.6
    ]
    try:
        map_output = (check_output(mapper_args, universal_newlines=True))
        logfile.write(map_output)
        logfile.close()
        print('Sparse map created')
        return True
    except CalledProcessError as exc:
        logfile.write(exc.output)
        logfile.close()
        print('Sparse map failed')
        return False


def run_dense(basedir):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a+')

    p = os.path.join(basedir, 'dense')
    if not os.path.exists(p):
        os.makedirs(p)

    image_undistorter_args = [
        'colmap', 'image_undistorter',
        '--image_path', os.path.join(basedir, 'images'),
        '--input_path', os.path.join(basedir, 'sparse/0'),
        '--output_path', os.path.join(basedir, 'dense'),
    ]

    output = (check_output(image_undistorter_args, universal_newlines=True))
    logfile.write(output)

    patch_match_stereo_args = [
        'colmap', 'patch_match_stereo',
        '--workspace_path', os.path.join(basedir, 'dense'),
    ]

    output = (check_output(patch_match_stereo_args, universal_newlines=True))
    logfile.write(output)

    stereo_fusion_args = [
        'colmap', 'stereo_fusion',
        '--workspace_path', os.path.join(basedir, 'dense'),
        '--output_path', os.path.join(basedir, 'dense/fused.ply'),
    ]

    output = (check_output(stereo_fusion_args, universal_newlines=True))
    logfile.write(output)

    # delaunay_mesher_args = [
    #     'colmap', 'delaunay_mesher',
    #     '--input_path', os.path.join(basedir, 'dense'),
    #     '--output_path', os.path.join(basedir, 'dense/meshed-delaunay.ply'),
    # ]
    #
    # output = (check_output(delaunay_mesher_args, universal_newlines=True))
    # logfile.write(output)
    #
    # logfile.close()
    print('dense map created')


def run_triangulator(imagedir, outdir):
    logfile_name = os.path.join(outdir, 'colmap_output.txt')
    logfile = open(logfile_name, 'a+')

    p = os.path.join(outdir, 'sparse/0')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'point_triangulator',
        '--database_path', os.path.join(outdir, 'database.db'),
        '--image_path', imagedir,
        '--input_path', os.path.join(outdir, 'manual'),
        '--output_path', os.path.join(outdir, 'sparse/0'),  # --export_path changed to --output_path in colmap 3.6
    ]
    try:
        map_output = (check_output(mapper_args, universal_newlines=True))
        logfile.write(map_output)
        logfile.close()
        print('point_triangulator succeed')
        return True
    except CalledProcessError as exc:
        logfile.write(exc.output)
        logfile.close()
        print('point_triangulator failed')
        return False


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    cam_ids = list(camdata.keys())
    im_ids = list(imdata.keys())

    if len(cam_ids) == 1:
        print('single camera')
        cam = camdata[cam_ids[0]]
        h, w, f = cam.height, cam.width, cam.params[0]
        # w, h, f = factor * w, factor * h, factor * f
        hwf = np.array([h, w, f]).reshape([3, 1])
        hwf = np.tile(hwf[..., np.newaxis], [1, 1, len(im_ids)])
    else:
        print('multiple camera')
        hwf = []
        if cam_ids != im_ids:
            print('cam_ids and im_ids are not the same.')
            print(cam_ids)
            print(im_ids)
            return

        for k in camdata:
            cam = camdata[k]
            hwf.append(np.array([cam.height, cam.width, cam.params[0]]).reshape([3, 1]))
        hwf = np.stack(hwf, 2)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for i, k in enumerate(imdata):
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
        filename = osp.join(realdir, 'images', f'{imdata[k].name}')
        if osp.exists(filename):
            new_filename = filename.replace(osp.basename(filename).split('.')[0], f'im_{i + 1:03d}')
            os.rename(filename, new_filename)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, hwf], 1)

    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        # cams = [0] * max(list(ids))
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    vis_arr = vis_arr[:, np.where(np.sum(vis_arr, 0) > 0)[0]]
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    zvals = np.sum((pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]  # zvals ???, vis_arr==1 ? ???.. ? ??? ????? ??? ????? ????? ?????.
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:  # perm. ???? ???.. 0~24
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]  # ? ????? ?? 3d pts? ????.. 1? ????? ?? pts? 1215? ??.
        close_depth, inf_depth = np.percentile(zs, 0.1), np.percentile(zs, 99.9)

        save_arr.append(np.concatenate([poses[..., i], np.array([close_depth, inf_depth, 0.0]).reshape([3, 1])], 1))
    save_arr = np.stack(save_arr, -1)
    # print(f"min : {np.min(save_arr[:, -2])}")
    # print(f"max : {np.max(save_arr[:, -1])}")
    np.save(os.path.join(basedir, 'images/cam_mats.npy'), save_arr)


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def load_colmap_geometry(basedir):
    depth_dir = os.path.join(basedir, 'dense/stereo/depth_maps')
    depth_files = [os.path.join(depth_dir, f) for f in sorted(os.listdir(depth_dir)) if
                   f.endswith('photometric.bin') or f.endswith('geometric.bin')]
    depth_gts = [os.path.join(basedir, 'depth', f'1_imdepth_{i + 1}.dat') for i in range(9)]

    normal_dir = os.path.join(basedir, 'dense/stereo/normal_maps')
    normal_files = [os.path.join(normal_dir, f) for f in sorted(os.listdir(normal_dir)) if
                    f.endswith('photometric.bin') or f.endswith('geometric.bin')]

    depth_maps = []
    normal_maps = []

    for normal_src, depth_src, depth_gt in zip(normal_files, depth_files, depth_gts):
        depth_map = read_array(depth_src)
        normal_map = read_array(normal_src)

        # min_depth = 0.0
        # max_depth = 100.0
        # depth_map[depth_map < min_depth] = min_depth
        # depth_map[depth_map > max_depth] = max_depth
        depth_max = np.percentile(depth_map, 96)
        depth_map = np.clip(depth_map, 0, depth_max)
        depth_map_norm = np.clip(depth_map / depth_max, 0, 1)

        depth_gt_map = loadBinary(depth_gt) * 2 / 3

        np.abs(depth_gt_map - depth_map)

        # plt.figure()
        # plt.imshow(depth_map)
        # plt.title("depth map")

        # Visualize the normal map.
        # plt.figure()
        # plt.imshow(normal_map)
        # plt.title("normal map")
        # plt.show()

    return depth_maps, normal_maps


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    imgdir = os.path.join(basedir, 'images_converted')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100. / r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        # print('Minifying', r, basedir)

        os.makedirs(imgdir)
        for i in glob.glob(osp.join(imgdir_orig, '*.jpg')):
            shutil.copy(i, osp.join(imgdir, osp.basename(i)))

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        # print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            for i in glob.glob(osp.join(imgdir, '*.jpg')):
                os.remove(i)
        # print('Done')
        org_hw = cv2.imread(osp.join(imgdir_orig, 'im_001.jpg')).shape[:2]
        resized_hw = cv2.imread(osp.join(imgdir, 'im_001.png')).shape[:2]

        poses = np.load(osp.join(imgdir_orig, 'cam_mats.npy'))
        factor = (np.array([org_hw[0], org_hw[1]]) / np.array([resized_hw[0], resized_hw[1]])).reshape([2, 1])
        poses[:2, 4, :] /= factor
        factor = factor[::-1]
        poses[2, 4:6, :] /= factor
        np.save(osp.join(imgdir, 'cam_mats.npy'), poses)


if __name__ == "__main__":
    basedir = 'D:/jy_create_data/s10 scene/0810-2'
    exec_colmap(basedir, single_camera='1', camera_model='SIMPLE_PINHOLE', match_type='exhaustive_matcher', dense=True)
    poses, pts3d, perm = load_colmap_data(basedir)
    save_poses(basedir, poses, pts3d, perm)
    depths, normals = load_colmap_geometry(basedir)
