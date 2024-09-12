import glob
from tqdm import tqdm
from cds_utils import *
from colmap_helper import exec_colmap, load_colmap_data, save_poses, minify
from colmap2mvsnet import processing_single_scene_my, processing_single_scene_deepview

img_h1 = 240
img_w1 = 320
img_h2 = 480
img_w2 = 640

root = '/home/vig-titan-118/PycharmProjects/MAIR_Open/Examples/input'
outroot = '/home/vig-titan-118/PycharmProjects/MAIR_Open/Examples/input_processed'

if __name__ == '__main__':
    basedir_list = sorted(glob.glob(osp.join(root, '*')))
    for basedir in tqdm(basedir_list):
        print(basedir)
        outdir = os.path.join(outroot, osp.basename(basedir))
        # if osp.exists(osp.join(outdir, 'pair.txt')):
        #     continue
        imagedir = osp.join(basedir, 'images')
        if exec_colmap(imagedir, basedir, single_camera='1', camera_model='PINHOLE', match_type='exhaustive_matcher'):
            print('colmap find poses')
            processing_single_scene_my(basedir, outdir)
        else:
            print(basedir, 'cant not find sparse!')

        resolutions = [(img_h1, img_w1), (img_h2, img_w2)]
        minify(outdir, resolutions=resolutions)
