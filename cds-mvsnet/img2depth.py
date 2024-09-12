import argparse
from tqdm import tqdm
import glob
from parse_config import ConfigParser
from cds_utils import *
from depth_func import mvs_depth
torch.backends.cudnn.benchmark = True

ckpt = 'pretrained/fine_tuning_on_blended/cds_mvsnet.ckpt'
outdir = 'cds_out'

batch_size = 1
num_worker = 3

num_depth = 1024
prob_threshold = 0.8
refinement = False
num_gpu = 1

org_h = 480
org_w = 640

if refinement:
    max_h = org_h * 2
    max_w = org_w * 2
else:
    max_h = org_h
    max_w = org_w

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--config', default=None, type=str, help='config file path (default: None)')
parser.add_argument('--batch_size', type=int, default=batch_size, help='testing batch size')
parser.add_argument('--max_h', type=int, default=max_h, help='testing batch size')
parser.add_argument('--max_w', type=int, default=max_w, help='testing batch size')
parser.add_argument('--org_h', type=int, default=org_h, help='testing batch size')
parser.add_argument('--org_w', type=int, default=org_w, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=num_depth, help='the number of depth values')

parser.add_argument('--resume', default=ckpt, help='load a specific checkpoint')
parser.add_argument('--outdir', default=outdir, help='output dir')
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--temperature', type=float, default=0.01, help='temperature of softmax')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization cds_base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--refinement', default=refinement, action="store_true", help='depth refinement in last stage')
parser.add_argument('--full_res', action="store_true", help='full resolution prediction')

parser.add_argument('--num_view', type=int, default=1, help='num of view')
parser.add_argument('--num_worker', type=int, default=num_worker, help='depth_filer worker')

# parse arguments and check
args = parser.parse_args()


root = '/home/vig-titan-118/PycharmProjects/MAIR_Open/Examples/input_processed'
if __name__ == '__main__':
    dir_suffix = f'images_{max_w}x{max_h}'
    basedir_list = sorted(glob.glob(osp.join(root, '*')))
    basedir_list_2 = []
    for basedir in tqdm(basedir_list):
        basedir = osp.join(basedir, dir_suffix)
        if len(glob.glob(osp.join(basedir, '*.dat'))) == 2 * len(glob.glob(osp.join(basedir, '*.png'))):
            continue
        basedir_list_2.append(basedir)

    config = ConfigParser.from_args(parser)
    gpu = 0
    args.num_view = 10
    mvs_depth(gpu, 1, args, basedir_list_2, config, False)
