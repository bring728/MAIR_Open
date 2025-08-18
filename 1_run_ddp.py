import sys, os, socket
import torch
from utils import wait_prev_work, count_digits
from train import train
from test import test
import argparse

# from stage_test import test
# import imageio
# imageio.plugins.freeimage.download()

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False

# python 1_run_ddp.py -m test --id 03220859_BRDF
if __name__ == '__main__':
    hostname = socket.gethostname()
    if hostname == 'vigtitan-System-Product-Name':
        root = ['/media/vig-titan/Samsung_T5/OpenRoomsFF320', '/media/vig-titan/Samsung_T5/MAIR_output']
        gpus = '0,1'
        accum_iter = 4
    elif hostname == 'daniff':
        root = ['/home/happily/Data/OpenRoomsFF320', '/home/happily/Data/MAIR_output']
        gpus = '0,1,2,3,4,5,6,7,'
        accum_iter = 1
    elif hostname == 'vigtitan168-MS-7B10':
        root = ['/home/vig-titan-168/Data/OpenRoomsFF320', '/home/vig-titan-168/Data/MAIR_output']
        gpus = '0,1'
        accum_iter = 4
    else:
        raise Exception('check setting')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', help='stage mode')
    parser.add_argument('-c', '--config', type=str, default=None, help='stage mode')
    parser.add_argument('-d', '--data', type=str, default=root[0], help='data root')
    parser.add_argument('-o', '--output', type=str, default=root[1], help='output root')
    parser.add_argument('--gpus', type=str, default=gpus, help='gpus')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--id', type=str, default=None, help='run id')
    parser.add_argument('--port', type=int, default=2912, help='ddp port')
    parser.add_argument('--num_K', type=int, default=None, help='ddp port')
    args = parser.parse_args()
    wait_prev_work(args.gpus, 60)

    # os.rename('/home/happily/Data/MAIR_output/03300605_BRDF/model_BRDF_030550.pth',
    #           '/home/happily/Data/MAIR_output/03300605_BRDF/model_BRDF_best.pth')

    mode = args.mode
    num_gpu = count_digits(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if mode == 'train':
        print(f'port: {args.port}, gpus : {args.gpus} and num of gpus : {num_gpu}')
        torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, args.data, args.output, args.config,
                                                                 args.debug, True, args.id, accum_iter, args.port))
        print('training is done.')
    else:
        if mode == 'test':
            run_mode = 'test'
            phase_list = ['test', ]
        elif mode == 'output':
            run_mode = 'output'
            phase_list = ['custom', ]

        torch.multiprocessing.spawn(test, nprocs=num_gpu, args=(num_gpu, run_mode, phase_list, args.data, args.output,
                                                                True, args.id, None, args.port, args.num_K))
        print(f'{run_mode} is done.')
