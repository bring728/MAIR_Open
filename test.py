import torch.multiprocessing
import socket
from record import eval_model
from network.net_backbone import ResUNet
from utils import *
from loader import load_id_wandb, load_dataloader, load_model
from loss import RecLoss


def test(gpu, num_gpu, run_mode, phase_list, dataRoot, outputRoot, is_DDP=False, run_id=None, config=None, port=2958, num_K=None):
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{port}', world_size=num_gpu,
                                             rank=gpu)
        torch.cuda.set_device(gpu)

    cfg, run_id, wandb_obj, experiment = load_id_wandb(config, False, outputRoot, run_id)
    if run_mode == 'test':
        if cfg.mode == 'VSG':
            cfg.batchsize = 1
            cfg.num_workers = 1
        else:
            cfg.batchsize = 4
            cfg.num_workers = 3
    elif run_mode == 'output':
        cfg.batchsize = 1
        cfg.num_workers = 1

    if num_K is not None:
        cfg.num_K = num_K
    cfg.full_load = True
    dict_loaders = load_dataloader(dataRoot, cfg, is_DDP, phase_list)

    model = load_model(cfg, gpu, experiment, is_train=False, is_DDP=is_DDP, wandb_obj=wandb_obj)
    model.switch_to_eval()
    if dict_loaders is not None:
        for phase in dict_loaders:
            data_loader, _ = dict_loaders[phase]
            if run_mode == 'test':
                cfg.losskey.append('rgb')
                cfg.losstype.append('l2')
                cfg.weight.append(1.0)
                cfg.lossmask.append('mask')
                loss_agent = RecLoss(cfg)
                eval_dict, _ = eval_model(model, data_loader, gpu, cfg, num_gpu, loss_agent, 'test', 0)
                print(eval_dict)
    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    dataroot = '/media/vig-titan/Samsung_T5/OpenRoomsFF320'
    pretrained = 'pretrained'
    root = [dataroot, pretrained]
    run_id = '05190941_VSG'
    run_mode = 'test'
    phase_list = ['test', ]
    test(0, 1, run_mode, phase_list, root[0], root[1], False, run_id=run_id, config=None)
