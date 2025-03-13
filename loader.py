import torch.cuda
from MAIR import MAIRorg, MAIRplusplus
from torch.utils.data import DataLoader
from mvsd_dataset import *
from datetime import datetime
import os
import os.path as osp
from cfgnode import CfgNode
import yaml
import wandb
from torch import nn


def load_id_wandb(config, record_flag, outputRoot, id=None):
    if (config is None) == (id is None):
        raise Exception('One of the two must be set.')

    if config is None:
        config = glob.glob(osp.join(outputRoot, id, '*.yml'))[0]
        run_id = id
        print('config restored from: ', run_id)
    else:
        if len(config.split('.')) == 1:
            config = config + '.yml'
        config = os.getcwd() + '/config/' + config

    with open(config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
        mode = cfg.mode
        seed = cfg.randomseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    wandb_obj = None
    if id is None:
        current_time = datetime.now().strftime('%m%d%H%M')
        run_id = f'{current_time}_{cfg.mode}'
        if record_flag:
            wandb_obj = wandb.init(project=f'MAIR-{mode}', id=run_id)
            wandb_obj.config.update(cfg)
    else:
        if record_flag:
            wandb_obj = wandb.init(project=f'MAIR-{mode}', id=run_id, resume=True)

    experiment = osp.join(outputRoot, run_id)
    if record_flag and id is None:
        os.makedirs(experiment, exist_ok=True)
        os.system(f'cp *.py {experiment}')
        os.system(f'cp network/*.py {experiment}')
        os.system(f'cp {config} {experiment}')
    return cfg, run_id, wandb_obj, experiment


def load_dataloader(dataRoot, cfg, is_DDP, phase_list, debug=False):
    worker_per_gpu = cfg.num_workers
    batch_per_gpu = cfg.batchsize
    print('batch_per_gpu', batch_per_gpu, 'worker_per_gpu', worker_per_gpu)

    dict_loader = {}
    for phase in phase_list:
        sampler = None
        is_shuffle = True
        if phase == 'custom':
            dataset = realworld_FF(dataRoot, cfg)
            is_shuffle = False
        elif phase == 'custom_MG':
            dataset = realworld_FF_singleview(dataRoot, cfg)

        elif phase == 'mat_edit':
            dataset = mat_edit_dataset(dataRoot, cfg)
            is_shuffle = False
        else:
            dataset = OpenroomsFF(dataRoot, cfg, phase, debug)
            if phase == 'test':
                is_shuffle = False
        if is_DDP:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            is_shuffle = False

        pinned = cfg.pinned and phase == 'TRAIN'
        loader = DataLoader(dataset, batch_size=batch_per_gpu, shuffle=is_shuffle, num_workers=worker_per_gpu,
                            pin_memory=pinned, sampler=sampler)

        dict_loader[phase] = [loader, sampler]
        print(f'create dataset - mode {cfg.mode}, shuffle: {is_shuffle}')
        print(f'{phase} dataset number of sample: {dataset.length}')
        print(f'{phase} loader number of sample: {len(loader)}')
    return dict_loader


def load_model(cfg, gpu, experiment, is_train, is_DDP, wandb_obj):
    if cfg.version == 'MAIR++':
        curr_model = MAIRplusplus(cfg, gpu, experiment, is_train, is_DDP=is_DDP)
    elif cfg.version == 'MAIR':
        curr_model = MAIRorg(cfg, gpu, experiment, is_train, is_DDP=is_DDP)

    do_watch = wandb_obj is not None
    if do_watch:
        watch_model = []
        for k in curr_model.train_key:
            watch_model.append(getattr(curr_model, k))
        wandb_obj.watch(watch_model, log='all')

    for name, model in curr_model.__dict__.items():
        if isinstance(model, nn.Module):
            # if isinstance(model, VSGEncoder):
            #     total_params = sum(p.numel() for p in model.enc.parameters() if p.requires_grad)
            #     print(f"Total trainable parameters in {name}-enc: {total_params}, {total_params * 1.e-6:.2f} M params.")
            #     total_params = sum(p.numel() for p in model.dec.parameters() if p.requires_grad)
            #     print(f"Total trainable parameters in {name}-dec: {total_params}, {total_params * 1.e-6:.2f} M params.")
            # else:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters in {name}: {total_params}, {total_params * 1.e-6:.2f} M params.")
    return curr_model
