import torch.multiprocessing
import socket
from record import print_state, record_images, eval_model, want_save_now
from utils import *
from loader import load_id_wandb, load_dataloader, load_model
from loss import RecLoss

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False


def train(gpu, num_gpu, dataRoot, outputRoot, config, debug=False, is_DDP=False, run_id=None, accum_iter=1, port=2958):
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{port}', world_size=num_gpu,
                                             rank=gpu)
        torch.cuda.set_device(gpu)
    record_flag = ((is_DDP and gpu == 0) or not is_DDP) and not debug
    # if debug:
    #     torch.autograd.set_detect_anomaly(True)
    cfg, run_id, wandb_obj, experiment = load_id_wandb(config, record_flag, outputRoot, run_id)
    if hasattr(cfg, 'accum_iter'):
        accum_iter = cfg.accum_iter

    ckpt_prefix = osp.join(experiment, f'model_{cfg.mode}')
    exp_name = osp.basename(experiment)

    phases = ['train', cfg.eval_format[0]]
    dict_loader = load_dataloader(dataRoot, cfg, is_DDP, phases, debug=debug)
    train_loader, train_sampler = dict_loader['train']
    eval_loader, _ = dict_loader[phases[1]]
    train_size = len(train_loader)
    assert cfg.eval_format[1] == 'step' or cfg.eval_format[1] == 'epoch'
    eval_period = cfg.eval_format[2] * train_size if cfg.eval_format[1] == 'epoch' else cfg.eval_format[2]
    total_steps_global = train_size * cfg.nepoch
    total_steps_record = math.ceil(train_size * cfg.nepoch / accum_iter)

    scaler = torch.cuda.amp.GradScaler()
    model = load_model(cfg, gpu, experiment, is_train=True, is_DDP=is_DDP, wandb_obj=wandb_obj)
    model.set_scheduler(cfg, total_steps_record)
    model.switch_to_train()

    record_step = model.start_step
    start_epoch = round(record_step * accum_iter / train_size)
    global_step = int(start_epoch * eval_period)
    if record_flag:
        print(f'start epoch: {start_epoch}, start step: {global_step}, record step: {record_step}')
    # model.save_model('best.pth')

    loss_agent = RecLoss(cfg)
    start_t = time.time()
    prev_t = time.time()

    for epoch in range(start_epoch + 1, cfg.nepoch + 1):
        if not train_sampler is None:
            train_sampler.set_epoch(epoch)
        for train_data in train_loader:
            global_step += 1
            train_data = tocuda(train_data, gpu, cfg.pinned)
            pred, gt, mask = model.forward(train_data, cfg, 'train')
            total_loss, scalars_to_log = loss_agent(train_data, pred, gt, mask, 'train', global_step)
            if debug:
                record_images(model, cfg, wandb_obj, train_data, pred, gt, mask, global_step)
                prev_t, sec_it = print_state(exp_name, start_t, prev_t, total_steps_global, global_step, 1, total_loss.item(),
                                             False)

            total_loss = total_loss / accum_iter
            scaler.scale(total_loss).backward()
            if global_step % accum_iter == 0 or global_step == total_steps_global:
                record_step += 1
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad()
                model.scheduler_step()
                if record_flag:
                    if record_step % cfg.i_print == 0:
                        prev_t, sec_it = print_state(exp_name, start_t, prev_t, total_steps_record, record_step,
                                                     cfg.i_print, total_loss.item())
                        scalars_to_log['train/sec_it'] = sec_it
                    if record_step % cfg.i_record == 0:
                        model.get_last_lr(scalars_to_log)
                        wandb_obj.log(scalars_to_log, step=record_step)
                    if want_save_now(record_step):
                        record_images(model, cfg, wandb_obj, train_data, pred, gt, mask, record_step)

            if global_step % eval_period == 0 or global_step == total_steps_global:
                eval_dict, _ = eval_model(model, eval_loader, gpu, cfg, num_gpu, loss_agent, phases[1], global_step, True)
                if record_flag:
                    fpath = f'{ckpt_prefix}_{record_step:0>6}.pth'
                    model.save_model(fpath)
                    # eval_dict['test/epoch'] = epoch
                    print(record_step, epoch, eval_dict)
                    wandb_obj.log(eval_dict, step=record_step)

    # torch.cuda.empty_cache()
    if record_flag:
        wandb_obj.finish()
    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":  # config = 'MG.yml'
    # config = 'incident_org.yml'
    # config = 'exitant_org.yml'
    # config = 'incident_sg.yml'
    # config = 'BRDF_feat_mvaattn_abl_1.yml'

    # config = 'BRDF_feat_rnn.yml'
    # config = 'BRDF_org_cds_abl_4.yml'

    config = 'MG.yml'
    # config = 'incident.yml'
    # config = 'AlbedoFusion.yml'
    # config = 'VSG.yml'
    # config = 'VSG_org.yml'
    # config = 'VSG_vit.yml'

    hostname = socket.gethostname()
    if hostname == 'vigtitan118-System-Product-Name':
        root = ['/media/vig-titan-118/Seagate/OpenRoomsFF320', '/media/vig-titan-118/Samsung_T5/MAIR_output']
    elif hostname == 'daniff':
        root = ['/home/happily/Data/OpenRoomsFF320', '/home/happily/Data/MAIR_output']
    elif hostname == 'vigtitan168-MS-7B10':
        root = ['/home/vig-titan-168/Data/OpenRoomsFF320', '/home/vig-titan-168/Data/MAIR_output']
    else:
        raise Exception('check setting')

    accum_iter = 1
    run_id = None
    # run_id = '04151435_BRDF'
    # config = None
    debug = True
    # debug = False
    train(1, 1, root[0], root[1], config, debug, False, run_id, accum_iter)
