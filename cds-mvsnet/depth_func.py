import argparse, os, time, sys, gc, cv2
import cds_datasets.data_loaders as module_data
import cds_models.model as module_arch
from cds_utils import *
import platform
import struct

if platform.platform().startswith('Windows'):
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

prob_threshold = 0.8


def th_save_img(image, filename):
    image_jpg = (image * 255.0).astype(np.uint8)
    cv2.imwrite(filename, image_jpg)


def saveBinary(filename, image):
    file = open(filename, "wb")
    h, w = image.shape
    data = struct.pack('i', h)
    file.write(data)
    data = struct.pack('i', w)
    file.write(data)
    depth = image.reshape(-1).astype(np.float32)
    depth.tofile(file)
    file.close()


# run model to save depth maps and confidence maps
def mvs_depth(gpu, num_gpu, args, mvs_list, config, is_DDP):
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2058', world_size=num_gpu,
                                             rank=gpu)
        torch.cuda.set_device(gpu)
    device = torch.device('cuda:{}'.format(gpu))

    if len(mvs_list) != 0:
        init_kwags = {
            "num_srcs": args.num_view,
            "num_depths": args.numdepth,
            "batch_size": args.batch_size,
            "num_worker": args.num_worker,
            "max_h": args.max_h,
            "max_w": args.max_w,
            "refine": args.refinement,
        }
        test_data_loader = module_data.FFLoader(mvs_list, **init_kwags, is_DDP=is_DDP)

        # model
        # build cds_models architecture
        if not args.refinement:
            config["arch"]["args"]["refine"] = False
        model = config.init_obj('arch', module_arch)

        checkpoint = torch.load(str(config.resume))
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key.replace('module.', '')] = val
        model.load_state_dict(new_state_dict, strict=False)

        # prepare cds_models for testing
        model = model.to(device)
        model.eval()
        times = []

        # f_error = open('mvs_depth_error.txt', 'r')
        # scene_unsafe = [a.strip() for a in f_error.readlines()]
        # f_error.close()

        with torch.no_grad():
            for batch_idx, sample in enumerate(test_data_loader):
                start_time = time.time()
                sample_cuda = tocuda(sample)
                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
                outputs = model(imgs, cam_params, sample_cuda["depth_values"], temperature=args.temperature)
                # try:
                #     if sample['filename'][0] in scene_unsafe:
                #         outputs = model(imgs, cam_params, sample_cuda["depth_values"], temperature=args.temperature, safe=True)
                #     else:
                #         outputs = model(imgs, cam_params, sample_cuda["depth_values"], temperature=args.temperature)
                # except RuntimeError:
                #     print(sample['filename'], '%^& error')
                #     f_error = open('mvs_depth_error.txt', 'a')
                #     f_error.writelines('\n'.join(sample['filename']))
                #     f_error.write('\n')
                #     f_error.close()
                #     continue
                # outputs["ps_map"] = model.feature.extract_ps_map()

                end_time = time.time()
                times.append(end_time - start_time)
                outputs = tensor2numpy(outputs)
                del sample_cuda
                print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(test_data_loader), end_time - start_time,
                                                          outputs["refined_depth"][0].shape))
                filenames = sample["filename"]

                # save depth maps and confidence maps
                for filename, depth_est, conf_stage1, conf_stage2, conf_stage3 in zip(filenames,
                                                                                      outputs["refined_depth"],
                                                                                      outputs["stage1"][
                                                                                          "photometric_confidence"],
                                                                                      outputs["stage2"][
                                                                                          "photometric_confidence"],
                                                                                      outputs[
                                                                                          "photometric_confidence"]):  # , outputs["ps_map"]):
                    depth_filename = filename.format('cdsdepthest', '.dat')
                    depth_low_filename = depth_filename.replace('640x480', '320x240')
                    confidence_filename = filename.format('cdsconf', '.dat')
                    confidence_low_filename = confidence_filename.replace('640x480', '320x240')
                    depthvis_filename = filename.format('cdsdepthvis', '.jpg')

                    # depth_est = cv2.resize(depth_est, (args.org_w, args.org_h), interpolation=cv2.INTER_NEAREST)
                    conf_stage1 = cv2.resize(conf_stage1, (args.org_w, args.org_h), interpolation=cv2.INTER_NEAREST)
                    conf_stage2 = cv2.resize(conf_stage2, (args.org_w, args.org_h), interpolation=cv2.INTER_NEAREST)
                    conf_stage3 = cv2.resize(conf_stage3, (args.org_w, args.org_h), interpolation=cv2.INTER_NEAREST)
                    prob_map = np.stack([conf_stage1, conf_stage2, conf_stage3]).transpose([1, 2, 0])
                    prob_map = np.min(prob_map, axis=-1)

                    depth_est_nan = depth_est.copy()
                    flag_success = False
                    prob_threshold_tmp = prob_threshold
                    for i in range(5):
                        mask = (prob_map > prob_threshold_tmp)
                        if np.sum(mask) >= 640 * 480 / 4:
                            # print(f'{depth_filename}, conf: {prob_threshold_tmp}')
                            flag_success = True
                            break
                        prob_threshold_tmp -= 0.05
                    if flag_success:
                        depth_est_nan[~mask] = np.nan
                    else:
                        print(f'{depth_filename}, conf: is low')
                        # f_conf = open('conf_low.txt', 'a')
                        # f_conf.write(confidence_filename)
                        # f_conf.write('\n')
                        # f_conf.close()

                    depth_max = np.nanpercentile(depth_est_nan, 99.0)
                    depth_min = np.nanpercentile(depth_est_nan, 1.0)
                    if depth_max < 0.4:
                        raise Exception(depth_filename, 'depth max error')
                    if depth_min < 0:
                        raise Exception(depth_filename, 'depth min error')
                    depth_est_low = cv2.resize(depth_est, (320, 240), interpolation=cv2.INTER_NEAREST)
                    prob_map_low = cv2.resize(prob_map, (320, 240), interpolation=cv2.INTER_NEAREST)
                    depthmvs_normalized = np.clip(depth_est_low / depth_max, 0, 1)
                    th_save_img(depthmvs_normalized, depthvis_filename)
                    saveBinary(depth_filename, depth_est)
                    saveBinary(depth_low_filename, depth_est_low)
                    saveBinary(confidence_filename, prob_map)
                    saveBinary(confidence_low_filename, prob_map_low)

        torch.cuda.empty_cache()
        gc.collect()
