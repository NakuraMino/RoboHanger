import subprocess
import multiprocessing
from typing import List, Dict
import os
import sys
import argparse
import time

import datetime
import robohang.common.utils as utils


TRAIN_CLOTHES_NUM = 120
VALID_CLOTHES_NUM = 24
FAST_PASS = False # for debug
OUTPUT_COMMON = "outputs/iterative/funnel"


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def random_base_dir(current: str):
    return f"{OUTPUT_COMMON}/{current}/random"


def train_base_dir(current: str, suffix: str):
    return f"{OUTPUT_COMMON}/{current}/train/{suffix}"


def valid_base_dir(current: str, suffix: str):
    return f"{OUTPUT_COMMON}/{current}/valid/{suffix}"


def net_base_dir(current: str, suffix: str):
    return f"{OUTPUT_COMMON}/{current}/learn/{suffix}"


def find_ckpt(tgt_dir: str):
    for l in sorted(os.listdir(tgt_dir)):
        if l.endswith(".ckpt"):
            return os.path.join(tgt_dir, l)
        elif os.path.isdir(os.path.join(tgt_dir, l)):
            r = find_ckpt(os.path.join(tgt_dir, l))
            if r is not None:
                return r


def collect(
    cuda: int, 
    output_dir: str,
    clothes_path: str,
    net: str, 
):
    cmd_str = (
        "python run/run_funnel_sim_unet.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.funnel_policy.ckpt={net} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
    )
    print(f"cmd_str\n{cmd_str}")
    if FAST_PASS: # debug
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cmd.txt"), "w") as f_obj:
            f_obj.write(cmd_str)
    else:
        ret = subprocess.run(cmd_str, shell=True)
        if ret.returncode != 0:
            exit(ret.returncode)


def validate(
    cuda: int, 
    output_dir: str,
    clothes_path: str,
    net: str, 
    seed: int, 
):
    cmd_str = (
        "python run/run_funnel_sim_unet.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.funnel_policy.ckpt={net} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"glb_cfg.seed={seed} "
    )
    print(f"cmd_str\n{cmd_str}")
    if FAST_PASS: # debug
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cmd.txt"), "w") as f_obj:
            f_obj.write(cmd_str)
    else:
        ret = subprocess.run(cmd_str, shell=True)
        if ret.returncode != 0:
            exit(ret.returncode)


def train_net(
    data_dirs: List[str], 
    gpu_idx: List[int], 
    batch_size: int,
    num_workers: int,
    out_dir: str, 
    ckpt_path: str, 
    step_offset: int,
    opt_step: int,
):
    assert isinstance(data_dirs, list)
    cmd_str = (
        f"python run/run_funnel_learn.py " + 
        f"'train.path.data_paths=[{', '.join(data_dirs)}]' " +
        f"'misc.hardware.gpuids=[{', '.join([str(_) for _ in gpu_idx])}]' " + 
        f"data.common.batch_size={batch_size} " + 
        f"data.common.num_workers={num_workers} " + 
        f"misc.step_offset={step_offset} " +
        f"train.path.exp_name=iterative " + 
        f"train.path.version_name=funnel " + 
        (f"train.path.ckpt={ckpt_path} " if ckpt_path is not None else "") + 
        f"pl.learn.valid.plot_dense_predict_num=1 " +
        f"train.cfg.max_steps={opt_step} " + 
        f"train.cfg.ckpt_every_n_steps={opt_step} " + 
        f"train.cfg.val_check_interval={opt_step // 2} " + 
        f"hydra.run.dir={out_dir}"
    )
    print(f"cmd_str\n{cmd_str}")
    if FAST_PASS: # debug
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "none.ckpt"), "w") as f_obj:
            f_obj.write(cmd_str)
        return
    ret = subprocess.run(cmd_str, shell=True)
    if ret.returncode != 0:
        exit(ret.returncode)


def collect_random(
    cuda: int, 
    output_dir: str,
    clothes_path: str,
):
    cmd_str = (
        "python run/run_funnel_sim_random.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
    )
    print(f"cmd_str\n{cmd_str}")
    if FAST_PASS: # debug
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cmd.txt"), "w") as f_obj:
            f_obj.write(cmd_str)
    else:
        ret = subprocess.run(cmd_str, shell=True)
        if ret.returncode != 0:
            exit(ret.returncode)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_n", type=int, default=6)
    parser.add_argument("--cudas", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--total_traj_valid", type=int, default=12)
    parser.add_argument("--total_traj_train", type=int, default=24)
    parser.add_argument("--valid_interval", type=int, default=1, help="only when iter_idx % valid_interval == 0, we perform validation")
    parser.add_argument("--skip_all_valid", action="store_true")
    parser.add_argument("--curr_train_clothes_idx", type=int, default=0)
    parser.add_argument("--skip_first_valid", action="store_true")
    parser.add_argument("--skip_first_collect", action="store_true")
    parser.add_argument("--opt_step", type=int, default=250)

    parser.add_argument("--net_gpu", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--net_batch_size", type=int, default=8)
    parser.add_argument("--net_num_workers", type=int, default=4)
    parser.add_argument("--net_step_offset", type=int, default=5000)
    parser.add_argument("--net_path", type=str, help="path/to/ckpt")
    parser.add_argument("--data_base_dirs", type=str, nargs="*", default=[], help="offline data path")

    parser.add_argument("--random_collect_first", action="store_true")
    parser.add_argument("--only_random_collect", action="store_true", help="if true, only randomly collect")
    parser.add_argument("--random_clothes_idx_offset", type=int, default=0)
    parser.add_argument("--random_total_traj", type=int, default=120)
    return parser.parse_args()


def main():
    current = get_current_time()
    os.makedirs(os.path.join(OUTPUT_COMMON, current))
    with open(os.path.join(OUTPUT_COMMON, current, "command.txt"), "w") as f_obj:
        f_obj.write(" ".join(sys.argv))
    args = get_args()
    
    # parameters
    iter_n = int(args.iter_n)
    cudas: List[int] = args.cudas
    total_traj_valid = int(args.total_traj_valid)
    total_traj_train = int(args.total_traj_train)

    # train
    train_net_gpu_idx: List[int] = args.net_gpu
    step_offset_dict = dict(net=int(args.net_step_offset))

    process_dict: Dict[int, multiprocessing.Process] = {k: None for k in cudas}
    def get_available_cuda():
        while True:
            for cuda in cudas:
                p = process_dict[cuda]
                if p is None:
                    return cuda
                elif not p.is_alive():
                    p.join()
                    assert p.exitcode == 0, p.exitcode
                    process_dict[cuda] = None
                    return cuda
            time.sleep(1.)
    def wait_all_to_finish():
        for p in process_dict.values():
            if p is not None:
                p.join()
    def collect_random_all_and_train(output_basedir: str):
        data_output_basedir = os.path.join(output_basedir, "data")
        net_output_basedir = os.path.join(output_basedir, "net")
        os.makedirs(data_output_basedir, exist_ok=True)
        data_base_dirs: List[str] = [*(args.data_base_dirs), data_output_basedir]
        
        # collect
        for i in range(args.random_total_traj):
            ## magic code for unexpected interruption
            ## if i not in ([199] + [x for x in range(360, 480)]):
            ##    continue
            cuda = get_available_cuda()
            clothes_idx_str = utils.format_int((args.random_clothes_idx_offset + i) % TRAIN_CLOTHES_NUM, TRAIN_CLOTHES_NUM - 1)
            clothes_path = f"assets/clothes/train/{clothes_idx_str}/clothes.obj"
            p = multiprocessing.Process(
                target=collect_random, 
                args=(
                    cuda,
                    os.path.join(data_output_basedir, str(cuda), utils.format_int(i, args.random_total_traj - 1)),
                    clothes_path,
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        wait_all_to_finish()

        # train
        train_net(
            data_dirs=data_base_dirs,
            gpu_idx=train_net_gpu_idx,
            batch_size=int(args.net_batch_size),
            num_workers=int(args.net_num_workers),
            out_dir=net_output_basedir,
            ckpt_path=None,
            step_offset=0,
            opt_step=step_offset_dict["net"],
        )

        curr_net_path = find_ckpt(net_output_basedir)
        return curr_net_path, data_base_dirs

    def validate_all(output_basedir: str, net: str, sync: bool):
        for i in range(total_traj_valid):
            cuda = get_available_cuda()
            clothes_idx_str = utils.format_int(i % VALID_CLOTHES_NUM, VALID_CLOTHES_NUM - 1)
            clothes_path = f"assets/clothes/valid/{clothes_idx_str}/clothes.obj"
            p = multiprocessing.Process(
                target=validate, 
                args=(
                    cuda,
                    os.path.join(output_basedir, str(cuda), str(i)),
                    clothes_path, net, i, 
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        if sync:
            wait_all_to_finish()

    def collect_all(output_basedir: str, net: str, current_clothes_idx: int, sync: bool):
        for i in range(current_clothes_idx, total_traj_train + current_clothes_idx):
            cuda = get_available_cuda()
            clothes_idx_str = utils.format_int(i % TRAIN_CLOTHES_NUM, TRAIN_CLOTHES_NUM - 1)
            clothes_path = f"assets/clothes/train/{clothes_idx_str}/clothes.obj"
            p = multiprocessing.Process(
                target=collect, 
                args=(
                    cuda,
                    os.path.join(output_basedir, str(cuda), str(i)),
                    clothes_path, net,
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        if sync:
            wait_all_to_finish()
        return total_traj_train + current_clothes_idx

    def train_all(
        data_base_dirs: List[str], 
        new_net_dir: str, 
        curr_net_path: str, 
        step_offset_dict: dict, 
    ):
        train_net(
            data_dirs=data_base_dirs, 
            gpu_idx=train_net_gpu_idx, 
            batch_size=int(args.net_batch_size),
            num_workers=int(args.net_num_workers),
            out_dir=new_net_dir, 
            ckpt_path=curr_net_path, 
            step_offset=step_offset_dict["net"], 
            opt_step=args.opt_step,
        )
        step_offset_dict["net"] += args.opt_step

    if args.random_collect_first:
        curr_net_path, data_base_dirs = collect_random_all_and_train(random_base_dir(current))
        if args.only_random_collect:
            return
    else:
        curr_net_path = str(args.net_path)
        data_base_dirs: List[str] = args.data_base_dirs

        assert f'step_{step_offset_dict["net"]}' in curr_net_path, f'{step_offset_dict["net"]} {curr_net_path}'
    curr_train_clothes_idx = int(args.curr_train_clothes_idx)
    for iter_idx in range(iter_n):
        assert os.path.exists(curr_net_path), curr_net_path

        sync = False
        # valid
        if (not (iter_idx == 0 and bool(args.skip_first_valid))) and (iter_idx % int(args.valid_interval) == 0):
            curr_valid_base_dir = valid_base_dir(current, utils.format_int(iter_idx, iter_n))
            if not args.skip_all_valid:
                validate_all(curr_valid_base_dir, curr_net_path, sync=sync)

        # collect training data
        if not (iter_idx == 0 and bool(args.skip_first_collect)):
            curr_train_base_dir = train_base_dir(current, utils.format_int(iter_idx, iter_n))
            curr_train_clothes_idx = collect_all(curr_train_base_dir, curr_net_path, curr_train_clothes_idx, sync=sync)
            data_base_dirs.append(curr_train_base_dir)
        
        if not sync:
            wait_all_to_finish()

        # train
        new_net_dir = net_base_dir(current, utils.format_int(iter_idx, iter_n) + "/net")
        train_all(data_base_dirs, new_net_dir, curr_net_path, step_offset_dict)

        # update net
        curr_net_path = find_ckpt(new_net_dir)

        assert (curr_net_path is not None), f"{curr_net_path}"

    curr_valid_base_dir = valid_base_dir(current, utils.format_int(iter_n, iter_n))
    if not args.skip_all_valid:
        validate_all(curr_valid_base_dir, curr_net_path, sync=True)


if __name__ == "__main__":
    main()