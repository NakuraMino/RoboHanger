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
HANGER_NUM = 2
FAST_PASS = False # for debug
OUTPUT_COMMON = "outputs/iterative/insert"


def parse_asset_idx(asset_idx: int, clothes_num: int):
    clothes_idx = (asset_idx // HANGER_NUM) % clothes_num
    hanger_idx = asset_idx % HANGER_NUM
    # clothes_idx = asset_idx % TRAIN_CLOTHES_NUM
    # hanger_idx = (asset_idx // TRAIN_CLOTHES_NUM) % HANGER_NUM
    return clothes_idx, hanger_idx


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
    hanger_path: str,
    net1: str, 
    net2: str,
    sim_batch_size: int,
):
    cmd_str = (
        "python run/run_insert_sim_unet.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.insert_policy.ckpt.left={net1} "
        f"+overwrite.insert_policy.ckpt.right={net2} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_path={hanger_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_vis_path={hanger_path.replace('.obj', '_vis.obj')} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"glb_cfg.batch_size={sim_batch_size} "
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
    hanger_path: str,
    net1: str, 
    net2: str,
    sim_batch_size: int,
    seed: int,
):
    cmd_str = (
        "python run/run_insert_sim_unet.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.insert_policy.ckpt.left={net1} "
        f"+overwrite.insert_policy.ckpt.right={net2} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_path={hanger_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_vis_path={hanger_path.replace('.obj', '_vis.obj')} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"glb_cfg.batch_size={sim_batch_size} "
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
    gpu_idx: int, 
    out_dir: str, 
    ckpt_path: str, 
    step_offset: int,
    endpoint: str,
    opt_step: int,
    use_inverse_mask: bool,
):
    assert isinstance(data_dirs, list)
    cmd_str = (
        f"python run/run_insert_learn.py run.endpoint={endpoint} " + 
        f"'train.path.data_paths=[{', '.join(data_dirs)}]' " +
        f"'misc.hardware.gpuids=[{gpu_idx}]' " + 
        f"misc.step_offset={step_offset} " +
        f"train.path.exp_name=iterative " + 
        f"train.path.version_name={endpoint} " + 
        (f"train.path.ckpt={ckpt_path} " if ckpt_path is not None else "") + 
        f"pl.learn.valid.plot_dense_predict_num=1 " +
        f"train.cfg.max_steps={opt_step * 2} " + 
        f"train.cfg.ckpt_every_n_steps={opt_step * 2} " + 
        f"train.cfg.val_check_interval={opt_step // 2} " + 
        f"data.make.use_inverse_mask={use_inverse_mask} " + 
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
    hanger_path: str, 
    sim_batch_size: int,
    random_failure_rate: float, 
):
    cmd_str = (
        "python run/run_insert_sim_random.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_path={hanger_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_vis_path={hanger_path.replace('.obj', '_vis.obj')} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"glb_cfg.batch_size={sim_batch_size} "
        f"+overwrite.insert_policy.press.prob.fail={random_failure_rate} "
        f"+overwrite.insert_policy.press.prob.success={1. - random_failure_rate} "
        f"+overwrite.insert_policy.lift.prob.fail={random_failure_rate} "
        f"+overwrite.insert_policy.lift.prob.success={1. - random_failure_rate} "
        f"+overwrite.insert_policy.drag.prob.fail={random_failure_rate} "
        f"+overwrite.insert_policy.drag.prob.success={1. - random_failure_rate} "
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
    parser.add_argument("--iter_n", type=int, default=5)
    parser.add_argument("--cudas", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--total_traj_valid", type=int, default=48)
    parser.add_argument("--total_traj_train", type=int, default=48)
    parser.add_argument("--valid_sim_batch_size", type=int, default=15)
    parser.add_argument("--train_sim_batch_size", type=int, default=25)

    parser.add_argument("--valid_interval", type=int, default=1, help="only when iter_idx % valid_interval == 0, we perform validation")
    parser.add_argument("--skip_all_valid", action="store_true")
    parser.add_argument("--curr_train_asset_idx", type=int, default=0)
    parser.add_argument("--skip_first_valid", action="store_true")
    parser.add_argument("--skip_first_collect", action="store_true")
    parser.add_argument("--opt1_step", type=int, default=100, help="opt1_step * 2 = actual step")
    parser.add_argument("--opt2_step", type=int, default=50, help="opt2_step * 2 = actual step")

    parser.add_argument("--net1_step_offset", type=int, default=3000, help="3000 = 1500 * 2, 3000 % 200 == 0")
    parser.add_argument("--net2_step_offset", type=int, default=1500, help="1500 = 750 * 2, 1500 % 100 == 0")
    parser.add_argument("--net1_path", type=str, help="path/to/ckpt")
    parser.add_argument("--net2_path", type=str, help="path/to/ckpt")
    parser.add_argument("--data_base_dirs", type=str, nargs="*", default=[], help="offline data path")

    parser.add_argument("--random_collect_first", action="store_true")
    parser.add_argument("--only_random_collect", action="store_true", help="if true, only randomly collect")
    parser.add_argument("--random_asset_idx_offset", type=int, default=0)
    parser.add_argument("--random_total_traj", type=int, default=240)
    parser.add_argument("--random_sim_batch_size", type=int, default=50)
    parser.add_argument("--random_failure_rate", type=float, default=0.25)

    # policy options
    parser.add_argument("--disable_inverse_mask", action="store_true")
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
    train_net1_gpu_idx = cudas[0]
    train_net2_gpu_idx = cudas[-1]
    step_offset_dict = dict(net1=int(args.net1_step_offset), net2=int(args.net2_step_offset))
    assert step_offset_dict["net1"] % 2 == 0
    assert step_offset_dict["net2"] % 2 == 0

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
        net1_output_basedir = os.path.join(output_basedir, "net1")
        net2_output_basedir = os.path.join(output_basedir, "net2")
        os.makedirs(data_output_basedir, exist_ok=True)
        data_base_dirs: List[str] = [*(args.data_base_dirs), data_output_basedir]
        
        # collect
        for i in range(args.random_total_traj):
            ## magic code for unexpected interruption
            ## if i not in ([199] + [x for x in range(360, 480)]):
            ##    continue
            cuda = get_available_cuda()
            clothes_idx, hanger_idx = parse_asset_idx(int(args.random_asset_idx_offset) + i, TRAIN_CLOTHES_NUM)
            clothes_path = f"assets/clothes/train/{utils.format_int(clothes_idx, TRAIN_CLOTHES_NUM - 1)}/clothes.obj"
            hanger_path = f"assets/hanger/{utils.format_int(hanger_idx, HANGER_NUM - 1)}/hanger.obj"
            p = multiprocessing.Process(
                target=collect_random, 
                args=(
                    cuda,
                    os.path.join(data_output_basedir, str(cuda), utils.format_int(i, args.random_total_traj - 1)),
                    clothes_path, hanger_path, int(args.random_sim_batch_size), args.random_failure_rate, 
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        wait_all_to_finish()
        
        # train
        process_list: List[multiprocessing.Process] = []
        for i in range(2):
            if i == 0:
                p = multiprocessing.Process(
                    target=train_net, 
                    args=(
                        data_base_dirs, 
                        train_net1_gpu_idx, 
                        net1_output_basedir, 
                        None, 
                        0, 
                        "left",
                        step_offset_dict["net1"] // 2,
                        not args.disable_inverse_mask, 
                    ),
                    daemon=True,
                )
                p.start()
                process_list.append(p)
            elif i == 1:
                p = multiprocessing.Process(
                    target=train_net, 
                    args=(
                        data_base_dirs, 
                        train_net2_gpu_idx, 
                        net2_output_basedir, 
                        None, 
                        0, 
                        "right",
                        step_offset_dict["net2"] // 2,
                        not args.disable_inverse_mask,
                    ),
                    daemon=True,
                )
                p.start()
                process_list.append(p)
        for p in process_list:
            p.join()

        curr_net1_path = find_ckpt(net1_output_basedir)
        curr_net2_path = find_ckpt(net2_output_basedir)
        return curr_net1_path, curr_net2_path, data_base_dirs

    def validate_all(output_basedir: str, net1: str, net2: str, sync: bool):
        for i in range(total_traj_valid):
            cuda = get_available_cuda()
            clothes_idx, hanger_idx = parse_asset_idx(i, VALID_CLOTHES_NUM)
            clothes_path = f"assets/clothes/valid/{utils.format_int(clothes_idx, VALID_CLOTHES_NUM - 1)}/clothes.obj"
            hanger_path = f"assets/hanger/{utils.format_int(hanger_idx, HANGER_NUM - 1)}/hanger.obj"
            p = multiprocessing.Process(
                target=validate, 
                args=(
                    cuda,
                    os.path.join(output_basedir, str(cuda), str(i)),
                    clothes_path, hanger_path, net1, net2, int(args.valid_sim_batch_size), i
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        if sync:
            wait_all_to_finish()

    def collect_all(output_basedir: str, net1: str, net2: str, curr_train_asset_idx: int, sync: bool):
        for i in range(curr_train_asset_idx, total_traj_train + curr_train_asset_idx):
            cuda = get_available_cuda()
            clothes_idx, hanger_idx = parse_asset_idx(i, TRAIN_CLOTHES_NUM)
            clothes_path = f"assets/clothes/train/{utils.format_int(clothes_idx, TRAIN_CLOTHES_NUM - 1)}/clothes.obj"
            hanger_path = f"assets/hanger/{utils.format_int(hanger_idx, HANGER_NUM - 1)}/hanger.obj"
            p = multiprocessing.Process(
                target=collect, 
                args=(
                    cuda,
                    os.path.join(output_basedir, str(cuda), str(i)),
                    clothes_path, hanger_path, net1, net2, int(args.train_sim_batch_size), 
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        if sync:
            wait_all_to_finish()
        return total_traj_train + curr_train_asset_idx

    def train_all(
        data_base_dirs: List[str], 
        new_net1_dir: str, 
        new_net2_dir: str, 
        curr_net1_path: str, 
        curr_net2_path: str, 
        step_offset_dict: dict, 
    ):
        process_list: List[multiprocessing.Process] = []
        for i in range(2):
            if i == 0:
                p = multiprocessing.Process(
                    target=train_net, 
                    args=(
                        data_base_dirs, 
                        train_net1_gpu_idx, 
                        new_net1_dir, 
                        curr_net1_path, 
                        step_offset_dict["net1"], 
                        "left",
                        int(args.opt1_step),
                        not args.disable_inverse_mask,
                    ),
                    daemon=True,
                )
                p.start()
                process_list.append(p)
                step_offset_dict["net1"] += int(args.opt1_step) * 2
            elif i == 1:
                p = multiprocessing.Process(
                    target=train_net, 
                    args=(
                        data_base_dirs, 
                        train_net2_gpu_idx, 
                        new_net2_dir, 
                        curr_net2_path, 
                        step_offset_dict["net2"], 
                        "right",
                        int(args.opt2_step),
                        not args.disable_inverse_mask,
                    ),
                    daemon=True,
                )
                p.start()
                process_list.append(p)
                step_offset_dict["net2"] += int(args.opt2_step) * 2
        for p in process_list:
            p.join()
    
    if args.random_collect_first:
        curr_net1_path, curr_net2_path, data_base_dirs = collect_random_all_and_train(random_base_dir(current))
        if args.only_random_collect:
            return
    else:
        curr_net1_path = str(args.net1_path)
        curr_net2_path = str(args.net2_path)
        data_base_dirs: List[str] = args.data_base_dirs

        if not FAST_PASS:
            assert f'step_{step_offset_dict["net1"]}' in curr_net1_path, f'{step_offset_dict["net1"]} {curr_net1_path}'
            assert f'step_{step_offset_dict["net2"]}' in curr_net2_path, f'{step_offset_dict["net2"]} {curr_net2_path}'

    curr_train_asset_idx = int(args.curr_train_asset_idx)
    for iter_idx in range(iter_n):
        assert os.path.exists(curr_net1_path), curr_net1_path
        assert os.path.exists(curr_net2_path), curr_net2_path

        sync = False
        # valid
        if (not (iter_idx == 0 and bool(args.skip_first_valid))) and (iter_idx % int(args.valid_interval) == 0):
            curr_valid_base_dir = valid_base_dir(current, utils.format_int(iter_idx, iter_n))
            if not args.skip_all_valid:
                validate_all(curr_valid_base_dir, curr_net1_path, curr_net2_path, sync=sync)

        # collect training data
        if not (iter_idx == 0 and bool(args.skip_first_collect)):
            curr_train_base_dir = train_base_dir(current, utils.format_int(iter_idx, iter_n))
            curr_train_asset_idx = collect_all(curr_train_base_dir, curr_net1_path, curr_net2_path, curr_train_asset_idx, sync=sync)
            data_base_dirs.append(curr_train_base_dir)
        
        if not sync:
            wait_all_to_finish()
        
        # train
        new_net1_dir = net_base_dir(current, utils.format_int(iter_idx, iter_n) + "/net1")
        new_net2_dir = net_base_dir(current, utils.format_int(iter_idx, iter_n) + "/net2")
        train_all(data_base_dirs, new_net1_dir, new_net2_dir, curr_net1_path, curr_net2_path, step_offset_dict)

        # update net
        curr_net1_path = find_ckpt(new_net1_dir)
        curr_net2_path = find_ckpt(new_net2_dir)

        assert (curr_net1_path is not None) and (curr_net2_path is not None), f"{curr_net1_path} {curr_net2_path}"

    curr_valid_base_dir = valid_base_dir(current, utils.format_int(iter_n, iter_n))
    if not args.skip_all_valid:
        validate_all(curr_valid_base_dir, curr_net1_path, curr_net2_path, sync=True)


if __name__ == "__main__":
    main()