import subprocess
import multiprocessing
import os
import sys
from typing import Dict
import argparse
import time
import datetime

import robohang.common.utils as utils


TRAIN_CLOTHES_NUM = 120
VALID_CLOTHES_NUM = 24
HANGER_NUM = 2


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def parse_asset_idx(asset_idx: int, clothes_num: int):
    clothes_idx = (asset_idx // HANGER_NUM) % clothes_num
    hanger_idx = asset_idx % HANGER_NUM
    # clothes_idx = asset_idx % TRAIN_CLOTHES_NUM
    # hanger_idx = (asset_idx // TRAIN_CLOTHES_NUM) % HANGER_NUM
    return clothes_idx, hanger_idx


def validate(
    cuda: int, 
    output_dir: str,
    clothes_path: str,
    hanger_path: str,
    sim_batch_size: int,
    device_memory_GB: int,
    ckpt: str,
    side_view: bool, 
    action_w: str, 
    policy: str,
):
    cmd_str = (
        f"python run/run_insert_sim_e2e_eval.py output.collect_mode={not side_view} output.total_traj=1 "
        f"glb_cfg.batch_size={sim_batch_size} setup.taichi.device_memory_GB={device_memory_GB} "
        f"insert_policy.act.ckpt={ckpt} insert_policy.dfp.ckpt={ckpt} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_path={hanger_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_vis_path={hanger_path.replace('.obj', '_vis.obj')} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"insert_policy.act.action_w={action_w} insert_policy.dfp.action_w={action_w} "
        f"insert_policy.name={policy} "
    )
    print(f"cmd_str\n{cmd_str}")
    ret = subprocess.run(cmd_str, shell=True)
    if ret.returncode != 0:
        exit(ret.returncode)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudas", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--total_traj_valid", type=int, default=48)
    parser.add_argument("--sim_batch_size", type=int, default=25)
    parser.add_argument("--device_memory_GB", type=int, default=12)
    parser.add_argument("--side_view", action="store_true")
    parser.add_argument("--action_w", type=str, default="-0.3")
    parser.add_argument("--output_basedir", type=str, default="outputs/exp_e2e")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--policy", type=str, choices=["act", "dfp"], required=True)
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = os.path.join(args.output_basedir, args.policy, get_current_time())
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "command.txt"), "w") as f_obj:
        f_obj.write(" ".join(sys.argv))

    process_dict: Dict[int, multiprocessing.Process] = {k: None for k in args.cudas}
    def get_available_cuda():
        while True:
            for cuda in args.cudas:
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

    def evaluate_all(output_basedir: str, sync: bool):
        for i in range(args.total_traj_valid):
            cuda = get_available_cuda()
            clothes_idx, hanger_idx = parse_asset_idx(i, TRAIN_CLOTHES_NUM)
            clothes_path = f"assets/clothes/valid/{utils.format_int(clothes_idx, VALID_CLOTHES_NUM - 1)}/clothes.obj"
            hanger_path = f"assets/hanger/{utils.format_int(hanger_idx, HANGER_NUM - 1)}/hanger.obj"
            p = multiprocessing.Process(
                target=validate, 
                args=(
                    cuda,
                    os.path.join(output_basedir, str(cuda), str(i)),
                    clothes_path, hanger_path,
                    args.sim_batch_size, args.device_memory_GB, args.ckpt,
                    args.side_view, args.action_w, args.policy, 
                ),
                daemon=True,
            )
            p.start()
            process_dict[cuda] = p
        if sync:
            wait_all_to_finish()
    
    evaluate_all(output_dir, True)


if __name__ == "__main__":
    main()