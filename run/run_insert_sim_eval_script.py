import subprocess
import multiprocessing
import os
import sys
from typing import Dict
import argparse
import time

import robohang.common.utils as utils


TRAIN_CLOTHES_NUM = 120
VALID_CLOTHES_NUM = 24
HANGER_NUM = 2


def parse_asset_idx(asset_idx: int, clothes_num: int):
    clothes_idx = (asset_idx // HANGER_NUM) % clothes_num
    hanger_idx = asset_idx % HANGER_NUM
    # clothes_idx = asset_idx % TRAIN_CLOTHES_NUM
    # hanger_idx = (asset_idx // TRAIN_CLOTHES_NUM) % HANGER_NUM
    return clothes_idx, hanger_idx


def validate_sac(
    cuda: int, 
    output_dir: str,
    clothes_path: str,
    hanger_path: str,
    net: str, 
    sim_batch_size: int,
    sim_gpu_memory: int, 
    seed: int,
):
    cmd_str = (
        "python run/run_insert_sim_sac.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.insert_policy.ckpt={net} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_path={hanger_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_vis_path={hanger_path.replace('.obj', '_vis.obj')} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"setup.taichi.device_memory_GB={sim_gpu_memory} "
        f"glb_cfg.batch_size={sim_batch_size} "
        f"glb_cfg.seed={seed} "
    )
    print(f"cmd_str\n{cmd_str}")
    ret = subprocess.run(cmd_str, shell=True)
    if ret.returncode != 0:
        exit(ret.returncode)


def validate_imi(
    cuda: int, 
    output_dir: str,
    clothes_path: str,
    hanger_path: str,
    net1: str, 
    net2: str, 
    sim_batch_size: int,
    sim_gpu_memory: int, 
    seed: int,
):
    cmd_str = (
        "python run/run_insert_sim_imi.py output.collect_mode=True output.total_traj=1 "
        f"+overwrite.insert_policy.ckpt.left={net1} "
        f"+overwrite.insert_policy.ckpt.right={net2} "
        f"+overwrite.sim_env.asset.garment.mesh_path={clothes_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_path={hanger_path} "
        f"+overwrite.sim_env.asset.hanger.mesh_vis_path={hanger_path.replace('.obj', '_vis.obj')} "
        f"hydra.run.dir={output_dir} "
        f"setup.cuda={cuda} "
        f"setup.taichi.device_memory_GB={sim_gpu_memory} "
        f"glb_cfg.batch_size={sim_batch_size} "
        f"glb_cfg.seed={seed} "
    )
    print(f"cmd_str\n{cmd_str}")
    ret = subprocess.run(cmd_str, shell=True)
    if ret.returncode != 0:
        exit(ret.returncode)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudas", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--total_traj_valid", type=int, default=48)
    parser.add_argument("--valid_sim_batch_size", type=int, default=5)
    parser.add_argument("--valid_sim_gpu_memory", type=int, default=5)
    parser.add_argument("--net_path", type=str, help="path/to/ckpt")
    parser.add_argument("--net1_path", type=str, help="path/to/ckpt")
    parser.add_argument("--net2_path", type=str, help="path/to/ckpt")
    parser.add_argument("--output_basedir", type=str, required=True)
    parser.add_argument("--method", required=True, choices=["sac", "imi"])
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_basedir)
    with open(os.path.join(args.output_basedir, "command.txt"), "w") as f_obj:
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

    def validate_all(output_basedir: str, sync: bool):
        for i in range(args.total_traj_valid):
            cuda = get_available_cuda()
            clothes_idx, hanger_idx = parse_asset_idx(i, VALID_CLOTHES_NUM)
            clothes_path = f"assets/clothes/valid/{utils.format_int(clothes_idx, VALID_CLOTHES_NUM - 1)}/clothes.obj"
            hanger_path = f"assets/hanger/{utils.format_int(hanger_idx, HANGER_NUM - 1)}/hanger.obj"
            if args.method == "sac":
                assert args.net_path is not None
                p = multiprocessing.Process(
                    target=validate_sac, 
                    args=(
                        cuda,
                        os.path.join(output_basedir, str(cuda), str(i)),
                        clothes_path, hanger_path, args.net_path, 
                        int(args.valid_sim_batch_size), 
                        int(args.valid_sim_gpu_memory), 
                        i
                    ),
                    daemon=True,
                )
            elif args.method == "imi":
                assert args.net1_path is not None and args.net2_path is not None
                p = multiprocessing.Process(
                    target=validate_imi, 
                    args=(
                        cuda,
                        os.path.join(output_basedir, str(cuda), str(i)),
                        clothes_path, hanger_path, args.net1_path, args.net2_path, 
                        int(args.valid_sim_batch_size), 
                        int(args.valid_sim_gpu_memory), 
                        i
                    ),
                    daemon=True,
                )
            else: raise ValueError(args.method)
            p.start()
            process_dict[cuda] = p
        if sync:
            wait_all_to_finish()
    
    validate_all(args.output_basedir, True)


if __name__ == "__main__":
    main()