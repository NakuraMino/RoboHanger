# RoboHanger
Source code for [RoboHanger](https://pku-epic.github.io/RoboHanger/).

## install 
You may have to install a specific version of Pytorch depend on your machine.
```
git clone git@github.com:chen01yx/RoboHanger_code.git --recurse-submodules
cd RoboHanger_code
git submodule update --init --recursive
conda activate YOUR_ENV
pip install -e .
```

see https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html to make sapien work. 

## usage
Run the full training:
```
python run/run_insert_sim_iterative.py --cudas 0 1 2 3 4 5 6 7 --random_collect_first
```

Run the heuristic policy:
```
python run/run_insert_sim_random.py output.collect_mode=false output.total_traj=1 \
+overwrite.sim_env.asset.garment.mesh_path=assets/clothes/train/000/clothes.obj \
+overwrite.sim_env.asset.hanger.mesh_path=assets/hanger/0/hanger.obj \
+overwrite.sim_env.asset.hanger.mesh_vis_path=assets/hanger/0/hanger_vis.obj \
setup.cuda=0
```

Run the U-Net policy:
```
python run/run_insert_sim_unet.py output.collect_mode=false output.total_traj=1 \
+overwrite.insert_policy.ckpt.left=xxx \
+overwrite.insert_policy.ckpt.right=xxx \
+overwrite.sim_env.asset.garment.mesh_path=assets/clothes/valid/00/clothes.obj \
+overwrite.sim_env.asset.hanger.mesh_path=assets/hanger/0/hanger.obj \
+overwrite.sim_env.asset.hanger.mesh_vis_path=assets/hanger/0/hanger_vis.obj \
setup.cuda=1
```