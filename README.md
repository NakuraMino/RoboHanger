# Hierarchical RoboHanger

## install

```
pixi init
pixi install
```

```
python run/run_insert_sim_random.py output.collect_mode=true output.total_traj=1 \
+overwrite.sim_env.asset.garment.mesh_path=assets/clothes/train/000/clothes.obj \
+overwrite.sim_env.asset.hanger.mesh_path=assets/hanger/0/hanger.obj \
+overwrite.sim_env.asset.hanger.mesh_vis_path=assets/hanger/0/hanger_vis.obj \
setup.cuda=1
```