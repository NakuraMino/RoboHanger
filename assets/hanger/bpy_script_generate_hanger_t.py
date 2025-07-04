import bpy
import numpy as np
import os
import trimesh
import trimesh.transformations as tra

# path
blend_file_path = bpy.data.filepath
curr_dir = os.path.dirname(blend_file_path)
cache_dir = os.path.join(curr_dir, "cache")
os.makedirs(os.path.join(curr_dir, "final_part_to_print"), exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# args
import_background_mesh = False
prepare_for_3d_print = True
add_rack = True

# delete all objects
for obj in bpy.data.objects:
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.select_set(True)
bpy.ops.object.delete()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# background
if import_background_mesh:
    bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble_old.stl"))
    obj = bpy.context.object
    obj.hide_set(True)

# create an arm
xy = np.array([
    [ 0.000, -0.10],
    [ 0.000,  0.00],
    [ 0.000,  0.10],
])
for i in range(len(xy) - 1):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64)
    obj = bpy.context.object
    obj.name = f"arm{i}"
    obj.scale = (0.007, 0.007, np.linalg.norm(xy[i+1] - xy[i]) / 2, )
    obj.location = (*((xy[i+1] + xy[i]) / 2), 0.0)
    obj.rotation_euler = (np.pi / 2, 0, -np.arctan2(*(xy[i+1] - xy[i])), )
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4)
    obj = bpy.context.object
    obj.name = f"ico{i}"
    obj.scale = (0.007, 0.007, 0.007)
    obj.location = (*(xy[i+1]), 0)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    obj = bpy.data.objects[f"arm{i}"]
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects[f"ico{i}"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_mesh.stl(filepath=os.path.join(cache_dir, f"armico{i}.stl"), use_selection=True)
    
    bpy.data.objects[f"arm{i}"].hide_set(True)
    bpy.data.objects[f"ico{i}"].hide_set(True)

for i in range(len(xy) - 1):
    bpy.ops.import_mesh.stl(filepath=os.path.join(cache_dir, f"armico{i}.stl"))

def mesh_union(infile1, infile2, outfile):
    bpy.ops.import_mesh.stl(filepath=infile1)
    obj = bpy.context.object
    obj.name = "infile1"
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.import_mesh.stl(filepath=infile2)
    obj = bpy.context.object
    obj.name = "infile2"
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["infile1"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_mesh.stl(filepath=outfile, use_selection=True)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["infile1"].select_set(True)
    bpy.data.objects["infile2"].select_set(True)
    bpy.ops.object.delete()
    
for i in range(1, len(xy) - 1):
    if i == 1:
        prev = f"armico0.stl"
    else:
        prev = f"armico0{i-1}.stl"
    mesh_union(
        os.path.join(cache_dir, prev),
        os.path.join(cache_dir, f"armico{i}.stl"),
        os.path.join(cache_dir, f"armico0{i}.stl"),
    )

bpy.ops.import_mesh.stl(filepath=os.path.join(cache_dir, f"armico0{len(xy) - 2}.stl"))
obj = bpy.context.object
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "arm.stl"), use_selection=True)

for i in range(len(xy) - 1):
    bpy.data.objects[f"armico{i}"].hide_set(True)
bpy.data.objects[f"armico0{len(xy) - 2}"].hide_set(True)

# create two arms
bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "arm.stl"))
obj = bpy.context.object
obj.name = "arm_left"
obj.rotation_euler = (0, 0, np.deg2rad(-110))
obj.location = (0.09, -0.05, 0.00)

bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "arm.stl"))
obj = bpy.context.object
obj.name = "arm_right"
obj.rotation_euler = (np.pi, 0, np.deg2rad(-70))
obj.location = (-0.09, -0.05, 0.00)

bpy.data.objects["arm_left"].hide_set(True)
bpy.data.objects["arm_right"].hide_set(True)

# import hook
bpy.ops.curve.primitive_bezier_curve_add()
obj = bpy.context.object
obj.name = "bezier"
curve_data = obj.data
for spline in curve_data.splines:
    for i, point in enumerate(spline.bezier_points):
        print(i)
        # Example: Move each control point to a new location
        point.co = [(+0.0012, 0.0241, 0.0000), (-0.0351, 0.0781, 0.0000)][i]  # Set new coordinates (x, y, z)
        point.handle_left = [(-0.0432, 0.0025, 0.0000), (-0.0046, 0.1546, 0.0000)][i]  # Optional: Adjust handle type
        point.handle_right = [(+0.0729, 0.0592, 0.0000), (-0.0524, 0.0347, 0.0000)][i]  # Optional: Adjust handle type

curve_data.resolution_u = 100
curve_data.bevel_depth = 0.006
curve_data.bevel_resolution = 32
curve_data.use_fill_caps = True

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "hook.stl"), use_selection=True)
bpy.data.objects["bezier"].hide_set(True)

bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "hook.stl"))
obj = bpy.context.object
obj.name = "hook"
obj.hide_set(True)

# add rack
if add_rack:
    bpy.ops.mesh.primitive_cylinder_add(vertices=64)
    obj = bpy.context.object
    obj.name = "rack"
    obj.scale = (0.007, 0.007, 0.18397)
    obj.rotation_euler = (0, np.pi / 2, 0)
    obj.location = (0, -0.084202, 0)
    obj.hide_set(True)

# connecting part
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
obj.name = "final"
obj.scale = (0.008, 0.030, 0.008)
obj.location = (0, 0.003, 0)

obj = bpy.data.objects["final"]
boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
boolean_modifier.operation = 'UNION'
boolean_modifier.object = bpy.data.objects["arm_left"]
bpy.ops.object.modifier_apply(modifier="Boolean")
boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
boolean_modifier.operation = 'UNION'
boolean_modifier.object = bpy.data.objects["arm_right"]
bpy.ops.object.modifier_apply(modifier="Boolean")
boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
boolean_modifier.operation = 'UNION'
boolean_modifier.object = bpy.data.objects["hook"]
if add_rack:
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["rack"]
    bpy.ops.object.modifier_apply(modifier="Boolean")

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble.stl"), use_selection=True)
# bpy.ops.export_scene.obj(filepath=os.path.join(curr_dir, "hanger_vis.obj"), use_selection=True)
obj.hide_set(True)

# add grip pad for sdf calculation
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
obj.name = "grip1"
obj.scale = (0.008, 0.008, 0.030)
obj.location = (-0.015, 0.0, 0.023)
obj.hide_set(True)

bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
obj.name = "grip2"
obj.scale = (0.008, 0.008, 0.030)
obj.location = (+0.015, 0.0, 0.023)
obj.hide_set(True)

# voxelize and export
bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble.stl"))
obj = bpy.context.object
obj.name = "hanger"
boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
boolean_modifier.operation = 'UNION'
boolean_modifier.object = bpy.data.objects["grip1"]
bpy.ops.object.modifier_apply(modifier="Boolean")
boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
boolean_modifier.operation = 'UNION'
boolean_modifier.object = bpy.data.objects["grip2"]
bpy.ops.object.modifier_apply(modifier="Boolean")
voxel_modifier = obj.modifiers.new(name="voxel", type="REMESH")
voxel_modifier.voxel_size=0.003

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble_voxel.stl"), use_selection=True)
# bpy.ops.export_scene.obj(filepath=os.path.join(curr_dir, "hanger.obj"), use_selection=True)
obj.hide_set(True)

# post process
mesh = trimesh.load_mesh(os.path.join(curr_dir, "hanger_assemble.stl"), force="mesh")
# mesh.apply_transform(tra.euler_matrix(np.pi / 2, 0., 0.))
mesh.export(os.path.join(curr_dir, "hanger_vis.obj"))

mesh = trimesh.load_mesh(os.path.join(curr_dir, "hanger_assemble_voxel.stl"), force="mesh")
# mesh.apply_transform(tra.euler_matrix(np.pi / 2, 0., 0.))
mesh.export(os.path.join(curr_dir, "hanger.obj"))

# prepare for 3D print
if prepare_for_3d_print:
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "mask_left"
    obj.scale = (0.2, 0.2, 0.2)
    obj.location = (-0.20, -0.13, 0.0)
    obj.rotation_euler = (0, 0, np.deg2rad(+20))
    obj.hide_set(True)

    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "mask_right"
    obj.scale = (0.2, 0.2, 0.2)
    obj.location = (+0.20, -0.13, 0.0)
    obj.rotation_euler = (0, 0, np.deg2rad(-20))
    obj.hide_set(True)

    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "mask_rack"
    obj.scale = (0.02, 0.02, 0.02)
    obj.location = (0.0, -0.08, 0.0)
    obj.hide_set(True)

    bpy.ops.mesh.primitive_cylinder_add()
    obj = bpy.context.object
    obj.name = "hole_left"
    obj.scale = (0.002, 0.002, 0.015)
    obj.location = (-0.022, -0.0251, 0.00)
    obj.rotation_euler = (0, np.deg2rad(90), np.deg2rad(+20))
    obj.hide_set(True)

    bpy.ops.mesh.primitive_cylinder_add()
    obj = bpy.context.object
    obj.name = "hole_right"
    obj.scale = (0.002, 0.002, 0.015)
    obj.location = (+0.022, -0.0251, 0.00)
    obj.rotation_euler = (0, np.deg2rad(90), np.deg2rad(-20))
    obj.hide_set(True)
    
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "hole_rack"
    obj.scale = (0.020, 0.010, 0.002)
    obj.location = (0, -0.075, 0)
    obj.hide_set(True)

    bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble.stl"))
    obj = bpy.context.object
    obj.name = "hook_3dp"
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["mask_rack"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["mask_left"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["mask_right"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["hole_left"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["hole_right"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "final_part_to_print", "hook.stl"), use_selection=True)
    # obj.hide_set(True)
    
    bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble.stl"))
    obj = bpy.context.object
    obj.name = "arm_left_3dp"
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["hole_rack"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'INTERSECT'
    boolean_modifier.object = bpy.data.objects["mask_left"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["hole_left"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "final_part_to_print", "arm_left.stl"), use_selection=True)
    # obj.hide_set(True)
    
    bpy.ops.import_mesh.stl(filepath=os.path.join(curr_dir, "hanger_assemble.stl"))
    obj = bpy.context.object
    obj.name = "arm_right_3dp"
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["hole_rack"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'INTERSECT'
    boolean_modifier.object = bpy.data.objects["mask_right"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'DIFFERENCE'
    boolean_modifier.object = bpy.data.objects["hole_right"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.ops.export_mesh.stl(filepath=os.path.join(curr_dir, "final_part_to_print", "arm_right.stl"), use_selection=True)
    # obj.hide_set(True)