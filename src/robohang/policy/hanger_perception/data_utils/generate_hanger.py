import os
import bpy
import numpy as np
import trimesh
import trimesh.transformations as tra
import omegaconf

### arm configurations 
TRI_HEIGHT_STD = 0.08
TRI_HEIGHT_OVERALL_SHIFT = -0.02
TRI_WIDTH_STD = 0.1879385241571817

TRI_HEIGHT_VAR = 0.03
TRI_LEFT_VAR = 0.03
TRI_RIGHT_VAR = 0.03

MAX_ARM_POINT = 4
ARM_RADIUS_STD = 0.010
ARM_RADIUS_VAR = 0.003

### rack configurations
RACK_RADIUS_STD = 0.005
RACK_RADIUS_VAR = 0.001

RACK_HEIGHT_VAR = 0.02

### hook configurations
HOOK_POSITIONS_STD = [(-0.0012, 0.0241, 0.0000), (0.0351, 0.0781, 0.0000)]
HOOK_HANDLE_LEFT_STD = [(0.0432, 0.0025, 0.0000), (0.0046, 0.1546, 0.0000)]
HOOK_HANDLE_LEFT_VAR = [(0.02, 0.001, 0.0000), (0.002, 0.07, 0.0000)]
HOOK_HANDLE_RIGHT_STD = [(-0.0729, 0.0592, 0.0000), (0.0524, 0.0347, 0.0000)]
HOOK_HANDLE_RIGHT_VAR = [(-0.036, 0.029, 0.0000), (0.026, 0.017, 0.0000)]

HOOK_BEVEL_DEPTH_STD = 0.013
HOOK_BEVEL_DEPTH_VAR = 0.002

HOOK_RESOLUTION_U_STD = 100
HOOK_BEVEL_RESOLUTION_STD = 32

HOOK_USE_FILL_CAPS_STD = True

### link configurations
LINK_SCALE_STD = (0.02, 0.03, 0.02)
LINK_SCALE_VAR = (0.004, 0.008, 0.004)
LINK_LOCATION_STD = (0, 0.003, 0)


def rand(left : np.float32, right : np.float32) -> np.float32:
    """Return random float between in [left, right]"""
    return np.random.rand() * (right-left) + left


def clear_stl(target_dir : str):
    """ clear auxiliary stl files generated """

    path = target_dir
    if os.path.isdir(path):
        for dir in os.listdir(path):
            new_dir = os.path.join(target_dir, dir)
            clear_stl(new_dir)
    elif os.path.split(path)[-1].split('.')[-1] == "stl" :
        os.remove(path)


def generate_one_hanger(
        target_dir : str,
        add_rack = False,
):
    """ generate one hanger and store its obj file in 'target_dir' """

    ## create arms
    def get_arm_points(height : np.float32, width : np.float32) -> np.ndarray:

        """get point coordinates, assume arm is between (0, 0, 0) and (width, height, 0)"""

        num_point = np.random.randint(1, MAX_ARM_POINT+1)

        h_proportions = np.zeros(num_point)
        v_proportions = np.zeros(num_point)

        for i in range(num_point):
            h_proportions[i] = rand(1, 2)
            v_proportions[i] = rand(1, 2)

        h_proportions = h_proportions / h_proportions.sum()
        v_proportions = v_proportions / v_proportions.sum()

        coordinates = np.zeros([len(h_proportions)+1, 3], dtype=np.float32)
        coordinates[1:, 0] = width * h_proportions
        coordinates[1:, 1] = height * v_proportions
        
        for i in range(1, num_point+1):
            coordinates[i, :] += coordinates[i-1, :]

        coordinates[:, 1] += TRI_HEIGHT_OVERALL_SHIFT
        return coordinates

    delta_height = TRI_HEIGHT_VAR * rand(-1.0, 1.0)
    triangle_height = TRI_HEIGHT_STD + delta_height
    
    delta_left = TRI_LEFT_VAR * rand(-1.0, 1.0)
    delta_right = TRI_RIGHT_VAR * rand(-1.0, 1.0)
    triangle_left = TRI_WIDTH_STD + delta_left
    triangle_right = TRI_WIDTH_STD + delta_right

    right_arm_points = get_arm_points(-triangle_height, triangle_right)
    left_arm_points = get_arm_points(-triangle_height, -triangle_left)

    delta_radius = rand(-1.0, 1.0) * ARM_RADIUS_VAR
    arm_radius = ARM_RADIUS_STD + delta_radius

    def create_arm_obj(
            arm_points : np.ndarray,
            arm_name : str
        ) -> bpy.types.Object:

        bpy.ops.object.select_all(action='DESELECT')
        arm_obj = None

        # add cylinder
        for i in range(1, arm_points.shape[0]):
            pos_0 = arm_points[i-1]
            pos_1 = arm_points[i]

            vec = pos_1 - pos_0
            len = np.linalg.norm(vec) + 2 * arm_radius
            theta = np.arctan(vec[0] / (-vec[1]))
            bpy.ops.mesh.primitive_cylinder_add(radius = arm_radius,
                                                depth = len, 
                                                scale = (1.0, 1.0, 1.0), 
                                                rotation = (np.pi / 2, 0, theta), 
                                                location = (pos_0+pos_1)/2)
            cylinder_obj = bpy.context.object
            cylinder_obj.select_set(False)

            if arm_obj == None:
                arm_obj = cylinder_obj
            else:
                # concatenate
                boolean_modifier = arm_obj.modifiers.new(name=f"cylinder_{i}", type="BOOLEAN")
                boolean_modifier.operation = 'UNION'
                boolean_modifier.object = cylinder_obj
                bpy.context.view_layer.objects.active = arm_obj
                bpy.ops.object.modifier_apply(modifier=f"cylinder_{i}")

        arm_obj.name = arm_name
        return arm_obj

    left_arm_obj = create_arm_obj(left_arm_points, "left_arm")
    right_arm_obj = create_arm_obj(right_arm_points, "right_arm")


    ## create rack
    if add_rack:
        delta_radius = RACK_RADIUS_VAR * rand(-1.0, 1.0)
        rack_radius = RACK_RADIUS_STD + delta_radius
        delta_y = RACK_HEIGHT_VAR * rand(0, 1.0)
        loc_y = -triangle_height+TRI_HEIGHT_OVERALL_SHIFT+delta_y
        left = triangle_left - delta_y / (left_arm_points[-2, 1] - left_arm_points[-1, 1]) \
                                       * (left_arm_points[-2, 0] - left_arm_points[-1, 0])
        right = triangle_right - delta_y / (right_arm_points[-2, 1] - right_arm_points[-1, 1]) \
                                         * (right_arm_points[-1, 0] - right_arm_points[-2, 0])
        rack_len = left + right
        loc_x = (right - left) / 2
        bpy.ops.mesh.primitive_cylinder_add(vertices = 64,
                                            depth = rack_len,
                                            radius = rack_radius,
                                            scale = (1.0, 1.0, 1.0),
                                            rotation = (0, np.pi / 2, 0),
                                            location = (loc_x, loc_y, 0))
        obj = bpy.context.object
        obj.name = "rack"
        obj.hide_set(True)


    ## create hook
    bpy.ops.curve.primitive_bezier_curve_add()
    hook_obj = bpy.context.object

    # Access the curve data
    curve_data = hook_obj.data

    # Iterate through the splines in the curve (there can be more than one)
    for spline in curve_data.splines:
        # Iterate through the control points in the spline
        for i, point in enumerate(spline.bezier_points):
            # Example: Move each control point to a new location
            point.co = HOOK_POSITIONS_STD[i]  # Set new coordinates (x, y, z)

            delta_handle_left = np.array(HOOK_HANDLE_LEFT_VAR[i]) * rand(-1.0, 1.0)
            point.handle_left = HOOK_HANDLE_LEFT_STD[i] + delta_handle_left
            delta_handle_right = np.array(HOOK_HANDLE_RIGHT_VAR[i]) * rand(-1.0, 1.0)
            point.handle_right = HOOK_HANDLE_RIGHT_STD[i] + delta_handle_right

    delta_depth = HOOK_BEVEL_DEPTH_VAR * rand(-1.0, 1.0)
    curve_data.bevel_depth = HOOK_BEVEL_DEPTH_STD + delta_depth

    curve_data.resolution_u = HOOK_RESOLUTION_U_STD
    curve_data.bevel_resolution = HOOK_BEVEL_RESOLUTION_STD
    curve_data.use_fill_caps = HOOK_USE_FILL_CAPS_STD

    bpy.ops.object.select_all(action='DESELECT')
    if rand(0, 1.0) < 0.5:
        hook_obj.rotation_euler = (0, np.pi, 0)
    hook_obj.select_set(True)
    bpy.ops.export_mesh.stl(filepath=os.path.join(target_dir, "hook.stl"), use_selection=True)


    ## create link
    bpy.ops.mesh.primitive_cube_add()
    link_obj = bpy.context.object
    link_obj.name = "link"

    delta_scale = np.array(LINK_SCALE_VAR) * rand(-1.0, 1.0)
    link_obj.scale = np.array(LINK_SCALE_STD) + delta_scale
    link_obj.location = LINK_LOCATION_STD


    ## concatenate everthing together
    obj = bpy.data.objects["link"]
    bpy.context.view_layer.objects.active = obj

    boolean_modifier = obj.modifiers.new(name="bool_1", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["left_arm"]
    bpy.ops.object.modifier_apply(modifier="bool_1")

    boolean_modifier = obj.modifiers.new(name="bool_2", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["right_arm"]
    bpy.ops.object.modifier_apply(modifier="bool_2")

    if add_rack:
        boolean_modifier = obj.modifiers.new(name="bool_3", type="BOOLEAN")
        boolean_modifier.operation = 'UNION'
        boolean_modifier.object = bpy.data.objects["rack"]
        bpy.ops.object.modifier_apply(modifier="bool_3")       

    bpy.ops.import_mesh.stl(filepath=os.path.join(target_dir, "hook.stl"))
    hook_obj = bpy.context.object
    hook_obj.name = "hook"
    bpy.data.objects["hook"].hide_set(True)

    boolean_modifier = obj.modifiers.new(name="bool_4", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["hook"]
    bpy.ops.object.modifier_apply(modifier="bool_4")

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_mesh.stl(filepath=os.path.join(target_dir, "hanger_vis.stl"), use_selection=True)


    ## create gripper
    # add grip pad for sdf calculation
    bpy.ops.mesh.primitive_cube_add()
    gripper1_obj = bpy.context.object
    gripper1_obj.name = "gripper1"
    gripper1_obj.scale = (0.008, 0.008, 0.030)
    gripper1_obj.location = (-0.015, 0.0, 0.023)
    gripper1_obj.hide_set(True)

    bpy.ops.mesh.primitive_cube_add()
    gripper2_obj = bpy.context.object
    gripper2_obj.name = "gripper2"
    gripper2_obj.scale = (0.008, 0.008, 0.030)
    gripper2_obj.location = (+0.015, 0.0, 0.023)
    gripper2_obj.hide_set(True)

    # voxelize and export
    obj = bpy.data.objects["link"]
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj

    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["gripper1"]
    bpy.ops.object.modifier_apply(modifier="Boolean")

    boolean_modifier = obj.modifiers.new(name="Boolean", type="BOOLEAN")
    boolean_modifier.operation = 'UNION'
    boolean_modifier.object = bpy.data.objects["gripper2"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    ## voxelize turns out too slow, so shut down...
    # voxel_modifier = obj.modifiers.new(name="voxel", type="REMESH")
    # voxel_modifier.voxel_size=0.005

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_mesh.stl(filepath=os.path.join(target_dir, "hanger.stl"), use_selection=True)
    obj.hide_set(True)


    ## post process
    mesh = trimesh.load_mesh(os.path.join(target_dir, f"hanger_vis.stl"), force="mesh")
    mesh.export(os.path.join(target_dir, f"hanger_vis.obj"))

    mesh = trimesh.load_mesh(os.path.join(target_dir, f"hanger.stl"), force="mesh")
    mesh.export(os.path.join(target_dir, f"hanger.obj"))

    label_y = float(TRI_HEIGHT_OVERALL_SHIFT - triangle_height)
    omegaconf.OmegaConf.save(
        omegaconf.DictConfig(dict(
            left = [float(-triangle_left), label_y, 0.0],
            right = [float(triangle_right), label_y, 0.0]
        )), os.path.join(target_dir, "hanger.obj.meta.yaml"))


    # delete all objects
    for obj in bpy.data.objects:
        obj.hide_viewport = False
        obj.hide_set(False)
        obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def generate_hangers(
        target_dir : str = os.path.join(".","data"),
        num_to_generate : int = 1,
):

    """ generate several hangers """

    for i in range(num_to_generate):

        # 70% of hangers will have rack
        add_rack = np.random.rand() < 0.7

        output_dir = os.path.join(target_dir, str(i))
        os.makedirs(output_dir, exist_ok=True)
        generate_one_hanger(output_dir, add_rack)

    clear_stl(target_dir)