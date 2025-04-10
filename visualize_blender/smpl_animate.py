import bpy
import os
import math
import matplotlib.colors
import numpy as np
import logging
import uuid


scene_name = "Scene"
base_dir = "D:/RESEARCH/Code/CoShMDM/results/"

directory = "Two_persons_doing_high_five"
repitition = 0
p1_offset = (0.27, 0.8, 0)
p2_offset = (0, -0.8, 0)



objs = directory + f"/objs_{repitition}/"
obj_directory_p1 = base_dir + objs + "p1/"
obj_directory_p2 = base_dir + objs + "p2/"


#p1_root_translations = np.load(base_dir + directory + f'/p1_{repitition}_smpl_params.npy', allow_pickle=True)[None][0]['root_translation']
#p2_root_translations = np.load(base_dir + directory + f'/p2_{repitition}_smpl_params.npy', allow_pickle=True)[None][0]['root_translation']
#p1_root_rotations = np.load(base_dir + directory + f'/p1_{repitition}_smpl_params.npy', allow_pickle=True)[None][0]['root_translation']
#p2_root_rotations = np.load(base_dir + directory + f'/p2_{repitition}_smpl_params.npy', allow_pickle=True)[None][0]['root_translation']







output_directory = "C:/animations"
animation_file_name = objs.split("/")[0] + ".mp4"



color_background = "#E2E2E2"
color_smpl_blue = (0, 0.181164, 0.48515, 1) #0076B9FF
color_smpl_red = (0.806948, 0.116971, 0, 1)  #E86000FF
color_plane1 = "#444444"
color_plane2 = "#121212"
color_plane3 = "#E2E2E2"


def hex_to_rgb(hex_color):
    rgb = list(matplotlib.colors.to_rgb(hex_color))
    rgb.append(1.0)
    return tuple(rgb)


def hsv_to_rgb(h, s, v):
    rgb = list(matplotlib.colors.hsv_to_rgb((h, s, v)))
    rgb.append(1.0)
    return tuple(rgb)




















def setup_renderer():

    # Other common render settings
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes[scene_name].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPTIX'

    bpy.context.scene.cycles.use_adaptive_sampling = True
    
    bpy.context.scene.cycles.samples = 4098
    
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.gamma = 1.2
    bpy.context.scene.view_settings.exposure = -0.75



def setup_scene(res="ultra"):
    scene = bpy.data.scenes[scene_name]
    assert res in ["ultra", "high", "med", "low"]

    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280//2
        scene.render.resolution_y = 1024//2
    elif res == "low":
        scene.render.resolution_x = 1280//4
        scene.render.resolution_y = 1024//4
    elif res == "ultra":
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080

    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()
        
#    if not 'Sun' in bpy.data.objects:
#        bpy.ops.object.light_add(type='SUN', align='WORLD',
#                                 location=(0, 0, 0), scale=(1, 1, 1))
#        bpy.data.objects["Sun"].data.energy = 3
        
    if not 'Point' in bpy.data.objects:        
        bpy.ops.object.light_add(type='POINT', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.data.objects["Point"].data.energy = 1500
        
        bpy.ops.transform.translate(value=(0, -2.08807, 2.91), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)




    if not 'Empty' in bpy.data.objects:
    # rotate camera
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.transform.resize(value=(10, 10, 10), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                 orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1,
                                 use_proportional_connected=False, use_proportional_projected=False)
    
    bpy.ops.object.select_all(action='DESELECT')

    return scene








def floor_small_material(material_name):
    material = None
    if material_name in bpy.data.materials:
        material = bpy.data.materials[material_name]
    else:
        material = bpy.data.materials.new(name=material_name)

    material.use_nodes = True
    bsdf_node = material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        bsdf_node.inputs['Base Color'].default_value = (0.320026, 0.320026, 0.320026, 1)
        bsdf_node.inputs['Metallic'].default_value = 0.0
        bsdf_node.inputs['Roughness'].default_value = 0.127
        bsdf_node.inputs['IOR'].default_value = 4.2
        
    return material
        

def floor_big_material(material_name):
    material = None
    if material_name in bpy.data.materials:
        material = bpy.data.materials[material_name]
    else:
        material = bpy.data.materials.new(name=material_name)

    material.use_nodes = True
    bsdf_node = material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        bsdf_node.inputs['Base Color'].default_value = (0.451586, 0.451586, 0.451586, 1)
        bsdf_node.inputs['Metallic'].default_value = 0.0
        bsdf_node.inputs['Roughness'].default_value = 0.175
        bsdf_node.inputs['IOR'].default_value = 8.3
        
    return material
    


def add_text(text="LLM-FQK-T2M"):
    bpy.ops.object.text_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.editmode_toggle()
    bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    bpy.ops.object.editmode_toggle()
    bpy.context.object.data.size = 0.29
    
    bpy.ops.transform.translate(value=(-0.770385, -0.144932, -1.10067), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
    
#    bpy.ops.object.convert(target='MESH')
#    bpy.ops.object.editmode_toggle()

#    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, -0.015415), "orient_type":'NORMAL', "orient_matrix":((-0.258545, 0.965999, 0), (-0.965999, -0.258545, 0), (0, 0, 1)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})

    





def plot_floor():
    # Create a floor
    location_small = (0,0,0)
    location_big = (0,0,-0.01)
    location_huge = (0,0,-0.057522)
    scale_small = (1,1,1)
    scale_big = (1.2,1.2,1.2)
    scale_huge = (1000,1000,1000)
    
    color_small = (0.1, 0.1, 0.1, 1)
    color_big = (0.2, 0.2, 0.2, 1)
    color_huge = (0.643, 0.643, 0.643, 1)

#    if not 'SmallPlane' in bpy.data.objects:
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location_small, scale=scale_small)

    bpy.ops.transform.resize(value=scale_small, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.context.selected_objects[0]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"
    obj.active_material = floor_mat(color=color_small)

#    obj.active_material = floor_small_material("smallFloorMaterial")
    



#    if not 'BigPlane' in bpy.data.objects:
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location_big, scale=scale_big)

    bpy.ops.transform.resize(value=scale_big, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

    obj = bpy.context.selected_objects[0]
    obj.name = "BigPlane"
    obj.data.name = "BigPlane"
    obj.active_material = floor_mat(color=color_big)

#    obj.active_material = floor_big_material("bigFloorMaterial")
    
    bpy.ops.object.editmode_toggle()
    
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, -0.0365686), "orient_type":'NORMAL', "orient_matrix":((0, -1, 0), (1, 0, -0), (0, 0, 1)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})


    
    bpy.ops.object.editmode_toggle()








        

#    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location_huge, scale=scale_huge)

#    bpy.ops.transform.resize(value=scale_huge, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
#                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
#                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
#                                 use_proportional_projected=False, release_confirm=True)

#    obj = bpy.context.selected_objects[0]
#    obj.name = "HugePlane"
#    obj.data.name = "HugePlane"
#    obj.active_material = floor_mat(color=color_huge)
        

def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material_diffuse_BSDF(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material

# keys:
# ['Base Color', 'Subsurface', 'Subsurface Radius', 'Subsurface Color', 'Metallic', 'Specular', 'Specular Tint', 'Roughness', 'Anisotropic', 'Anisotropic Rotation', 'Sheen', 1Sheen Tint', 'Clearcoat', 'Clearcoat Roughness', 'IOR', 'Transmission', 'Transmission Roughness', 'Emission', 'Emission Strength', 'Alpha', 'Normal', 'Clearcoat Normal', 'Tangent']
DEFAULT_BSDF_SETTINGS = {"Subsurface": 0.15,
                         "Subsurface Radius": [1.1, 0.2, 0.1],
                         "Metallic": 0.15,
                         "Specular": 0.5,
                         "Specular Tint": 0.5,
                         "Roughness": 0.75,
                         "Anisotropic": 0.25,
                         "Anisotropic Rotation": 0.25,
                         "Sheen": 0.75,
                         "Sheen Tint": 0.5,
                         "Clearcoat": 0.5,
                         "Clearcoat Roughness": 0.5,
                         "IOR": 1.450,
                         "Transmission": 0.1,
                         "Transmission Roughness": 0.1,
                         "Emission": (0, 0, 0, 1),
                         "Emission Strength": 0.0,
                         "Alpha": 1.0}

def body_material(r, g, b, a=1, name="body"):
    materials = bpy.data.materials
    material = materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    diffuse = nodes["Principled BSDF"]
    inputs = diffuse.inputs

    settings = DEFAULT_BSDF_SETTINGS.copy()
    settings["Base Color"] = (r, g, b, a)
    settings["Subsurface Color"] = (r, g, b, a)
    settings["Subsurface"] = 0.0

    for setting, val in settings.items():
        inputs[setting].default_value = val

    return material


def colored_material_bsdf(name, **kwargs):
    materials = bpy.data.materials
    material = materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    diffuse = nodes["Principled BSDF"]
    inputs = diffuse.inputs

    settings = DEFAULT_BSDF_SETTINGS.copy()
    for key, val in kwargs.items():
        settings[key] = val

    for setting, val in settings.items():
        inputs[setting].default_value = val

    return material


def floor_mat(name="floor_mat", color=(0.1, 0.1, 0.1, 1), roughness=0.127451):
    return colored_material_diffuse_BSDF(color[0], color[1], color[2], a=color[3], roughness=roughness)




setup_renderer()
setup_scene()
plot_floor()
add_text()
























obj_files_p1 = sorted([f for f in os.listdir(obj_directory_p1) if f.endswith(".obj")])
obj_files_p2 = sorted([f for f in os.listdir(obj_directory_p2) if f.endswith(".obj")])



# Set up scene
scene = bpy.context.scene
frame_start = 1 


   
    
    

# List to keep track of all imported objects
smpls_p1 = []
smpls_p2 = []

# Loop through all OBJ files and import them
for index, (obj_file_p1, obj_file_p2) in enumerate(zip(obj_files_p1, obj_files_p2)):
    
    
    
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_directory_p1, obj_file_p1))
    

    smpl = bpy.context.selected_objects[0]
    smpls_p1.append(smpl)
    bpy.ops.object.shade_smooth()
    
#    bpy.context.object.location[0] = p1_root_translations[0, index]
#    bpy.context.object.location[1] = p1_root_translations[1, index]

    bpy.context.object.location[0] = p1_offset[0]
    bpy.context.object.location[1] = p1_offset[1]
    bpy.context.object.location[2] = p1_offset[2]
    
        
    smpl_material_name = f"{uuid.uuid4()}"

    if smpl_material_name in bpy.data.materials:
        smpl_material = bpy.data.materials[smpl_material_name]
    else:
        smpl_material = bpy.data.materials.new(name=smpl_material_name)

    smpl_material.use_nodes = True
    bsdf_node = smpl_material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        bsdf_node.inputs['Base Color'].default_value = color_smpl_blue
        bsdf_node.inputs['Metallic'].default_value = 0.15
        bsdf_node.inputs['Roughness'].default_value = 0.265
        bsdf_node.inputs['IOR'].default_value = 1.0
        
    
    if len(smpl.data.materials):
        smpl.data.materials[0] = smpl_material
    else:
        smpl.data.materials.append(smpl_material)
        
        
        
    
    
    
    
    
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_directory_p2, obj_file_p2))
    

    smpl = bpy.context.selected_objects[0]
    smpls_p2.append(smpl)
    bpy.ops.object.shade_smooth()


#    bpy.context.object.location[0] = p2_root_translations[0, index]
#    bpy.context.object.location[1] = p2_root_translations[1, index]

    bpy.context.object.location[0] = p2_offset[0]
    bpy.context.object.location[1] = p2_offset[1]
    bpy.context.object.location[2] = p2_offset[2]


    
        
    smpl_material_name = f"{uuid.uuid4()}"

    if smpl_material_name in bpy.data.materials:
        smpl_material = bpy.data.materials[smpl_material_name]
    else:
        smpl_material = bpy.data.materials.new(name=smpl_material_name)

    smpl_material.use_nodes = True
    bsdf_node = smpl_material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        bsdf_node.inputs['Base Color'].default_value = color_smpl_red
        bsdf_node.inputs['Metallic'].default_value = 0.15
        bsdf_node.inputs['Roughness'].default_value = 0.265
        bsdf_node.inputs['IOR'].default_value = 1.0
        
    
    if len(smpl.data.materials):
        smpl.data.materials[0] = smpl_material
    else:
        smpl.data.materials.append(smpl_material)
        
        
        
        
        
        
    







# Set visibility keyframes
for index, (smpl_p1, smpl_p2) in enumerate(zip(smpls_p1, smpls_p2)):
    frame_number = frame_start + index

    # Set the visibility for the current frame
    for obj in smpls_p1:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
        obj.keyframe_insert(data_path="hide_render", frame=frame_number)

    # Only make the current object visible
    smpl_p1.hide_viewport = False
    smpl_p1.hide_render = False
    smpl_p1.keyframe_insert(data_path="hide_viewport", frame=frame_number)
    smpl_p1.keyframe_insert(data_path="hide_render", frame=frame_number)
    
    
    
    # Set the visibility for the current frame
    for obj in smpls_p2:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
        obj.keyframe_insert(data_path="hide_render", frame=frame_number)

    # Only make the current object visible
    smpl_p2.hide_viewport = False
    smpl_p2.hide_render = False
    smpl_p2.keyframe_insert(data_path="hide_viewport", frame=frame_number)
    smpl_p2.keyframe_insert(data_path="hide_render", frame=frame_number)

# Set the end frame of the animation to the total number of OBJ files
scene.frame_end = frame_start + len(obj_files_p1) - 1












# Set up render settings to export animation as a video

scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
scene.render.filepath = os.path.join(output_directory, animation_file_name)
scene.render.fps = 24  # Set your desired frames per second
scene.frame_start = frame_start  # Set the start frame
scene.frame_end = frame_start + len(obj_files_p1) - 1  # Set the end frame

print("Animation setup completed.")

# Render the animation to the output video file
#bpy.ops.render.render(animation=True)



