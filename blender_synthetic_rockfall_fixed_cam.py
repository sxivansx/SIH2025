# blender_synthetic_rockfall_fixed_cam.py - Environmental Parameter Variation Dataset Generator
# 
# Features:
# - Fixed camera position for consistent viewpoint
# - Environmental parameter variations (rainfall, temperature, vibration, strain)
# - Particle system rainfall simulation
# - Temperature-based material variations
# - Vibration effects with motion blur and displacement
# - Strain-based terrain deformation
# - Multi-modal outputs (RGB, depth, normals, object indices)
# - Comprehensive parameter metadata tracking
#
# Usage:
# blender --background --python blender_synthetic_rockfall_fixed_cam.py -- <out_dir> <num_samples> <width> <height> [seed] [dem_path] [dem_strength]
#
# Parameters:
#   out_dir: Output directory for dataset
#   num_samples: Number of parameter variations to generate
#   width, height: Render resolution
#   seed: Random seed for reproducibility
#   dem_path: Optional path to DEM file (supports GeoTIFF)
#   dem_strength: Displacement strength for DEM (default: 5.0)
#
# Output Structure:
#   dataset/
#     Image_NNNN.png          - RGB images with environmental effects
#     Depth_NNNN.exr          - Depth maps
#     Normal_NNNN.exr         - Surface normals
#     Index_NNNN.png          - Object indices
#     ann_NNNN.json           - Per-sample annotations with parameters
#     dataset_index.json      - Master dataset metadata

import bpy
import sys
import os
import random
import math
import json
from datetime import datetime
from mathutils import Vector, Euler

# Try to import rasterio for DEM processing (optional)
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available. DEM georeferencing will be limited.")

# Try to import numpy for mesh generation (optional but recommended)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Using fallback mesh generation.")

# -------------------------
# Helpers (robust)
# -------------------------
def parse_argv():
    argv = sys.argv
    if "--" in argv:
        user_args = argv[argv.index("--")+1:]
    else:
        user_args = argv[1:]
    if len(user_args) < 4:
        print("Usage: blender --background --python blender_synthetic_rockfall_fixed_cam.py -- <out_dir> <num_samples> <width> <height> [seed] [dem_path] [dem_strength]")
        sys.exit(1)
    out_dir = user_args[0]
    num_samples = int(user_args[1])
    width = int(user_args[2])
    height = int(user_args[3])
    seed = int(user_args[4]) if len(user_args) > 4 else 42
    dem_path = user_args[5] if len(user_args) > 5 else None
    dem_strength = float(user_args[6]) if len(user_args) > 6 else 5.0
    return out_dir, num_samples, width, height, seed, dem_path, dem_strength

def clear_scene_full():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # try to release datablocks
    for m in list(bpy.data.meshes):
        try: bpy.data.meshes.remove(m)
        except Exception: pass
    for mat in list(bpy.data.materials):
        try: bpy.data.materials.remove(mat)
        except Exception: pass
    for img in list(bpy.data.images):
        try: bpy.data.images.remove(img)
        except Exception: pass
    for tex in list(bpy.data.textures):
        try: bpy.data.textures.remove(tex)
        except Exception: pass

def get_active_view_layer(scene):
    try:
        return scene.view_layers.active
    except Exception:
        if "ViewLayer" in scene.view_layers:
            return scene.view_layers["ViewLayer"]
        if "View Layer" in scene.view_layers:
            return scene.view_layers["View Layer"]
        return scene.view_layers[0]

def ensure_rigidbody_world(scene):
    try:
        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()
    except Exception:
        try:
            bpy.ops.scene.rigidbody_world_add()
        except Exception:
            pass

def safe_remove_all_file_slots(node):
    try:
        while len(node.file_slots) > 0:
            node.file_slots.remove(0)
    except Exception:
        try:
            node.file_slots.clear()
        except Exception:
            pass

# -------------------------
# Environmental Parameter Generation
# -------------------------
def generate_parameter_variations(num_samples, seed=42):
    """Generate systematic variations of environmental parameters."""
    random.seed(seed)
    variations = []
    
    # Define parameter ranges
    rainfall_levels = ["none", "light", "heavy"]
    temperature_ranges = ["cold", "normal", "hot"]  # Affects material properties
    vibration_levels = ["none", "low", "medium", "high"]
    strain_levels = ["none", "low", "medium", "high"]
    
    for i in range(num_samples):
        # Systematic variation with some randomness
        variation = {
            "sample_id": i + 1,
            "rainfall": {
                "level": random.choice(rainfall_levels),
                "intensity": random.uniform(0.0, 50.0) if random.choice(rainfall_levels) != "none" else 0.0,
                "particle_count": random.randint(1000, 5000) if random.choice(rainfall_levels) != "none" else 0
            },
            "temperature": {
                "level": random.choice(temperature_ranges),
                "celsius": random.uniform(-5, 35),
                "surface_roughness": random.uniform(0.1, 2.0),
                "color_shift": random.uniform(-0.2, 0.2)  # HSV shift
            },
            "vibration": {
                "level": random.choice(vibration_levels),
                "amplitude": random.uniform(0.0, 0.5),
                "frequency": random.uniform(1.0, 10.0),
                "motion_blur": random.uniform(0.0, 1.0)
            },
            "strain_displacement": {
                "level": random.choice(strain_levels),
                "displacement_strength": random.uniform(0.0, 2.0),
                "crack_density": random.uniform(0.0, 1.0),
                "deformation_scale": random.uniform(0.0, 0.3)
            },
            "lighting": {
                "sun_strength": random.uniform(2.0, 8.0),
                "sun_angle": random.uniform(20, 80),
                "sky_turbidity": random.uniform(2.0, 10.0)
            }
        }
        variations.append(variation)
    
    return variations

# -------------------------
# Terrain creation with deformation
# -------------------------
def make_terrain_with_deformation(size=30, subdivisions=8, base_seed=42, deformation_params=None):
    """Create terrain with optional strain-based deformation."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0,0,0))
    terrain = bpy.context.active_object
    terrain.name = "Terrain_Deformable"
    
    # Add subdivision for detail
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        for _ in range(subdivisions):
            bpy.ops.mesh.subdivide()
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    
    # Base procedural displacement
    try:
        tex = bpy.data.textures.new("terrain_base", type='CLOUDS')
        tex.noise_scale = 1.5
        tex.noise_basis = 'ORIGINAL_PERLIN'
        
        disp = terrain.modifiers.new("Base_Displace", type='DISPLACE')
        disp.texture = tex
        disp.strength = 4.0
    except Exception:
        pass
    
    # Add strain-based deformation if specified
    if deformation_params and deformation_params["level"] != "none":
        try:
            # Create crack texture
            crack_tex = bpy.data.textures.new("crack_texture", type='VORONOI')
            crack_tex.distance_metric = 'DISTANCE'
            crack_tex.noise_scale = deformation_params["crack_density"] * 5.0
            
            # Add crack displacement
            crack_disp = terrain.modifiers.new("Crack_Displace", type='DISPLACE')
            crack_disp.texture = crack_tex
            crack_disp.strength = deformation_params["displacement_strength"]
            crack_disp.mid_level = 0.8  # Create cracks (negative displacement)
        except Exception:
            pass
    
    # Add rigid body
    try:
        bpy.ops.rigidbody.object_add()
        terrain.rigid_body.type = 'PASSIVE'
        terrain.rigid_body.friction = 1.0
    except Exception:
        pass
    
    return terrain

# -------------------------
# Dynamic Material System
# -------------------------
def create_dynamic_terrain_material(temperature_params, strain_params):
    """Create terrain material that responds to temperature and strain."""
    mat = bpy.data.materials.new("Dynamic_Terrain")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create base nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Base color with temperature variation
    base_color = [0.4, 0.3, 0.2, 1.0]  # Brown rock color
    temp_shift = temperature_params["color_shift"]
    
    # Adjust color based on temperature
    if temperature_params["level"] == "cold":
        base_color[2] += abs(temp_shift)  # More blue
    elif temperature_params["level"] == "hot":
        base_color[0] += abs(temp_shift)  # More red
    
    principled.inputs['Base Color'].default_value = base_color
    principled.inputs['Roughness'].default_value = temperature_params["surface_roughness"]
    
    # Add strain-based color variation
    if strain_params["level"] != "none":
        try:
            # Create noise texture for strain patterns
            noise_tex = nodes.new(type='ShaderNodeTexNoise')
            noise_tex.inputs['Scale'].default_value = strain_params["crack_density"] * 10
            
            # Color ramp for strain visualization
            color_ramp = nodes.new(type='ShaderNodeValToRGB')
            color_ramp.color_ramp.elements[0].color = base_color
            color_ramp.color_ramp.elements[1].color = [0.6, 0.1, 0.1, 1.0]  # Red for high strain
            
            # Connect nodes
            links.new(noise_tex.outputs['Fac'], color_ramp.inputs['Fac'])
            links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
        except Exception:
            pass
    
    # Connect to output
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_rock_material_with_temperature(temperature_params, rock_index=0):
    """Create rock material that varies with temperature."""
    mat = bpy.data.materials.new(f"Rock_Material_{rock_index}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Base rock colors with temperature variation
    base_colors = [
        [0.3, 0.3, 0.3, 1.0],  # Dark gray
        [0.5, 0.4, 0.3, 1.0],  # Brown
        [0.4, 0.4, 0.4, 1.0],  # Light gray
    ]
    
    color = base_colors[rock_index % len(base_colors)]
    temp_shift = temperature_params["color_shift"]
    
    # Temperature effects on color
    if temperature_params["level"] == "cold":
        color[2] += abs(temp_shift) * 0.5  # Slightly more blue
    elif temperature_params["level"] == "hot":
        color[0] += abs(temp_shift) * 0.5  # Slightly more red
        color[1] += abs(temp_shift) * 0.3  # Slightly more yellow
    
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = temperature_params["surface_roughness"]
    
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

# -------------------------
# Rainfall Particle System
# -------------------------
def add_rainfall_system(rainfall_params):
    """Add particle system for rainfall simulation."""
    if rainfall_params["level"] == "none":
        return None
    
    try:
        # Create emitter plane above the scene
        bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 30))
        emitter = bpy.context.active_object
        emitter.name = "Rain_Emitter"
        
        # Add particle system
        bpy.ops.object.particle_system_add()
        ps = emitter.particle_systems[0]
        settings = ps.settings
        
        # Configure particle system
        settings.type = 'EMITTER'
        settings.count = rainfall_params["particle_count"]
        settings.emit_from = 'FACE'
        settings.use_emit_random = True
        
        # Particle physics
        settings.physics_type = 'NEWTON'
        settings.mass = 0.01
        settings.particle_size = 0.05
        settings.size_random = 0.5
        
        # Velocity (downward)
        settings.normal_factor = -10.0  # Downward velocity
        settings.factor_random = 2.0
        
        # Lifetime
        settings.lifetime = 100
        settings.lifetime_random = 0.5
        
        # Render settings
        settings.render_type = 'LINE'
        settings.material_slot = 'DEFAULT'
        
        # Create rain material
        rain_mat = bpy.data.materials.new("Rain_Material")
        rain_mat.use_nodes = True
        nodes = rain_mat.node_tree.nodes
        nodes.clear()
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = [0.8, 0.9, 1.0, 1.0]  # Light blue
        emission.inputs['Strength'].default_value = 0.5
        
        rain_mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        # Assign material
        emitter.data.materials.append(rain_mat)
        
        return emitter
        
    except Exception as e:
        print(f"Failed to create rainfall system: {e}")
        return None

# -------------------------
# Vibration Effects
# -------------------------
def apply_vibration_effects(scene, vibration_params):
    """Apply vibration effects to camera and objects."""
    if vibration_params["level"] == "none":
        return
    
    try:
        # Enable motion blur if vibration is present
        scene.render.use_motion_blur = vibration_params["motion_blur"] > 0.1
        if scene.render.use_motion_blur:
            scene.render.motion_blur_shutter = vibration_params["motion_blur"]
        
        # Add camera shake (slight random offset)
        camera = scene.camera
        if camera and vibration_params["amplitude"] > 0:
            shake_x = random.uniform(-vibration_params["amplitude"], vibration_params["amplitude"])
            shake_y = random.uniform(-vibration_params["amplitude"], vibration_params["amplitude"])
            shake_z = random.uniform(-vibration_params["amplitude"] * 0.5, vibration_params["amplitude"] * 0.5)
            
            camera.location.x += shake_x
            camera.location.y += shake_y
            camera.location.z += shake_z
    
    except Exception as e:
        print(f"Failed to apply vibration effects: {e}")

# -------------------------
# Rocks + camera + light
# -------------------------
def create_rock_with_materials(seed=0, temperature_params=None):
    """Create rock with temperature-responsive materials."""
    random.seed(seed)
    size = random.uniform(0.4, 1.4)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=size, location=(0,0,0))
    rock = bpy.context.active_object
    rock.name = f"Rock_{seed}"
    
    # Add bumpy displacement
    try:
        rtex = bpy.data.textures.new(f"rock_tex_{seed}", type='STUCCI')
        rtex.noise_scale = random.uniform(0.3,1.2)
        
        mod = rock.modifiers.new("Displace", type='DISPLACE')
        mod.texture = rtex
        mod.strength = random.uniform(0.1,0.6)
        bpy.ops.object.modifier_apply(modifier="Displace")
    except Exception:
        pass
    
    # Apply temperature-responsive material
    if temperature_params:
        mat = create_rock_material_with_temperature(temperature_params, seed)
        rock.data.materials.append(mat)
    
    return rock

def spawn_rocks_with_materials(count=12, area=18, zmin=6.0, zmax=16.0, seed=0, temperature_params=None):
    """Spawn rocks with temperature-responsive materials."""
    random.seed(seed)
    rocks = []
    for i in range(count):
        r = create_rock_with_materials(seed + i, temperature_params)
        x = random.uniform(-area, area)
        y = random.uniform(-area, area)
        z = random.uniform(zmin, zmax)
        r.location = (x,y,z)
        r.rotation_euler = (random.random()*math.pi, random.random()*math.pi, random.random()*math.pi)
        s = random.uniform(0.6,1.4)
        r.scale = (s, s*random.uniform(0.8,1.2), s*random.uniform(0.8,1.2))
        
        try:
            bpy.ops.rigidbody.object_add()
            r.rigid_body.mass = random.uniform(10.0, 60.0)
            r.rigid_body.friction = 0.8
            r.rigid_body.restitution = 0.05
        except Exception:
            pass
        
        try:
            r.pass_index = (i % 254) + 1
        except Exception:
            pass
        
        rocks.append(r)
    return rocks

def add_fixed_camera(location=(25, -25, 15), look_at=(0,0,0), focal_mm=35):
    """Add fixed camera position for consistent viewpoint."""
    cam_data = bpy.data.cameras.new("FixedCam")
    cam = bpy.data.objects.new("FixedCam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    
    # Point camera at target
    dir_vec = Vector(look_at) - Vector(location)
    if dir_vec.length == 0:
        dir_vec = Vector((0,0,-1))
    cam.rotation_euler = dir_vec.to_track_quat('-Z','Y').to_euler()
    
    try:
        cam.data.lens = focal_mm
    except Exception:
        pass
    
    bpy.context.scene.camera = cam
    return cam

def add_dynamic_sun(lighting_params):
    """Add sun with dynamic properties based on parameters."""
    light_data = bpy.data.lights.new(name="Dynamic_Sun", type='SUN')
    light_data.energy = lighting_params["sun_strength"]
    
    sun = bpy.data.objects.new("Dynamic_Sun", light_data)
    bpy.context.collection.objects.link(sun)
    
    # Set sun angle
    angle_rad = math.radians(lighting_params["sun_angle"])
    sun.rotation_euler = (angle_rad, 0, math.radians(40))
    
    return sun

# -------------------------
# Compositor outputs (robust)
# -------------------------
def setup_compositor(out_dir, base_slot_names=None):
    """Setup compositor for multi-modal output."""
    if base_slot_names is None:
        base_slot_names = ["Image", "Depth", "Normal", "Index"]
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    rl = tree.nodes.new(type='CompositorNodeRLayers')
    file_out = tree.nodes.new(type='CompositorNodeOutputFile')
    file_out.base_path = out_dir if out_dir else ""
    safe_remove_all_file_slots(file_out)
    
    # create slots
    for name in base_slot_names:
        try:
            file_out.file_slots.new(name)
        except Exception:
            pass
    
    # link outputs
    links = tree.links
    for idx, name in enumerate(base_slot_names):
        try:
            target = file_out.inputs[idx]
        except Exception:
            try:
                target = file_out.inputs[0]
            except Exception:
                target = None
        if target is None:
            continue
        lname = name.lower()
        try:
            if "image" in lname and "Image" in rl.outputs:
                links.new(rl.outputs["Image"], target)
            elif "depth" in lname:
                if "Depth" in rl.outputs:
                    links.new(rl.outputs["Depth"], target)
                elif "Z" in rl.outputs:
                    links.new(rl.outputs["Z"], target)
            elif "normal" in lname and "Normal" in rl.outputs:
                links.new(rl.outputs["Normal"], target)
            elif "index" in lname:
                if "IndexOB" in rl.outputs:
                    links.new(rl.outputs["IndexOB"], target)
                elif "Object Index" in rl.outputs:
                    links.new(rl.outputs["Object Index"], target)
        except Exception:
            pass
    
    # enable view layer passes
    vl = get_active_view_layer(scene)
    try:
        vl.use_pass_z = True
        vl.use_pass_normal = True
        vl.use_pass_object_index = True
    except Exception:
        pass
    
    return rl, file_out, base_slot_names

# -------------------------
# annotation helper
# -------------------------
try:
    from bpy_extras.object_utils import world_to_camera_view
except Exception:
    def world_to_camera_view(scene, camera, coord):
        co_local = camera.matrix_world.normalized().inverted() @ coord
        if co_local.z == 0:
            return Vector((0.5,0.5,0.0))
        frame = camera.data.view_frame(scene=scene)
        x = 0.5 + (co_local.x / -co_local.z) * 0.5
        y = 0.5 + (co_local.y / -co_local.z) * 0.5
        return Vector((x,y,-co_local.z))

def bbox_2d_from_object(obj, cam, scene, render_size):
    """Calculate 2D bounding box from 3D object."""
    mat_world = obj.matrix_world
    coords_2d = []
    for corner in obj.bound_box:
        v_world = mat_world @ Vector(corner)
        co_ndc = world_to_camera_view(scene, cam, v_world)
        if co_ndc.z <= 0:
            continue
        x = co_ndc.x * render_size[0]
        y = (1.0 - co_ndc.y) * render_size[1]
        coords_2d.append((x,y))
    if not coords_2d:
        return [0,0,0,0]
    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]
    minx, maxx = max(0, min(xs)), min(render_size[0], max(xs))
    miny, maxy = max(0, min(ys)), min(render_size[1], max(ys))
    return [minx, miny, maxx, maxy]

# -------------------------
# Main generator
# -------------------------
def generate_parameter_variation_dataset(out_dir, num_samples=10, width=640, height=480, seed=42, dem_path=None, dem_strength=5.0):
    """Generate dataset with systematic parameter variations."""
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    
    # Generate parameter variations
    parameter_variations = generate_parameter_variations(num_samples, seed)
    
    annotations = []
    base_time = datetime.now()
    
    for sample_idx, params in enumerate(parameter_variations):
        print(f"\nGenerating sample {sample_idx + 1}/{num_samples}")
        print(f"Parameters: Rainfall={params['rainfall']['level']}, Temp={params['temperature']['level']}, Vibration={params['vibration']['level']}, Strain={params['strain_displacement']['level']}")
        
        # Clear scene for each sample
        clear_scene_full()
        ensure_rigidbody_world(bpy.context.scene)
        
        # Create terrain with deformation based on strain parameters
        terrain = make_terrain_with_deformation(
            size=30, 
            subdivisions=8, 
            base_seed=seed + sample_idx,
            deformation_params=params["strain_displacement"]
        )
        
        # Apply dynamic terrain material
        terrain_mat = create_dynamic_terrain_material(
            params["temperature"], 
            params["strain_displacement"]
        )
        terrain.data.materials.append(terrain_mat)
        
        # Spawn rocks with temperature-responsive materials
        rocks = spawn_rocks_with_materials(
            count=15, 
            area=18, 
            zmin=6.0, 
            zmax=16.0, 
            seed=seed + sample_idx,
            temperature_params=params["temperature"]
        )
        
        # Fixed camera setup
        cam = add_fixed_camera(location=(25, -25, 15), look_at=(0,0,0), focal_mm=35)
        
        # Dynamic lighting
        sun = add_dynamic_sun(params["lighting"])
        
        # Add rainfall if specified
        rain_emitter = add_rainfall_system(params["rainfall"])
        
        # Render settings
        scene = bpy.context.scene
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        
        try:
            scene.render.engine = 'CYCLES'
            scene.cycles.samples = 64
        except Exception:
            pass
        
        # Apply vibration effects
        apply_vibration_effects(scene, params["vibration"])
        
        # Setup compositor
        rl_node, file_out_node, base_names = setup_compositor(out_dir, ["Image","Depth","Normal","Index"])
        
        # Configure output paths
        sample_num = sample_idx + 1
        scene.frame_set(sample_num)
        file_out_node.base_path = os.path.abspath(out_dir)
        
        for idx, slot in enumerate(file_out_node.file_slots):
            bname = base_names[idx] if idx < len(base_names) else f"slot{idx}"
            slot.path = f"{bname}_{sample_num:04d}"
        
        scene.render.filepath = os.path.join(out_dir, f"Image_{sample_num:04d}.png")
        
        # Render
        try:
            bpy.ops.render.render(write_still=True)
            print(f"Rendered sample {sample_num}")
        except Exception as e:
            print(f"Render error for sample {sample_num}: {e}")
        
        # Generate annotations
        cam_data = cam.data
        try:
            f_mm = cam_data.lens
            sensor_w = getattr(cam_data, "sensor_width", 32.0)
        except Exception:
            f_mm, sensor_w = 35.0, 32.0
        fx = (f_mm / sensor_w) * width
        fy = fx
        cx = width/2.0; cy = height/2.0
        
        # Camera annotation
        cam_ann = {
            "location": list(cam.location), 
            "rotation_euler": list(cam.rotation_euler),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height},
            "type": "fixed_camera"
        }
        
        # Rock annotations
        rocks_ann = []
        for r in rocks:
            try:
                pos = list(r.matrix_world.translation)
                rot = list(r.rotation_euler)
                pid = getattr(r, "pass_index", 0)
                bbox = bbox_2d_from_object(r, cam, scene, (width, height))
                velocity = [0, 0, 0]  # Fixed camera, so rocks start stationary
            except Exception:
                pos, rot, pid, bbox, velocity = [0,0,0],[0,0,0],0,[0,0,0,0],[0,0,0]
            
            rocks_ann.append({
                "name": r.name, 
                "pass_index": pid, 
                "location": pos, 
                "rotation": rot, 
                "bbox_2d": bbox,
                "velocity": velocity
            })
        
        # Complete sample annotation
        sample_ann = {
            "sample_id": sample_num,
            "timestamp": (base_time).isoformat(),
            "files": {
                "image": f"Image_{sample_num:04d}.png",
                "depth": f"Depth_{sample_num:04d}.exr",
                "normal": f"Normal_{sample_num:04d}.exr",
                "index": f"Index_{sample_num:04d}.png"
            },
            "camera": cam_ann,
            "rocks": rocks_ann,
            "environmental_parameters": params,
            "render_settings": {
                "width": width,
                "height": height,
                "engine": "CYCLES",
                "samples": 64
            }
        }
        
        annotations.append(sample_ann)
        
        # Save individual annotation
        try:
            with open(os.path.join(out_dir, f"ann_{sample_num:04d}.json"), "w") as jf:
                json.dump(sample_ann, jf, indent=2)
        except Exception as e:
            print(f"Failed to save annotation for sample {sample_num}: {e}")
    
    # Save master dataset index
    dataset_metadata = {
        "dataset_info": {
            "created": datetime.now().isoformat(),
            "num_samples": num_samples,
            "resolution": {"width": width, "height": height},
            "seed": seed,
            "dataset_type": "environmental_parameter_variations",
            "camera_type": "fixed"
        },
        "parameter_ranges": {
            "rainfall": ["none", "light", "heavy"],
            "temperature": ["cold", "normal", "hot"],
            "vibration": ["none", "low", "medium", "high"],
            "strain_displacement": ["none", "low", "medium", "high"]
        },
        "samples": annotations
    }
    
    try:
        with open(os.path.join(out_dir, "dataset_index.json"), "w") as mf:
            json.dump(dataset_metadata, mf, indent=2)
    except Exception as e:
        print(f"Failed to save dataset index: {e}")
    
    print(f"\nDataset generation complete!")
    print(f"Generated {num_samples} samples with environmental parameter variations")
    print(f"Output directory: {os.path.abspath(out_dir)}")
    print(f"Features: Fixed camera, rainfall particles, temperature materials, vibration effects, strain deformation")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    out_dir, num_samples, width, height, seed, dem_path, dem_strength = parse_argv()
    print("Fixed Camera Environmental Parameter Variation Dataset Generator")
    print("OUT:", out_dir, "N:", num_samples, "RES:", f"{width}x{height}", "seed:", seed)
    if dem_path:
        print("DEM:", dem_path, "DEM_strength:", dem_strength)
    generate_parameter_variation_dataset(
        out_dir, 
        num_samples=num_samples, 
        width=width, 
        height=height, 
        seed=seed, 
        dem_path=dem_path, 
        dem_strength=dem_strength
    )
