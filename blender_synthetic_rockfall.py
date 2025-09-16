# blender_synthetic_rockfall.py - Enhanced Synthetic Rockfall Dataset Generator
# 
# Features:
# - DEM support with georeferencing (requires rasterio)
# - Drone flight path simulation (circular, grid, spiral patterns)
# - Geotechnical sensor simulation (slope, pore pressure, displacement, strain, vibrations)
# - Environmental time-series (rainfall, temperature, wind)
# - Multi-modal outputs (RGB, depth, normals, object indices)
# - Comprehensive JSON annotations with drone metadata
# - CSV exports for sensor and environmental data
#
# Usage:
# blender --background --python blender_synthetic_rockfall.py -- <out_dir> <num_samples> <width> <height> [seed] [dem_path] [dem_strength] [flight_pattern] [num_sensors]
#
# Parameters:
#   out_dir: Output directory for dataset
#   num_samples: Number of frames to generate
#   width, height: Render resolution
#   seed: Random seed for reproducibility
#   dem_path: Optional path to DEM file (supports GeoTIFF)
#   dem_strength: Displacement strength for DEM (default: 5.0)
#   flight_pattern: Drone flight pattern - 'circular', 'grid', or 'spiral' (default: 'circular')
#   num_sensors: Number of geotechnical sensors to place (default: 5)
#
# Dependencies:
#   Required: bpy (Blender Python API)
#   Optional: rasterio (for DEM georeferencing), numpy (for mesh generation)
#
# Output Structure:
#   dataset/
#     Image_NNNN.png          - RGB images
#     Depth_NNNN.exr          - Depth maps
#     Normal_NNNN.exr         - Surface normals
#     Index_NNNN.png          - Object indices
#     ann_NNNN.json           - Per-frame annotations
#     sensor_NN_timeseries.csv - Per-sensor measurements
#     env_timeseries.csv       - Environmental data
#     dataset_index.json       - Master dataset metadata

import bpy
import sys
import os
import random
import math
import json
import csv
from datetime import datetime, timedelta
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
        print("Usage: blender --background --python blender_synthetic_rockfall.py -- <out_dir> <num_samples> <width> <height> [seed] [dem_path] [dem_strength] [flight_pattern] [num_sensors]")
        sys.exit(1)
    out_dir = user_args[0]
    num_samples = int(user_args[1])
    width = int(user_args[2])
    height = int(user_args[3])
    seed = int(user_args[4]) if len(user_args) > 4 else 42
    dem_path = user_args[5] if len(user_args) > 5 else None
    dem_strength = float(user_args[6]) if len(user_args) > 6 else 5.0
    flight_pattern = user_args[7] if len(user_args) > 7 else "circular"
    num_sensors = int(user_args[8]) if len(user_args) > 8 else 5
    return out_dir, num_samples, width, height, seed, dem_path, dem_strength, flight_pattern, num_sensors

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
# DEM Processing with Georeferencing
# -------------------------
def load_dem_with_georef(dem_path):
    """Load DEM with full georeferencing information using rasterio."""
    if not HAS_RASTERIO or not dem_path or not os.path.exists(dem_path):
        return None, None, None, None
    
    try:
        with rasterio.open(dem_path) as src:
            elevation_data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # Convert transform to list for JSON serialization
            geotransform = [transform.a, transform.b, transform.c, 
                          transform.d, transform.e, transform.f]
            
            return elevation_data, geotransform, crs.to_string() if crs else None, bounds
    except Exception as e:
        print(f"Failed to load DEM with georeferencing: {e}")
        return None, None, None, None

def create_mesh_from_dem_data(elevation_data, geotransform, terrain_size=30, name="Terrain_DEM_Exact"):
    """Create precise mesh from DEM elevation data."""
    if not HAS_NUMPY or elevation_data is None:
        return None
        
    try:
        # Downsample for performance if too large
        max_size = 200
        if elevation_data.shape[0] > max_size or elevation_data.shape[1] > max_size:
            step_y = max(1, elevation_data.shape[0] // max_size)
            step_x = max(1, elevation_data.shape[1] // max_size)
            elevation_data = elevation_data[::step_y, ::step_x]
        
        rows, cols = elevation_data.shape
        
        # Create vertices
        vertices = []
        faces = []
        
        # Scale factors to fit terrain_size
        scale_x = terrain_size / cols
        scale_y = terrain_size / rows
        
        # Generate vertices
        for row in range(rows):
            for col in range(cols):
                x = (col - cols/2) * scale_x
                y = (row - rows/2) * scale_y
                z = elevation_data[row, col] * 0.1  # Scale height
                vertices.append((x, y, z))
        
        # Generate faces (quads -> triangles)
        for row in range(rows - 1):
            for col in range(cols - 1):
                # Vertex indices for current quad
                v1 = row * cols + col
                v2 = row * cols + col + 1
                v3 = (row + 1) * cols + col
                v4 = (row + 1) * cols + col + 1
                
                # Create two triangles from quad
                faces.append((v1, v2, v4))
                faces.append((v1, v4, v3))
        
        # Create mesh in Blender
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        
        # Create object
        terrain_obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(terrain_obj)
        
        # Add rigid body
        bpy.context.view_layer.objects.active = terrain_obj
        try:
            bpy.ops.rigidbody.object_add()
            terrain_obj.rigid_body.type = 'PASSIVE'
            terrain_obj.rigid_body.friction = 1.0
        except Exception:
            pass
            
        return terrain_obj
        
    except Exception as e:
        print(f"Failed to create mesh from DEM data: {e}")
        return None

# -------------------------
# Terrain creation: DEM or procedural
# -------------------------
def make_terrain_from_dem(dem_path, size=30, subdiv_levels=4, disp_strength=5.0, apply_modifiers=False):
    """
    Create a plane and use an IMAGE-type texture for Displace, loading dem_path image.
    subdiv_levels controls the Subdivision Surface modifier levels (tradeoff speed vs detail).
    """
    if not dem_path or not os.path.exists(dem_path):
        raise FileNotFoundError("DEM path missing or not found")

    # Load image
    try:
        img = bpy.data.images.load(dem_path)
    except Exception as e:
        raise RuntimeError(f"Could not load DEM image: {e}")

    # Create plane (we'll map the image onto it)
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0,0,0))
    terrain = bpy.context.active_object
    terrain.name = "Terrain_DEM"

    # Ensure UVs exist and map the whole image over the plane
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    # Add Subdivision Surface to get enough geometry (simple/fast)
    try:
        sub = terrain.modifiers.new("Subsurf", type='SUBSURF')
        sub.levels = max(1, subdiv_levels)
        sub.render_levels = max(1, subdiv_levels)
    except Exception:
        sub = None

    # Create texture of type IMAGE and assign image
    try:
        tex = bpy.data.textures.new("DEM_image_texture", type='IMAGE')
        tex.image = img
    except Exception:
        # if image texture type not available, fall back to Clouds procedural
        tex = bpy.data.textures.new("DEM_fallback_noise", type='CLOUDS')

    # Add Displace modifier using the image texture
    try:
        disp = terrain.modifiers.new("DEM_Displace", type='DISPLACE')
        disp.texture = tex
        disp.strength = disp_strength
        disp.mid_level = 0.0
        # Texture coordinates UV mapping
        disp.texture_coords = 'UV'
    except Exception:
        disp = None

    # Optional: apply modifiers to bake the geometry (be careful with very dense subdivisions)
    if apply_modifiers:
        try:
            bpy.context.view_layer.objects.active = terrain
            if sub:
                bpy.ops.object.modifier_apply(modifier=sub.name)
            if disp:
                bpy.ops.object.modifier_apply(modifier=disp.name)
        except Exception:
            pass

    # Add passive rigid body
    try:
        bpy.ops.rigidbody.object_add()
        terrain.rigid_body.type = 'PASSIVE'
        terrain.rigid_body.friction = 1.0
    except Exception:
        pass

    return terrain

def make_terrain_procedural(size=30, subdivisions=160, displace_strength=4.0, seed=0):
    # fallback procedural terrain (coarse but quick)
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0,0,0))
    terrain = bpy.context.active_object
    terrain.name = "Terrain_Procedural"
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=8)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        tex = bpy.data.textures.new("terrain_noise", type='CLOUDS')
        tex.noise_scale = 1.5
        tex.noise_basis = 'ORIGINAL_PERLIN'
    except Exception:
        tex = None
    try:
        disp = terrain.modifiers.new("Displace", type='DISPLACE')
        if tex:
            disp.texture = tex
        disp.strength = displace_strength
    except Exception:
        pass
    try:
        bpy.ops.rigidbody.object_add()
        terrain.rigid_body.type = 'PASSIVE'
        terrain.rigid_body.friction = 1.0
    except Exception:
        pass
    return terrain

# -------------------------
# Drone Flight Path Generation
# -------------------------
def generate_drone_flight_path(pattern="circular", num_frames=10, center=(0,0,0), 
                              radius=30, height=80, look_at=(0,0,0)):
    """Generate drone camera positions and orientations for different flight patterns."""
    flight_path = []
    
    if pattern == "circular":
        for i in range(num_frames):
            theta = 2 * math.pi * (i / num_frames)
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            z = height + random.uniform(-3, 3)  # Altitude variation
            
            # Calculate heading (yaw) to face forward in flight direction
            heading = math.degrees(theta + math.pi/2)
            pitch = random.uniform(-45, -15)  # Downward looking
            roll = random.uniform(-5, 5)  # Slight banking
            
            flight_path.append({
                "frame": i + 1,
                "location": (x, y, z),
                "look_at": look_at,
                "heading_deg": heading,
                "pitch_deg": pitch,
                "roll_deg": roll,
                "altitude_m": z,
                "focal_mm": random.uniform(20, 50)
            })
    
    elif pattern == "grid":
        grid_size = int(math.sqrt(num_frames))
        spacing = (radius * 2) / max(1, grid_size - 1)
        
        frame_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if frame_idx >= num_frames:
                    break
                    
                x = center[0] - radius + i * spacing
                y = center[1] - radius + j * spacing
                z = height + random.uniform(-2, 2)
                
                flight_path.append({
                    "frame": frame_idx + 1,
                    "location": (x, y, z),
                    "look_at": look_at,
                    "heading_deg": random.uniform(0, 360),
                    "pitch_deg": random.uniform(-60, -20),
                    "roll_deg": random.uniform(-3, 3),
                    "altitude_m": z,
                    "focal_mm": random.uniform(24, 35)
                })
                frame_idx += 1
    
    elif pattern == "spiral":
        for i in range(num_frames):
            t = i / num_frames
            theta = 4 * math.pi * t  # Multiple spirals
            r = radius * (1 - t * 0.7)  # Spiral inward
            
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            z = height + 20 * t  # Ascending spiral
            
            flight_path.append({
                "frame": i + 1,
                "location": (x, y, z),
                "look_at": look_at,
                "heading_deg": math.degrees(theta),
                "pitch_deg": -30 - 20 * t,
                "roll_deg": random.uniform(-2, 2),
                "altitude_m": z,
                "focal_mm": 35
            })
    
    return flight_path

def create_drone_camera_from_waypoint(waypoint, cam_name="DroneCam"):
    """Create and position camera based on drone waypoint data."""
    cam_data = bpy.data.cameras.new(f"{cam_name}_data")
    cam = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.collection.objects.link(cam)
    
    # Set position
    cam.location = waypoint["location"]
    
    # Set focal length
    try:
        cam.data.lens = waypoint["focal_mm"]
    except Exception:
        pass
    
    # Set rotation (convert from heading/pitch/roll to Euler)
    heading_rad = math.radians(waypoint["heading_deg"])
    pitch_rad = math.radians(waypoint["pitch_deg"])
    roll_rad = math.radians(waypoint["roll_deg"])
    
    # Convert to Blender's coordinate system
    cam.rotation_euler = (pitch_rad + math.pi/2, 0, heading_rad)
    
    return cam

# -------------------------
# Environmental Simulation
# -------------------------
def generate_environment_series(num_frames, seed=42):
    """Generate environmental time series data."""
    random.seed(seed)
    
    # Rainfall simulation with storm events
    rainfall = [0.0] * num_frames
    
    # Add 1-3 storm events
    num_storms = random.randint(1, 3)
    for storm in range(num_storms):
        if num_frames > 10:
            start = random.randint(2, max(2, num_frames - 8))
            duration = random.randint(3, 8)
            peak_intensity = random.uniform(10, 50)  # mm per frame
            
            for i in range(start, min(num_frames, start + duration)):
                # Storm intensity curve (rise and fall)
                progress = (i - start) / duration
                if progress < 0.3:
                    intensity = peak_intensity * (progress / 0.3)
                elif progress < 0.7:
                    intensity = peak_intensity
                else:
                    intensity = peak_intensity * (1 - (progress - 0.7) / 0.3)
                
                rainfall[i] += max(0, intensity + random.gauss(0, 2))
    
    # Temperature simulation (diurnal cycle + seasonal + noise)
    temperature = []
    base_temp = 15.0
    for i in range(num_frames):
        # Diurnal cycle
        diurnal = 5 * math.sin(2 * math.pi * (i / num_frames) * 24)  # Assuming frames = hours
        # Seasonal (simplified)
        seasonal = 3 * math.sin(2 * math.pi * (i / num_frames))
        # Random noise
        noise = random.gauss(0, 1.5)
        
        temp = base_temp + diurnal + seasonal + noise
        temperature.append(temp)
    
    # Wind speed simulation
    wind_speed = []
    for i in range(num_frames):
        base_wind = 5.0
        gust = 10 * random.random() if random.random() < 0.1 else 0  # 10% chance of gusts
        wind = base_wind + gust + random.gauss(0, 2)
        wind_speed.append(max(0, wind))
    
    return {
        "rainfall": rainfall,
        "temperature": temperature,
        "wind_speed": wind_speed
    }

def save_environment_csv(env_data, output_path, num_frames):
    """Save environment data to CSV file."""
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'timestamp', 'rainfall_mm', 'temperature_c', 'wind_speed_ms'])
            
            base_time = datetime.now()
            for i in range(num_frames):
                timestamp = base_time + timedelta(hours=i)
                writer.writerow([
                    i + 1,
                    timestamp.isoformat(),
                    round(env_data["rainfall"][i], 2),
                    round(env_data["temperature"][i], 2),
                    round(env_data["wind_speed"][i], 2)
                ])
    except Exception as e:
        print(f"Failed to save environment CSV: {e}")

# -------------------------
# Geotechnical Sensor Simulation
# -------------------------
class GeotechnicalSensor:
    def __init__(self, sensor_id, x, y, depth=5.0):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.depth = depth
        self.readings = []
        self.cumulative_rainfall = 0.0
        self.last_displacement = 0.0
    
    def sample_slope_at_location(self, terrain_obj):
        """Sample terrain slope at sensor location using raycast."""
        try:
            origin = Vector((self.x, self.y, 100.0))
            direction = Vector((0, 0, -1))
            result, location, normal, index, obj, matrix = terrain_obj.ray_cast(origin, direction)
            
            if result and normal:
                # Calculate slope from normal vector
                slope_rad = math.acos(min(1.0, max(-1.0, normal.z)))
                slope_deg = math.degrees(slope_rad)
                return slope_deg, location, normal
        except Exception:
            pass
        
        return 0.0, Vector((self.x, self.y, 0)), Vector((0, 0, 1))
    
    def calculate_pore_pressure(self, rainfall_mm, temperature_c):
        """Calculate pore pressure based on rainfall and temperature."""
        self.cumulative_rainfall += rainfall_mm
        
        # Drainage factor (higher temp = more evaporation)
        drainage_factor = 0.1 + 0.05 * max(0, (temperature_c - 10) / 20)
        
        # Pore pressure model
        base_pressure = 0.2 * self.cumulative_rainfall
        drainage_loss = drainage_factor * self.cumulative_rainfall * 0.1
        
        pore_pressure = max(0, base_pressure - drainage_loss + random.gauss(0, 0.5))
        
        # Gradual drainage
        self.cumulative_rainfall *= 0.95
        
        return pore_pressure
    
    def calculate_displacement(self, slope_deg, pore_pressure, vibration_level):
        """Calculate ground displacement."""
        # Base displacement from slope and pore pressure
        stability_factor = 0.01 * slope_deg * pore_pressure
        
        # Vibration-induced displacement
        vibration_displacement = 0.1 * vibration_level
        
        # Random geological noise
        noise = random.gauss(0, 0.1)
        
        displacement = stability_factor + vibration_displacement + noise
        
        # Ensure realistic bounds
        displacement = max(-5.0, min(50.0, displacement))
        
        return displacement
    
    def calculate_strain(self, current_displacement):
        """Calculate strain from displacement change."""
        strain = abs(current_displacement - self.last_displacement) / self.depth
        self.last_displacement = current_displacement
        return strain
    
    def detect_vibrations(self, rocks, vibration_threshold=5.0, detection_radius=5.0):
        """Detect vibrations from nearby rock impacts."""
        total_vibration = 0.0
        sensor_pos = Vector((self.x, self.y, 0))
        
        for rock in rocks:
            try:
                rock_pos = Vector((rock.location.x, rock.location.y, 0))
                distance = (sensor_pos - rock_pos).length
                
                if distance < detection_radius and hasattr(rock, 'rigid_body') and rock.rigid_body:
                    # Get rock velocity (approximation)
                    if hasattr(rock.rigid_body, 'linear_velocity'):
                        velocity = rock.rigid_body.linear_velocity.length
                    else:
                        velocity = 0.0
                    
                    if velocity > vibration_threshold:
                        # Vibration intensity decreases with distance
                        vibration_intensity = velocity * (detection_radius - distance) / detection_radius
                        total_vibration += vibration_intensity
            except Exception:
                pass
        
        # Add background seismic noise
        background_noise = random.uniform(0, 0.5)
        
        return total_vibration + background_noise
    
    def record_reading(self, frame, timestamp, terrain_obj, rocks, rainfall_mm, temperature_c):
        """Record a complete sensor reading for the current frame."""
        slope_deg, location, normal = self.sample_slope_at_location(terrain_obj)
        pore_pressure = self.calculate_pore_pressure(rainfall_mm, temperature_c)
        vibration = self.detect_vibrations(rocks)
        displacement = self.calculate_displacement(slope_deg, pore_pressure, vibration)
        strain = self.calculate_strain(displacement)
        
        reading = {
            'frame': frame,
            'timestamp': timestamp,
            'slope_deg': round(slope_deg, 3),
            'rainfall_mm': round(rainfall_mm, 2),
            'pore_pressure_kpa': round(pore_pressure, 3),
            'displacement_mm': round(displacement * 1000, 3),  # Convert to mm
            'strain': round(strain, 6),
            'vibration': round(vibration, 3),
            'temperature_c': round(temperature_c, 2)
        }
        
        self.readings.append(reading)
        return reading
    
    def save_to_csv(self, output_path):
        """Save sensor readings to CSV file."""
        try:
            with open(output_path, 'w', newline='') as csvfile:
                if self.readings:
                    fieldnames = self.readings[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.readings)
        except Exception as e:
            print(f"Failed to save sensor CSV {output_path}: {e}")

def place_sensors_on_terrain(num_sensors, terrain_size=25, seed=42):
    """Place sensors at strategic locations on the terrain."""
    random.seed(seed)
    sensors = []
    
    for i in range(num_sensors):
        # Place sensors in areas likely to experience rockfall
        if i == 0:
            # Central sensor
            x, y = 0, 0
        elif i < num_sensors // 2:
            # Sensors in potential impact zones
            angle = 2 * math.pi * i / (num_sensors // 2)
            radius = random.uniform(5, 15)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
        else:
            # Random placement for coverage
            x = random.uniform(-terrain_size/2, terrain_size/2)
            y = random.uniform(-terrain_size/2, terrain_size/2)
        
        depth = random.uniform(3, 8)  # Sensor depth in meters
        sensor = GeotechnicalSensor(i + 1, x, y, depth)
        sensors.append(sensor)
    
    return sensors

# -------------------------
# Rocks + camera + light
# -------------------------
def create_rock(seed=0):
    random.seed(seed)
    size = random.uniform(0.4, 1.4)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=size, location=(0,0,0))
    rock = bpy.context.active_object
    rock.name = f"Rock_{seed}"
    # bumpy displacement
    try:
        rtex = bpy.data.textures.new(f"rock_tex_{seed}", type='STUCCI')
        rtex.noise_scale = random.uniform(0.3,1.2)
    except Exception:
        rtex = None
    try:
        mod = rock.modifiers.new("Displace", type='DISPLACE')
        if rtex:
            mod.texture = rtex
        mod.strength = random.uniform(0.1,0.6)
        bpy.ops.object.modifier_apply(modifier="Displace")
    except Exception:
        pass
    return rock

def spawn_rocks(count=12, area=18, zmin=6.0, zmax=16.0, seed=0):
    random.seed(seed)
    rocks = []
    for i in range(count):
        r = create_rock(seed + i)
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

def add_camera(location=(28, -28, 18), look_at=(0,0,0), focal_mm=35):
    cam_data = bpy.data.cameras.new("DatasetCam")
    cam = bpy.data.objects.new("DatasetCam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
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

def add_sun(strength=4.0, rotation=(math.radians(60), 0, math.radians(40))):
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = strength
    sun = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = Euler(rotation)
    return sun

# -------------------------
# Compositor outputs (robust)
# -------------------------
def setup_compositor(out_dir, base_slot_names=None):
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
    # link outputs where possible by iterating created slots and rl.outputs
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
    # enable view layer passes robustly
    vl = get_active_view_layer(scene)
    try:
        vl.use_pass_z = True
    except Exception:
        pass
    try:
        vl.use_pass_normal = True
    except Exception:
        pass
    try:
        vl.use_pass_object_index = True
    except Exception:
        pass
    return rl, file_out, base_slot_names

# -------------------------
# COCO Format Export
# -------------------------
def create_coco_annotations(annotations, width, height, categories=None):
    """Convert annotations to COCO format for machine learning compatibility."""
    if categories is None:
        categories = [{"id": 1, "name": "rock", "supercategory": "object"}]
    
    coco_format = {
        "info": {
            "description": "Synthetic Rockfall Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Blender Synthetic Dataset Generator",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [{
            "id": 1,
            "name": "Custom License",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    
    for sample in annotations:
        frame = sample["frame"]
        
        # Image info
        image_info = {
            "id": frame,
            "width": width,
            "height": height,
            "file_name": f"Image_{frame:04d}.png",
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": sample.get("timestamp", "")
        }
        coco_format["images"].append(image_info)
        
        # Rock annotations
        for rock in sample["rocks"]:
            bbox = rock["bbox_2d"]
            if bbox and len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                
                if bbox_width > 0 and bbox_height > 0:
                    annotation = {
                        "id": annotation_id,
                        "image_id": frame,
                        "category_id": 1,  # Rock category
                        "segmentation": [],  # Could be filled from index masks
                        "area": bbox_width * bbox_height,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "iscrowd": 0,
                        "attributes": {
                            "rock_name": rock["name"],
                            "pass_index": rock["pass_index"],
                            "3d_location": rock["location"],
                            "3d_rotation": rock["rotation"],
                            "velocity": rock.get("velocity", [0, 0, 0])
                        }
                    }
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1
    
    return coco_format

def save_coco_annotations(coco_data, output_path):
    """Save COCO format annotations to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"COCO annotations saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save COCO annotations: {e}")

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
def generate_dataset(out_dir, num_samples=10, width=640, height=480, seed=42, dem_path=None, dem_strength=5.0, flight_pattern="circular", num_sensors=5):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    # clear and prepare physics
    clear_scene_full()
    ensure_rigidbody_world(bpy.context.scene)

    # Enhanced DEM processing with georeferencing
    terrain = None
    dem_metadata = None
    
    if dem_path:
        try:
            print("Loading DEM with georeferencing:", dem_path)
            elevation_data, geotransform, crs, bounds = load_dem_with_georef(dem_path)
            
            if elevation_data is not None and HAS_NUMPY:
                # Create precise mesh from DEM data
                terrain = create_mesh_from_dem_data(elevation_data, geotransform, terrain_size=30)
                dem_metadata = {
                    "path": os.path.abspath(dem_path),
                    "method": "exact_mesh",
                    "geotransform": geotransform,
                    "crs": crs,
                    "bounds": list(bounds) if bounds else None,
                    "shape": list(elevation_data.shape)
                }
                print("Created exact mesh from DEM data")
            else:
                # Fallback to image displacement method
                terrain = make_terrain_from_dem(dem_path, size=30, subdiv_levels=3, disp_strength=dem_strength, apply_modifiers=False)
                dem_metadata = {
                    "path": os.path.abspath(dem_path),
                    "method": "image_displacement",
                    "subdivision_levels": 3,
                    "displacement_strength": dem_strength
                }
                print("Created terrain using image displacement method")
                
        except Exception as e:
            print("DEM load failed, falling back to procedural terrain. Error:", e)
            terrain = make_terrain_procedural(size=30, subdivisions=160, displace_strength=4.0, seed=seed)
            dem_metadata = {"method": "procedural_fallback", "error": str(e)}
    else:
        terrain = make_terrain_procedural(size=30, subdivisions=160, displace_strength=4.0, seed=seed)
        dem_metadata = {"method": "procedural"}

    # Generate drone flight path
    flight_path = generate_drone_flight_path(
        pattern=flight_pattern,
        num_frames=num_samples,
        center=(0, 0, 0),
        radius=30,
        height=80,
        look_at=(0, 0, 0)
    )
    
    # Create initial camera (will be updated per frame)
    cam = add_camera(location=(28,-28,18), look_at=(0,0,0), focal_mm=35)
    sun = add_sun(strength=4.0)
    
    # Generate environmental time series
    env_data = generate_environment_series(num_samples, seed)
    env_csv_path = os.path.join(out_dir, "env_timeseries.csv")
    save_environment_csv(env_data, env_csv_path, num_samples)
    
    # Place geotechnical sensors
    sensors = place_sensors_on_terrain(num_sensors, terrain_size=25, seed=seed)
    print(f"Placed {len(sensors)} geotechnical sensors")

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

    # Compositor outputs
    rl_node, file_out_node, base_names = setup_compositor(out_dir, ["Image","Depth","Normal","Index"])

    # set output file formats best-effort
    try:
        # configure file_out_node.format where supported
        fmt = file_out_node.format
        if fmt:
            fmt.file_format = 'PNG'
            fmt.color_mode = 'RGBA'
    except Exception:
        pass

    # spawn rocks
    rocks = spawn_rocks(count=18, area=18, zmin=6.0, zmax=16.0, seed=seed)

    # frames
    scene.frame_start = 1
    scene.frame_end = num_samples

    annotations = []
    base_time = datetime.now()
    
    for f in range(1, num_samples+1):
        scene.frame_set(f)
        frame_time = base_time + timedelta(hours=f-1)
        
        # Update camera position based on drone flight path
        if f-1 < len(flight_path):
            waypoint = flight_path[f-1]
            
            # Update camera position and orientation
            cam.location = waypoint["location"]
            
            # Set rotation from heading/pitch/roll
            heading_rad = math.radians(waypoint["heading_deg"])
            pitch_rad = math.radians(waypoint["pitch_deg"])
            roll_rad = math.radians(waypoint["roll_deg"])
            
            # Convert to Blender's coordinate system
            cam.rotation_euler = (pitch_rad + math.pi/2, roll_rad, heading_rad)
            
            # Update focal length
            try:
                cam.data.lens = waypoint["focal_mm"]
            except Exception:
                pass
        
        # set base path absolute
        file_out_node.base_path = os.path.abspath(out_dir)
        # assign clean per-frame slot paths
        for idx, slot in enumerate(file_out_node.file_slots):
            bname = base_names[idx] if idx < len(base_names) else f"slot{idx}"
            # choose extension by intended content: Depth/Normal -> EXR, Index -> PNG, Image -> PNG
            if "depth" in bname.lower() or "normal" in bname.lower():
                slot.path = f"{bname}_{f:04d}"
            else:
                slot.path = f"{bname}_{f:04d}"

        scene.render.filepath = os.path.join(out_dir, f"Image_{f:04d}.png")

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print("Render error at frame", f, ":", e)
        
        # Record sensor readings for this frame
        frame_rainfall = env_data["rainfall"][f-1] if f-1 < len(env_data["rainfall"]) else 0.0
        frame_temperature = env_data["temperature"][f-1] if f-1 < len(env_data["temperature"]) else 15.0
        
        for sensor in sensors:
            sensor.record_reading(f, frame_time.isoformat(), terrain, rocks, frame_rainfall, frame_temperature)

        # Enhanced annotations with drone metadata
        cam_data = cam.data
        try:
            f_mm = cam_data.lens
            sensor_w = getattr(cam_data, "sensor_width", 32.0)
        except Exception:
            f_mm, sensor_w = 35.0, 32.0
        fx = (f_mm / sensor_w) * width
        fy = fx
        cx = width/2.0; cy = height/2.0

        # Enhanced camera annotation with drone metadata
        cam_ann = {
            "location": list(cam.location), 
            "rotation_euler": list(cam.rotation_euler),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}
        }
        
        # Add drone-specific metadata if available
        if f-1 < len(flight_path):
            waypoint = flight_path[f-1]
            cam_ann["drone_metadata"] = {
                "altitude_m": waypoint["altitude_m"],
                "heading_deg": waypoint["heading_deg"],
                "pitch_deg": waypoint["pitch_deg"],
                "roll_deg": waypoint["roll_deg"],
                "focal_mm": waypoint["focal_mm"],
                "flight_pattern": flight_pattern
            }
            
            # Add GPS coordinates if DEM has georeferencing
            if dem_metadata and dem_metadata.get("geotransform") and dem_metadata.get("bounds"):
                # Simple approximation - map camera location to geographic coordinates
                bounds = dem_metadata["bounds"]
                terrain_size = 30  # From terrain creation
                
                # Map from terrain coordinates to geographic coordinates
                x_norm = (cam.location.x + terrain_size/2) / terrain_size
                y_norm = (cam.location.y + terrain_size/2) / terrain_size
                
                lon = bounds[0] + x_norm * (bounds[2] - bounds[0])
                lat = bounds[1] + y_norm * (bounds[3] - bounds[1])
                
                cam_ann["drone_metadata"]["gps"] = {"lat": lat, "lon": lon}
        
        # Environmental data for this frame
        env_ann = {
            "rainfall_mm": frame_rainfall,
            "temperature_c": frame_temperature,
            "wind_speed_ms": env_data["wind_speed"][f-1] if f-1 < len(env_data["wind_speed"]) else 0.0,
            "timestamp": frame_time.isoformat()
        }
        
        rocks_ann = []
        for r in rocks:
            try:
                pos = list(r.matrix_world.translation)
                rot = list(r.rotation_euler)
                pid = getattr(r, "pass_index", 0)
                bbox = bbox_2d_from_object(r, cam, scene, (width, height))
                
                # Add velocity information if available
                velocity = [0, 0, 0]
                if hasattr(r, 'rigid_body') and r.rigid_body and hasattr(r.rigid_body, 'linear_velocity'):
                    velocity = list(r.rigid_body.linear_velocity)
                    
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

        sample_ann = {
            "frame": f,
            "timestamp": frame_time.isoformat(),
            "files": {
                "image": os.path.join(out_dir, f"Image_{f:04d}.png"),
                "depth": os.path.join(out_dir, f"Depth_{f:04d}.exr"),
                "normal": os.path.join(out_dir, f"Normal_{f:04d}.exr"),
                "index": os.path.join(out_dir, f"Index_{f:04d}.png")
            },
            "camera": cam_ann,
            "rocks": rocks_ann,
            "environment": env_ann,
            "sensors_snapshot": [{
                "id": sensor.id,
                "location": [sensor.x, sensor.y],
                "latest_reading": sensor.readings[-1] if sensor.readings else None
            } for sensor in sensors]
        }
        annotations.append(sample_ann)
        # write per-frame annotation
        try:
            with open(os.path.join(out_dir, f"ann_{f:04d}.json"), "w") as jf:
                json.dump(sample_ann, jf, indent=2)
        except Exception:
            pass

    # Save sensor data to individual CSV files
    sensor_files = []
    for sensor in sensors:
        sensor_csv_path = os.path.join(out_dir, f"sensor_{sensor.id:02d}_timeseries.csv")
        sensor.save_to_csv(sensor_csv_path)
        sensor_files.append({
            "id": sensor.id,
            "file": f"sensor_{sensor.id:02d}_timeseries.csv",
            "location": [sensor.x, sensor.y],
            "depth_m": sensor.depth
        })
    
    # Enhanced master index with all metadata
    dataset_metadata = {
        "dataset_info": {
            "created": datetime.now().isoformat(),
            "num_samples": num_samples,
            "resolution": {"width": width, "height": height},
            "seed": seed,
            "flight_pattern": flight_pattern
        },
        "dem": dem_metadata,
        "environment": {
            "file": "env_timeseries.csv",
            "variables": ["rainfall_mm", "temperature_c", "wind_speed_ms"]
        },
        "sensors": sensor_files,
        "flight_path": flight_path,
        "samples": annotations,
        "output_files": {
            "coco_annotations": "annotations_coco.json",
            "environment_data": "env_timeseries.csv",
            "sensor_data": [f"sensor_{i+1:02d}_timeseries.csv" for i in range(len(sensors))]
        }
    }
    
    try:
        with open(os.path.join(out_dir, "dataset_index.json"), "w") as mf:
            json.dump(dataset_metadata, mf, indent=2)
    except Exception as e:
        print(f"Failed to save dataset index: {e}")

    print("Done. Enhanced dataset written to:", os.path.abspath(out_dir))
    print(f"Generated {num_samples} samples with:")
    print(f"  - DEM method: {dem_metadata.get('method', 'unknown')}")
    print(f"  - Flight pattern: {flight_pattern}")
    print(f"  - {len(sensors)} geotechnical sensors")
    print(f"  - Environmental time series")
    print(f"  - Enhanced annotations with drone metadata")
    
    # Generate COCO format annotations
    try:
        coco_data = create_coco_annotations(annotations, width, height)
        coco_path = os.path.join(out_dir, "annotations_coco.json")
        save_coco_annotations(coco_data, coco_path)
        print(f"  - COCO format annotations: {len(coco_data['annotations'])} objects")
    except Exception as e:
        print(f"Warning: Failed to generate COCO annotations: {e}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    out_dir, num_samples, width, height, seed, dem_path, dem_strength, flight_pattern, num_sensors = parse_argv()
    print("OUT:", out_dir, "N:", num_samples, "RES:", f"{width}x{height}", "seed:", seed)
    print("DEM:", dem_path, "DEM_strength:", dem_strength)
    print("Flight pattern:", flight_pattern, "Sensors:", num_sensors)
    generate_dataset(
        out_dir, 
        num_samples=num_samples, 
        width=width, 
        height=height, 
        seed=seed, 
        dem_path=dem_path, 
        dem_strength=dem_strength,
        flight_pattern=flight_pattern,
        num_sensors=num_sensors
    )
