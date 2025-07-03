import time
import tempfile
import os
import json
from PIL import Image
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed

import configargparse
import gradio as gr
import jax

import numpy as np
import pyvista as pv

# --- Local Imports ---
from IK_Helper import (
    deform_mesh,
    load_mesh_data_from_gltf,
    load_mesh_data_from_urdf,
)
from IK_Hand_Specification import HandSpecification
from IK_SMPLX_Statics import left_arm_bounds_dict, right_arm_bounds_dict, complete_full_body_bounds_dict
from IK_jax import InverseKinematicsSolver
from IK_objectives_jax import (
    BoneZeroRotationObj,
    CombinedDerivativeObj,
    DistanceObjTraj,
    SphereCollisionPenaltyObjTraj,
)

class IKGradioApp:
    def __init__(self, args):
        self.args = args

        # Solver caching to avoid recreating solvers
        self.solver_cache = {}  # Cache for virtual agent solvers
        self.urdf_solver_cache = {}  # Cache for URDF robot solvers

        # Initialize both demos
        self.init_virtual_agent_demo()
        self.init_urdf_robot_demo()

        # Animation parameters
        self.animation_fps = 10
        self.animation_duration_frames = 20
        self.temp_gif_files = []
        self.n_jobs = min(24, os.cpu_count())

    def get_solver_cache_key(self, controlled_bones):
        """Generate a cache key for a given set of controlled bones."""
        return tuple(sorted(controlled_bones))

    def get_cached_solver(self, controlled_bones, is_urdf=False):
        """Get or create a cached solver for the given bone configuration."""
        cache_key = self.get_solver_cache_key(controlled_bones)
        cache = self.urdf_solver_cache if is_urdf else self.solver_cache

        if cache_key in cache:
            print(f"Using cached solver for bones: {controlled_bones}")
            return cache[cache_key]

        print(f"Creating new solver for bones: {controlled_bones}")

        if is_urdf:
            # Create URDF solver
            solver = InverseKinematicsSolver(
                model_file=self.urdf_file,
                controlled_bones=controlled_bones,
                bounds=None,  # Use URDF limits
                threshold=self.args.threshold,
                num_steps=self.args.num_steps,
            )
        else:
            # Extract bounds for selected bones
            bounds = []
            for bone_name in controlled_bones:
                if bone_name in self.bounds_dict:
                    lower, upper = self.bounds_dict[bone_name]
                    for i in range(3):
                        bounds.append((lower[i], upper[i]))
                else:
                    bounds.extend([(-90, 90), (-90, 90), (-90, 90)])

            # Create virtual agent solver
            solver = InverseKinematicsSolver(
                model_file=self.args.gltf_file,
                controlled_bones=controlled_bones,
                bounds=bounds,
                threshold=self.args.threshold,
                num_steps=self.args.num_steps,
            )

        # Cache the solver
        cache[cache_key] = solver
        return solver

    def init_virtual_agent_demo(self):
        """Initialize the virtual agent (GLTF) demo."""
        # Create a basic solver to get available bones
        basic_solver = InverseKinematicsSolver(
            model_file=self.args.gltf_file,
            controlled_bones=["left_collar"],  # Minimal bones for initialization
            bounds=[(-90, 90), (-90, 90), (-90, 90)],
            threshold=self.args.threshold,
            num_steps=self.args.num_steps,
        )

        # Get all available bones for selection
        self.available_bones = basic_solver.fk_solver.bone_names

        # Setup bounds and controlled bones based on hand
        if self.args.hand == "left":
            self.bounds_dict = complete_full_body_bounds_dict
            self.default_controlled_bones = list(left_arm_bounds_dict.keys())
            self.default_end_effector = "left_index3"
        else:
            self.bounds_dict = complete_full_body_bounds_dict
            self.default_controlled_bones = list(right_arm_bounds_dict.keys())
            self.default_end_effector = "right_index3"

        # Filter available bones to only include those in bounds_dict
        self.selectable_bones = [bone for bone in self.available_bones if bone in self.bounds_dict]

        # Initialize with default configuration
        self.current_controlled_bones = self.default_controlled_bones.copy()
        self.current_end_effector = self.default_end_effector

        # Get the initial solver from cache
        self.solver = self.get_cached_solver(self.current_controlled_bones, is_urdf=False)

        self.initial_rotations = np.zeros(len(self.solver.controlled_bones) * 3, dtype=np.float32)
        self.best_angles = self.initial_rotations.copy()

        # Load mesh data for virtual agent
        self.mesh_data = load_mesh_data_from_gltf(self.args.gltf_file, self.solver.fk_solver)
        vertices = np.asarray(self.mesh_data["vertices"])[:, :3]
        faces = self.mesh_data["faces"]
        pv_faces = np.hstack((np.full((faces.shape[0], 1), 3, dtype=int), faces))
        self.pv_mesh = pv.PolyData(vertices, pv_faces)

        # Initialize objectives for virtual agent
        self.setup_virtual_agent_objectives()

    def init_urdf_robot_demo(self):
        """Initialize the URDF robot demo."""
        # Default URDF robot configuration
        self.urdf_file = "/home/mei/Downloads/robots/pepper_description-master/urdf/pepper.urdf"

        # Create a basic solver to get available bones
        basic_urdf_solver = InverseKinematicsSolver(
            model_file=self.urdf_file,
            controlled_bones=["LShoulder"],  # Minimal bones for initialization
            bounds=None,
            threshold=self.args.threshold,
            num_steps=self.args.num_steps,
        )

        # Get all available bones for URDF robot
        self.urdf_available_bones = basic_urdf_solver.fk_solver.bone_names

        # Default configuration
        self.urdf_default_controlled_bones = ["LShoulder", "LBicep","LElbow", "LForeArm", "l_wrist"]
        self.urdf_default_end_effector = "LFinger13_link"

        # Filter to bones that exist in the robot
        self.urdf_selectable_bones = [bone for bone in self.urdf_available_bones]

        # Initialize with default configuration
        self.urdf_current_controlled_bones = self.urdf_default_controlled_bones.copy()
        self.urdf_current_end_effector = self.urdf_default_end_effector

        # Get the initial solver from cache
        self.urdf_solver = self.get_cached_solver(self.urdf_current_controlled_bones, is_urdf=True)

        self.urdf_initial_rotations = np.zeros(len(self.urdf_solver.controlled_bones) * 3, dtype=np.float32)
        self.urdf_best_angles = self.urdf_initial_rotations.copy()

        # Load mesh data for URDF robot
        self.urdf_mesh_data = load_mesh_data_from_urdf(self.urdf_file, self.urdf_solver.fk_solver)
        if self.urdf_mesh_data:
            vertices = np.asarray(self.urdf_mesh_data["vertices"])[:, :3]
            faces = self.urdf_mesh_data["faces"]
            pv_faces = np.hstack((np.full((faces.shape[0], 1), 3, dtype=int), faces))
            self.urdf_pv_mesh = pv.PolyData(vertices, pv_faces)
        else:
            # Fallback empty mesh
            self.urdf_pv_mesh = pv.PolyData()

        # Initialize objectives for URDF robot
        self.setup_urdf_robot_objectives()

    def setup_virtual_agent_objectives(self):
        """Initialize default objectives for virtual agent."""
        target_point = np.array([0.0, 0.2, 0.35])
        self.distance_obj = DistanceObjTraj(
            target_points=target_point,
            bone_name=self.current_end_effector,
            use_head=True,
            weight=1.0,
        )

        sphere_collider = {"center": [0.1, 0.0, 0.35], "radius": 0.1}
        self.collision_obj = SphereCollisionPenaltyObjTraj(sphere_collider, min_clearance=0.0, weight=1.0)
        self.collision_enabled = False

    def setup_urdf_robot_objectives(self):
        """Initialize default objectives for URDF robot."""
        target_point = np.array([0.3, 0.3, 0.35])
        self.urdf_distance_obj = DistanceObjTraj(
            target_points=target_point,
            bone_name=self.urdf_current_end_effector,
            use_head=True,
            weight=1.0,
        )

        sphere_collider = {"center": [0.2, 0.0, 0.35], "radius": 0.1}
        self.urdf_collision_obj = SphereCollisionPenaltyObjTraj(sphere_collider, min_clearance=0.0, weight=1.0)
        self.urdf_collision_enabled = False

    def update_virtual_agent_configuration(self, end_effector, *bone_selections):
        """Update virtual agent solver configuration based on UI selections."""
        # Get selected bones from checkboxes
        selected_bones = []
        for i, bone_name in enumerate(self.selectable_bones):
            if i < len(bone_selections) and bone_selections[i]:
                selected_bones.append(bone_name)

        # Use defaults if nothing selected
        if not selected_bones:
            selected_bones = self.default_controlled_bones

        # Validate end effector
        if end_effector not in self.available_bones:
            end_effector = self.default_end_effector

        # Check if controlled bones changed
        bones_changed = selected_bones != self.current_controlled_bones
        end_effector_changed = end_effector != self.current_end_effector

        if bones_changed or end_effector_changed:
            # Update controlled bones if they changed
            if bones_changed:
                print(f"Controlled bones changed from {self.current_controlled_bones} to {selected_bones}")
                self.current_controlled_bones = selected_bones

                # Get solver from cache (creates if not exists)
                self.solver = self.get_cached_solver(self.current_controlled_bones, is_urdf=False)

                # Reset rotations for new bone configuration
                self.initial_rotations = np.zeros(len(self.solver.controlled_bones) * 3, dtype=np.float32)
                self.best_angles = self.initial_rotations.copy()

            # Update end effector (this is just a parameter change, no solver recreation needed)
            if end_effector_changed:
                print(f"End effector changed from {self.current_end_effector} to {end_effector}")
                self.current_end_effector = end_effector

            # Update objectives with new end effector
            self.setup_virtual_agent_objectives()

            return f"Updated: {len(self.current_controlled_bones)} bones, end-effector: {self.current_end_effector}"

        return "Configuration unchanged"

    def update_urdf_robot_configuration(self, end_effector, *bone_selections):
        """Update URDF robot solver configuration based on UI selections."""
        # Get selected bones from checkboxes
        selected_bones = []
        for i, bone_name in enumerate(self.urdf_selectable_bones):
            if i < len(bone_selections) and bone_selections[i]:
                selected_bones.append(bone_name)

        # Use defaults if nothing selected
        if not selected_bones:
            selected_bones = self.urdf_default_controlled_bones

        # Validate end effector
        if end_effector not in self.urdf_available_bones:
            end_effector = self.urdf_default_end_effector

        # Check if controlled bones changed
        bones_changed = selected_bones != self.urdf_current_controlled_bones
        end_effector_changed = end_effector != self.urdf_current_end_effector

        if bones_changed or end_effector_changed:
            # Update controlled bones if they changed
            if bones_changed:
                print(f"URDF controlled bones changed from {self.urdf_current_controlled_bones} to {selected_bones}")
                self.urdf_current_controlled_bones = selected_bones

                # Get solver from cache (creates if not exists)
                self.urdf_solver = self.get_cached_solver(self.urdf_current_controlled_bones, is_urdf=True)

                # Reset rotations for new bone configuration
                self.urdf_initial_rotations = np.zeros(len(self.urdf_solver.controlled_bones) * 3, dtype=np.float32)
                self.urdf_best_angles = self.urdf_initial_rotations.copy()

            # Update end effector (this is just a parameter change, no solver recreation needed)
            if end_effector_changed:
                print(f"URDF end effector changed from {self.urdf_current_end_effector} to {end_effector}")
                self.urdf_current_end_effector = end_effector

            # Update objectives with new end effector
            self.setup_urdf_robot_objectives()

            return f"Updated: {len(self.urdf_current_controlled_bones)} bones, end-effector: {self.urdf_current_end_effector}"

        return "Configuration unchanged"

    def create_single_frame_data(self, angles, show_skeleton=False):
        """Create frame data that can be safely passed to parallel workers."""
        return {
            "angles": angles.copy() if hasattr(angles, "copy") else np.array(angles),
            "show_skeleton": show_skeleton,
            "mesh_vertices": self.pv_mesh.points.copy(),
            "mesh_faces": self.pv_mesh.faces.copy(),
            "target_points": [np.array(pt) for pt in self.distance_obj.target_points],
            "collision_center": np.array(self.collision_obj.center),
            "collision_radius": float(self.collision_obj.radius),
        }

    def create_single_frame_data_urdf(self, angles, show_skeleton=False):
        """Create frame data for URDF robot."""
        return {
            "angles": angles.copy() if hasattr(angles, "copy") else np.array(angles),
            "show_skeleton": show_skeleton,
            "mesh_vertices": self.urdf_pv_mesh.points.copy() if self.urdf_pv_mesh.points.size > 0 else np.array([]),
            "mesh_faces": self.urdf_pv_mesh.faces.copy() if self.urdf_pv_mesh.faces.size > 0 else np.array([]),
            "target_points": [np.array(pt) for pt in self.urdf_distance_obj.target_points],
            "collision_center": np.array(self.urdf_collision_obj.center),
            "collision_radius": float(self.urdf_collision_obj.radius),
            "is_urdf": True,
        }

    def create_frame_from_data(self, frame_data):
        """Create a single frame from frame data - safe for parallel execution."""
        plotter = None
        try:
            plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
            plotter.clear()

            is_urdf = frame_data.get("is_urdf", False)

            # Simplified lighting for faster rendering
            plotter.add_light(pv.Light(position=(2, 2, 2), focal_point=(0, 0, 0), color="white", intensity=1.0))
            plotter.add_light(pv.Light(position=(-2, -2, 2), focal_point=(0, 0, 0), color="white", intensity=0.7))
            plotter.add_light(pv.Light(position=(0, 0, -2), focal_point=(0, 0, 0), color="white", intensity=0.3))
            plotter.camera_position = [0.0, 0.0, 2.0]
            plotter.camera.focal_point = [0.0, 0.0, 0.0]
            plotter.camera.up = [0.0, 1.0, 0.0]
            plotter.camera.view_angle = 45

            # Handle mesh rendering based on type

            if len(frame_data["mesh_vertices"]) > 0 and len(frame_data["mesh_faces"]) > 0:
                # Create mesh from stored data
                mesh = pv.PolyData(frame_data["mesh_vertices"], frame_data["mesh_faces"])

                # Deform mesh with current angles
                if is_urdf:
                    deformed_verts = deform_mesh(frame_data["angles"], self.urdf_solver.fk_solver, self.urdf_mesh_data)
                else:
                    deformed_verts = deform_mesh(frame_data["angles"], self.solver.fk_solver, self.mesh_data)
                mesh.points = np.asarray(deformed_verts)
                plotter.add_mesh(mesh, color="lightblue", show_edges=False, smooth_shading=True)

            if frame_data["show_skeleton"]:
                if is_urdf:
                    fk_transforms = self.urdf_solver.fk_solver.compute_fk_from_angles(frame_data["angles"])
                    for bone_name in self.urdf_solver.fk_solver.bone_names:
                        try:
                            head, tail = self.urdf_solver.fk_solver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
                            line_points = np.array([np.asarray(head), np.asarray(tail)])
                            plotter.add_lines(line_points, color="blue", width=3)
                            sphere = pv.Sphere(radius=0.01, center=np.asarray(head))
                            plotter.add_mesh(sphere, color="red")
                        except:
                            continue
                else:
                    fk_transforms = self.solver.fk_solver.compute_fk_from_angles(frame_data["angles"])
                    for bone_name in self.solver.fk_solver.bone_names:
                        head, tail = self.solver.fk_solver.get_bone_head_tail_from_fk(fk_transforms, bone_name)
                        line_points = np.array([np.asarray(head), np.asarray(tail)])
                        plotter.add_lines(line_points, color="blue", width=3)
                        sphere = pv.Sphere(radius=0.01, center=np.asarray(head))
                        plotter.add_mesh(sphere, color="red")

            # Add target points
            for pt in frame_data["target_points"]:
                target_sphere = pv.Sphere(radius=0.02, center=np.asarray(pt))
                plotter.add_mesh(target_sphere, color="green")

            collision_enabled = frame_data.get("collision_enabled", False)
            if is_urdf:
                collision_enabled = self.urdf_collision_enabled
            else:
                collision_enabled = self.collision_enabled

            if collision_enabled:
                # Add collision spheres
                collision_sphere = pv.Sphere(
                    radius=frame_data["collision_radius"],
                    center=frame_data["collision_center"],
                )
                plotter.add_mesh(collision_sphere, color="yellow", opacity=0.5)

            # Render to numpy array
            img_array = plotter.screenshot(transparent_background=False, return_img=True)
            plotter.clear()

        except Exception as e:
            print(f"Error creating frame: {e}")
            return np.zeros((300, 400, 3), dtype=np.uint8)
        finally:
            if plotter is not None:
                try:
                    plotter.close()
                except:
                    pass
                del plotter

        return img_array

    def create_single_frame(self, angles, show_skeleton=False, is_urdf=False):
        """Create a single frame of the visualization - wrapper for compatibility."""
        if is_urdf:
            frame_data = self.create_single_frame_data_urdf(angles, show_skeleton)
        else:
            frame_data = self.create_single_frame_data(angles, show_skeleton)
        return self.create_frame_from_data(frame_data)

    def create_animated_gif(self, trajectory, show_skeleton=False, is_urdf=False):
        """Create an animated GIF from a trajectory using parallel processing."""
        try:
            if len(trajectory) < 2:
                # Fall back to single frame if not enough points
                return self.create_single_frame(trajectory[-1], show_skeleton, is_urdf)

            print(f"Creating animated GIF with {len(trajectory)} trajectory points...")

            # Create smooth interpolated trajectory for animation
            original_t = np.linspace(0, 1, len(trajectory))
            animation_t = np.linspace(0, 1, self.animation_duration_frames)

            # Interpolate each DOF using cubic spline
            interpolated_trajectory = []
            for i in range(trajectory.shape[1]):  # For each DOF
                spline = CubicSpline(original_t, trajectory[:, i])
                interpolated_values = spline(animation_t)
                interpolated_trajectory.append(interpolated_values)

            # Transpose to get frames x DOF
            smooth_trajectory = np.array(interpolated_trajectory).T

            print(f"Creating {len(smooth_trajectory)} animation frames using {self.n_jobs} parallel workers...")

            # Prepare frame data for parallel processing
            frame_data_list = []
            for frame_angles in smooth_trajectory:
                if is_urdf:
                    frame_data = self.create_single_frame_data_urdf(frame_angles, show_skeleton)
                else:
                    frame_data = self.create_single_frame_data(frame_angles, show_skeleton)
                frame_data_list.append(frame_data)

            # Create frames in parallel
            start_time = time.time()
            frame_arrays = Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(self.create_frame_from_data)(frame_data) for frame_data in frame_data_list)
            parallel_time = time.time() - start_time
            print(f"Parallel frame creation completed in {parallel_time:.2f}s")

            # Convert to PIL Images
            frames = []
            for frame_array in frame_arrays:
                if frame_array is not None:
                    pil_img = Image.fromarray(frame_array)
                    frames.append(pil_img)

            if not frames:
                print("No frames created, falling back to single frame")
                return self.create_single_frame(trajectory[-1], show_skeleton, is_urdf)

            # Add multiple copies of the last frame to ensure the animation stops
            last_frame = frames[-1]
            num_stop_frames = 10
            for _ in range(num_stop_frames):
                frames.append(last_frame.copy())

            # Create temporary GIF file with a unique name
            gif_fd, gif_path = tempfile.mkstemp(suffix=".gif", prefix="ik_animation_")
            os.close(gif_fd)

            # Clean up old GIF files
            self.cleanup_temp_files()

            # Store this GIF for later cleanup
            self.temp_gif_files.append(gif_path)

            print(f"Saving GIF to: {gif_path}")

            # Create different frame durations
            durations = []
            normal_duration = 1000 // self.animation_fps
            for i in range(len(frames)):
                if i < len(frames) - num_stop_frames:
                    durations.append(normal_duration)
                else:
                    durations.append(2000)

            # Save as animated GIF
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=1,
                optimize=True,
                disposal=2,
            )

            total_time = time.time() - start_time
            print(f"GIF creation completed successfully in {total_time:.2f}s total")
            return gif_path

        except Exception as e:
            print(f"Error creating animated GIF: {e}")
            # Fallback to single frame on error
            final_angles = trajectory[-1] if len(trajectory) > 0 else (self.urdf_best_angles if is_urdf else self.best_angles)
            return self.create_single_frame(final_angles, show_skeleton, is_urdf)

    def cleanup_temp_files(self):
        """Clean up old temporary GIF files."""
        for gif_path in self.temp_gif_files[:]:  # Create a copy of the list
            try:
                if os.path.exists(gif_path):
                    os.unlink(gif_path)
                self.temp_gif_files.remove(gif_path)
            except Exception as e:
                print(f"Warning: Could not clean up temp file {gif_path}: {e}")

    def create_visualization_image(self, show_skeleton=False, trajectory=None, is_urdf=False):
        """Create 3D visualization and return as numpy array or GIF path."""
        best_angles = self.urdf_best_angles if is_urdf else self.best_angles
        if best_angles is None:
            return np.zeros((300, 400, 3), dtype=np.uint8)

        # If trajectory is provided and has multiple points, create animated GIF
        if trajectory is not None and len(trajectory) > 1:
            print(f"Creating animation for trajectory with {len(trajectory)} points")
            return self.create_animated_gif(trajectory, show_skeleton, is_urdf)
        else:
            # Single frame for static image
            return self.create_single_frame(best_angles, show_skeleton, is_urdf)

    def on_image_click(
        self, evt: gr.SelectData, target_z, subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, hand_shape, hand_position, show_skeleton
    ):
        """Handle clicks on the virtual agent visualization image."""
        # Get click coordinates in image space (0 to image dimensions)
        click_x, click_y = evt.index[0], evt.index[1]
        print(f"Clicked at: ({click_x}, {click_y})")

        # Get image dimensions - PyVista default is 800x600
        img_width, img_height = 800, 600

        # Convert to normalized coordinates (0 to 1)
        norm_x = click_x / img_width
        norm_y = click_y / img_height

        # Convert to world coordinates (approximate mapping)
        world_x = (norm_x - 0.5) * 1.5  # Scale and center
        world_y = (0.5 - norm_y) * 1.5  # Flip Y and scale

        # Use clicked coordinates and provided target_z
        new_target_x = world_x
        new_target_y = world_y
        new_target_z = target_z

        # Solve with updated coordinates
        viz_result, status = self.solve_ik(
            new_target_x, new_target_y, new_target_z, subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, hand_shape, hand_position, show_skeleton
        )

        # Return updates but don't trigger change events by using gr.update with specific value
        return viz_result, status, gr.update(value=new_target_x), gr.update(value=new_target_y)

    def on_urdf_image_click(
        self, evt: gr.SelectData, target_z, subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, show_skeleton
    ):
        """Handle clicks on the URDF robot visualization image."""
        # Get click coordinates and convert to world coordinates (same logic as virtual agent)
        click_x, click_y = evt.index[0], evt.index[1]
        print(f"URDF Robot clicked at: ({click_x}, {click_y})")

        img_width, img_height = 800, 600
        norm_x = click_x / img_width
        norm_y = click_y / img_height
        world_x = (norm_x - 0.5) * 1.5
        world_y = (0.5 - norm_y) * 1.5

        new_target_x = world_x
        new_target_y = world_y
        new_target_z = target_z

        # Solve with updated coordinates for URDF robot
        viz_result, status = self.solve_urdf_ik(
            new_target_x, new_target_y, new_target_z, subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, show_skeleton
        )

        return viz_result, status, gr.update(value=new_target_x), gr.update(value=new_target_y)

    def solve_ik(self, *args):
        """Solve IK with current parameters and return visualization."""
        (
            target_x,
            target_y,
            target_z,
            subpoints,
            distance_weight,
            collision_weight,
            distance_enabled,
            collision_enabled,
            bone_zero_enabled,
            bone_zero_weight,
            derivative_enabled,
            derivative_weight,
            hand_shape,
            hand_position,
            show_skeleton,
        ) = args

        try:
            new_target = np.array([target_x, target_y, target_z])
            self.distance_obj.update_params({"bone_name": self.current_end_effector, "use_head": True, "target_points": new_target, "weight": distance_weight})
            self.collision_obj.update_params({"weight": collision_weight})

            mandatory_fns, optional_fns = [], []
            if distance_enabled:
                mandatory_fns.append(self.distance_obj)
            if collision_enabled:
                optional_fns.append(self.collision_obj)
            self.collision_enabled = collision_enabled

            # Add optional objective functions based on toggles
            if bone_zero_enabled:
                optional_fns.append(BoneZeroRotationObj(weight=bone_zero_weight))

            if derivative_enabled and subpoints > 1:
                optional_fns.append(CombinedDerivativeObj(max_order=3, weights=[derivative_weight] * 3))
            elif not bone_zero_enabled and not derivative_enabled:
                # Fallback minimal regularization if no objectives are enabled
                optional_fns.append(BoneZeroRotationObj(weight=0.01))

            hand_spec_params = {
                "is_pointing": hand_shape == "Pointing",
                "is_shaping": hand_shape == "Shaping",
                "is_flat": hand_shape == "Flat",
                "look_forward": hand_position == "Look Forward",
                "look_45_up": hand_position == "Look 45° Up",
                "look_45_down": hand_position == "Look 45° Down",
                "look_up": hand_position == "Look Up",
                "look_down": hand_position == "Look Down",
                "look_45_x_downwards": hand_position == "Look 45° X Downwards",
                "look_45_x_upwards": hand_position == "Look 45° X Upwards",
                "look_x_inward": hand_position == "Look X Inward",
                "look_to_body": hand_position == "Look to Body",
                "arm_down": hand_position == "Arm Down",
                "arm_45_down": hand_position == "Arm 45° Down",
                "arm_flat": hand_position == "Arm Flat",
            }

            if any(hand_spec_params.values()):
                hand_spec = HandSpecification(**hand_spec_params)
                spec_objectives = hand_spec.get_objectives(
                    left_hand=self.args.hand == "left",
                    controlled_bones=self.current_controlled_bones,
                    full_trajectory=subpoints > 1,
                    last_position=True,
                    weight=0.5,
                )
                optional_fns.extend(spec_objectives)

            print(f"Starting IK solve with {subpoints} subpoints...")
            start_time = time.time()
            best_angles, obj_value, steps = self.solver.solve(
                initial_rotations=self.initial_rotations,
                learning_rate=self.args.learning_rate,
                mandatory_objective_functions=tuple(mandatory_fns),
                optional_objective_functions=tuple(optional_fns),
                ik_points=subpoints,
                verbose=False,
            )
            solve_time = time.time() - start_time
            print(f"IK solve completed in {solve_time:.2f}s")

            # Store the final angles for next iteration
            self.best_angles = best_angles[-1]
            self.initial_rotations = self.best_angles

            # Create visualization based on subpoints
            if subpoints > 1:
                print("Creating animated visualization...")
                # Create animated GIF for trajectory
                viz_result = self.create_visualization_image(show_skeleton, trajectory=best_angles)
            else:
                print("Creating static visualization...")
                # Create static image for single point
                viz_result = self.create_visualization_image(show_skeleton)

            status = f"Solved in {solve_time:.2f}s, {steps} iterations, objective: {obj_value:.6f}"
            print(f"Visualization created. Status: {status}")
            return viz_result, status

        except Exception as e:
            print(f"Error in solve_ik: {e}")
            error_status = f"Error: {str(e)}"
            # Return a fallback image on error
            fallback_img = self.create_single_frame(self.best_angles if self.best_angles is not None else self.initial_rotations, show_skeleton)
            return fallback_img, error_status

    def solve_urdf_ik(self, *args):
        """Solve IK for URDF robot with current parameters and return visualization."""
        (
            target_x,
            target_y,
            target_z,
            subpoints,
            distance_weight,
            collision_weight,
            distance_enabled,
            collision_enabled,
            bone_zero_enabled,
            bone_zero_weight,
            derivative_enabled,
            derivative_weight,
            show_skeleton,
        ) = args

        try:
            new_target = np.array([target_x, target_y, target_z])
            self.urdf_distance_obj.update_params({"bone_name": self.urdf_current_end_effector, "target_points": new_target, "weight": distance_weight})
            self.urdf_collision_obj.update_params({"weight": collision_weight})

            mandatory_fns, optional_fns = [], []
            if distance_enabled:
                mandatory_fns.append(self.urdf_distance_obj)
            if collision_enabled:
                optional_fns.append(self.urdf_collision_obj)
            self.urdf_collision_enabled = collision_enabled

            # Add optional objective functions based on toggles
            if bone_zero_enabled:
                optional_fns.append(BoneZeroRotationObj(weight=bone_zero_weight))

            if derivative_enabled and subpoints > 1:
                optional_fns.append(CombinedDerivativeObj(max_order=3, weights=[derivative_weight] * 3))
            elif not bone_zero_enabled and not derivative_enabled:
                # Fallback minimal regularization if no objectives are enabled
                optional_fns.append(BoneZeroRotationObj(weight=0.01))

            print(f"Starting URDF IK solve with {subpoints} subpoints...")
            start_time = time.time()
            best_angles, obj_value, steps = self.urdf_solver.solve(
                initial_rotations=self.urdf_initial_rotations,
                learning_rate=self.args.learning_rate,
                mandatory_objective_functions=tuple(mandatory_fns),
                optional_objective_functions=tuple(optional_fns),
                ik_points=subpoints,
                verbose=False,
            )
            solve_time = time.time() - start_time
            print(f"URDF IK solve completed in {solve_time:.2f}s")

            # Store the final angles for next iteration
            self.urdf_best_angles = best_angles[-1]
            self.urdf_initial_rotations = self.urdf_best_angles

            # Create visualization based on subpoints
            if subpoints > 1:
                print("Creating animated URDF visualization...")
                viz_result = self.create_visualization_image(show_skeleton, trajectory=best_angles, is_urdf=True)
            else:
                print("Creating static URDF visualization...")
                viz_result = self.create_visualization_image(show_skeleton, is_urdf=True)

            status = f"URDF Robot solved in {solve_time:.2f}s, {steps} iterations, objective: {obj_value:.6f}"
            print(f"URDF Visualization created. Status: {status}")
            return viz_result, status

        except Exception as e:
            print(f"Error in solve_urdf_ik: {e}")
            error_status = f"URDF Error: {str(e)}"
            fallback_img = self.create_single_frame(
                self.urdf_best_angles if self.urdf_best_angles is not None else self.urdf_initial_rotations,
                show_skeleton,
                is_urdf=True
            )
            return fallback_img, error_status

    def create_virtual_agent_tab(self):
        """Create the virtual agent demo tab."""
        with gr.Row():
            with gr.Column(scale=2):
                viz_image = gr.Image(value=self.create_visualization_image(), label="3D Visualization - Click to set target position", interactive=False)
                status_text = gr.Textbox(value="Ready", label="Status", interactive=False)

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🎯 Target & IK"):
                        with gr.Group():
                            gr.Markdown("### Target Position")
                            target_x = gr.Number(value=0.0, label="Target X", step=0.01, precision=3, interactive=True)
                            target_y = gr.Number(value=0.2, label="Target Y", step=0.01, precision=3, interactive=True)
                            target_z = gr.Number(
                                value=0.35,
                                label="Target Z",
                                step=0.01,
                                precision=3,
                            )
                        with gr.Group():
                            gr.Markdown("### IK Parameters")
                            gr.Markdown("Please note that Gif generation is very slow.")
                            subpoints = gr.Slider(
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=self.args.subpoints,
                                label="Trajectory Subpoints \n(>1 creates animated GIF)",
                            )
                            show_skeleton = gr.Checkbox(label="Show Skeleton", value=False)
                    with gr.TabItem("🔧 Objectives"):
                        with gr.Group():
                            gr.Markdown("### Primary Objective Functions")
                            distance_enabled = gr.Checkbox(label="Distance Objective", value=True)
                            gr.Markdown("*Drives the end-effector toward the target position*")
                            distance_weight = gr.Number(value=1.0, label="Distance Weight", step=0.1)

                            collision_enabled = gr.Checkbox(label="Collision Avoidance", value=False)
                            gr.Markdown("*Prevents bone segments from intersecting with collision spheres*")
                            collision_weight = gr.Number(value=1.0, label="Collision Weight", step=0.1)

                        with gr.Group():
                            gr.Markdown("### Regularization Objective Functions")
                            bone_zero_enabled = gr.Checkbox(label="Bone Zero Rotation", value=True)
                            gr.Markdown("*Keeps joint angles close to zero for natural poses*")
                            bone_zero_weight = gr.Number(value=0.05, label="Bone Zero Weight", step=0.01)

                            derivative_enabled = gr.Checkbox(label="Trajectory Smoothing", value=True)
                            gr.Markdown("*Smooths velocity, acceleration, and jerk for natural motion (only for trajectories)*")
                            derivative_weight = gr.Number(value=0.05, label="Derivative Weight", step=0.01)
                    with gr.TabItem("✋ Hand Spec"):
                        with gr.Group():
                            gr.Markdown("### Hand Shape")
                            hand_shape = gr.Radio(["None", "Pointing", "Shaping", "Flat"], label="Shape", value="None")
                        with gr.Group():
                            gr.Markdown("### Hand Position/Direction")
                            hand_position = gr.Radio(
                                [
                                    "None",
                                    "Look Forward",
                                    "Look 45° Up",
                                    "Look 45° Down",
                                    "Look Up",
                                    "Look Down",
                                    "Look 45° X Downwards",
                                    "Look 45° X Upwards",
                                    "Look X Inward",
                                    "Look to Body",
                                    "Arm Down",
                                    "Arm 45° Down",
                                    "Arm Flat",
                                ],
                                label="Position",
                                value="None",
                            )
                    with gr.TabItem("⚙️ Configuration"):
                        with gr.Group():
                            gr.Markdown("### End Effector")
                            end_effector_dropdown = gr.Dropdown(
                                choices=self.available_bones,
                                value=self.current_end_effector,
                                label="End Effector Bone",
                                info="Select which bone to use as the end effector"
                            )
                        with gr.Accordion("Controlled Bones", open=False):
                            gr.Markdown("### Select bones to solve for:")
                            bone_checkboxes = []
                            for bone_name in self.selectable_bones:
                                is_default = bone_name in self.current_controlled_bones
                                checkbox = gr.Checkbox(label=bone_name, value=is_default)
                                bone_checkboxes.append(checkbox)

                        config_status = gr.Textbox(value="Current configuration loaded", label="Configuration Status", interactive=False)

        # Define inputs and outputs for virtual agent
        solve_inputs = [target_x, target_y, target_z, subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, hand_shape, hand_position, show_skeleton]
        click_inputs = [subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, hand_shape, hand_position, show_skeleton]
        outputs = [viz_image, status_text]

        # Configuration update inputs
        config_inputs = [end_effector_dropdown] + bone_checkboxes

        # Handle configuration changes
        for config_input in config_inputs:
            config_input.change(
                fn=self.update_virtual_agent_configuration,
                inputs=config_inputs,
                outputs=[config_status],
                show_progress="hidden"
            )

        # Handle image clicks for virtual agent
        viz_image.select(
            fn=self.on_image_click,
            inputs=[target_z] + click_inputs,
            outputs=[viz_image, status_text, target_x, target_y],
            show_progress="hidden",
        )

        # Regular inputs trigger the standard solve function
        auto_solve_inputs = [target_z, subpoints, distance_weight, collision_weight, distance_enabled, collision_enabled, bone_zero_enabled, bone_zero_weight, derivative_enabled, derivative_weight, hand_shape, hand_position, show_skeleton]
        for inp in auto_solve_inputs:
            inp.change(fn=self.solve_ik, inputs=solve_inputs, outputs=outputs, show_progress="hidden")

        # Store components for later use
        return viz_image, status_text, solve_inputs, outputs

    def create_urdf_robot_tab(self):
        """Create the URDF robot demo tab."""
        with gr.Row():
            with gr.Column(scale=2):
                urdf_viz_image = gr.Image(value=self.create_visualization_image(is_urdf=True), label="URDF Robot - Click to set target position", interactive=False)
                urdf_status_text = gr.Textbox(value="URDF Robot Ready", label="Status", interactive=False)

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🎯 Target & IK"):
                        with gr.Group():
                            gr.Markdown("### Target Position")
                            urdf_target_x = gr.Number(value=0.3, label="Target X", step=0.01, precision=3, interactive=True)
                            urdf_target_y = gr.Number(value=0.3, label="Target Y", step=0.01, precision=3, interactive=True)
                            urdf_target_z = gr.Number(value=0.35, label="Target Z", step=0.01, precision=3)
                        with gr.Group():
                            gr.Markdown("### IK Parameters")
                            gr.Markdown("Please note that Gif generation is very slow.")
                            urdf_subpoints = gr.Slider(
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=self.args.subpoints,
                                label="Trajectory Subpoints \n(>1 creates animated GIF) \n Please note that Gif generation is very slow.",
                            )
                            urdf_show_skeleton = gr.Checkbox(label="Show Skeleton", value=False)
                    with gr.TabItem("🔧 Objectives"):
                        with gr.Group():
                            gr.Markdown("### Primary Objective Functions")
                            urdf_distance_enabled = gr.Checkbox(label="Distance Objective", value=True)
                            gr.Markdown("*Drives the end-effector toward the target position*")
                            urdf_distance_weight = gr.Number(value=1.0, label="Distance Weight", step=0.1)

                            urdf_collision_enabled = gr.Checkbox(label="Collision Avoidance", value=False)
                            gr.Markdown("*Prevents bone segments from intersecting with collision spheres*")
                            urdf_collision_weight = gr.Number(value=1.0, label="Collision Weight", step=0.1)

                        with gr.Group():
                            gr.Markdown("### Regularization Objective Functions")
                            urdf_bone_zero_enabled = gr.Checkbox(label="Bone Zero Rotation", value=True)
                            gr.Markdown("*Keeps joint angles close to zero for natural poses*")
                            urdf_bone_zero_weight = gr.Number(value=0.05, label="Bone Zero Weight", step=0.01)

                            urdf_derivative_enabled = gr.Checkbox(label="Trajectory Smoothing", value=True)
                            gr.Markdown("*Smooths velocity, acceleration, and jerk for natural motion (only for trajectories)*")
                            urdf_derivative_weight = gr.Number(value=0.05, label="Derivative Weight", step=0.01)
                    with gr.TabItem("⚙️ Configuration"):
                        with gr.Group():
                            gr.Markdown("### End Effector")
                            urdf_end_effector_dropdown = gr.Dropdown(
                                choices=self.urdf_available_bones,
                                value=self.urdf_current_end_effector,
                                label="End Effector Bone",
                                info="Select which bone to use as the end effector"
                            )
                        with gr.Accordion("Controlled Bones", open=False):
                            gr.Markdown("### Select bones to solve for:")
                            urdf_bone_checkboxes = []
                            for bone_name in self.urdf_selectable_bones:
                                is_default = bone_name in self.urdf_current_controlled_bones
                                checkbox = gr.Checkbox(label=bone_name, value=is_default)
                                urdf_bone_checkboxes.append(checkbox)

                        urdf_config_status = gr.Textbox(value="Current configuration loaded", label="Configuration Status", interactive=False)

        # Define inputs and outputs for URDF robot
        urdf_solve_inputs = [urdf_target_x, urdf_target_y, urdf_target_z, urdf_subpoints, urdf_distance_weight, urdf_collision_weight, urdf_distance_enabled, urdf_collision_enabled, urdf_bone_zero_enabled, urdf_bone_zero_weight, urdf_derivative_enabled, urdf_derivative_weight, urdf_show_skeleton]
        urdf_click_inputs = [urdf_subpoints, urdf_distance_weight, urdf_collision_weight, urdf_distance_enabled, urdf_collision_enabled, urdf_bone_zero_enabled, urdf_bone_zero_weight, urdf_derivative_enabled, urdf_derivative_weight, urdf_show_skeleton]
        urdf_outputs = [urdf_viz_image, urdf_status_text]

        # Configuration update inputs for URDF
        urdf_config_inputs = [urdf_end_effector_dropdown] + urdf_bone_checkboxes

        # Handle configuration changes for URDF
        for config_input in urdf_config_inputs:
            config_input.change(
                fn=self.update_urdf_robot_configuration,
                inputs=urdf_config_inputs,
                outputs=[urdf_config_status],
                show_progress="hidden"
            )

        # Handle image clicks for URDF robot
        urdf_viz_image.select(
            fn=self.on_urdf_image_click,
            inputs=[urdf_target_z] + urdf_click_inputs,
            outputs=[urdf_viz_image, urdf_status_text, urdf_target_x, urdf_target_y],
            show_progress="hidden",
        )

        # Regular inputs trigger the URDF solve function
        urdf_auto_solve_inputs = [urdf_target_z, urdf_subpoints, urdf_distance_weight, urdf_collision_weight, urdf_distance_enabled, urdf_collision_enabled, urdf_bone_zero_enabled, urdf_bone_zero_weight, urdf_derivative_enabled, urdf_derivative_weight, urdf_show_skeleton]
        for inp in urdf_auto_solve_inputs:
            inp.change(fn=self.solve_urdf_ik, inputs=urdf_solve_inputs, outputs=urdf_outputs, show_progress="hidden")

        # Store components for later use
        return urdf_viz_image, urdf_status_text, urdf_solve_inputs, urdf_outputs

    def create_interface(self):
        """Create the Gradio interface with both virtual agent and URDF robot tabs."""
        with gr.Blocks(title="Interactive Inverse Kinematics Solver", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Interactive Inverse Kinematics Solver")
            gr.Markdown("Choose between Virtual Agent (GLTF) and URDF Robot demos. Adjust parameters and **click on the visualization** to set target positions.")
            gr.Markdown(
                "Please note that enabling/disabling an objective function, changing the trajectory points, or setting different controlled bounds forces a retrace the first time, which can take 5-10 seconds."
            )
            with gr.Tabs():
                # Virtual Agent Tab
                with gr.TabItem("🤖 Virtual Agent"):
                    va_viz_image, va_status_text, va_solve_inputs, va_outputs = self.create_virtual_agent_tab()

                # URDF Robot Tab
                with gr.TabItem("🦾 URDF Robot"):
                    urdf_viz_image, urdf_status_text, urdf_solve_inputs, urdf_outputs = self.create_urdf_robot_tab()

            # Set up initial load for virtual agent tab
            interface.load(fn=self.solve_ik, inputs=va_solve_inputs, outputs=va_outputs)

        return interface

def main():
    parser = configargparse.ArgumentParser(
        description="Interactive Inverse Kinematics Solver - Gradio Version",
        default_config_files=["config.ini"],
    )
    parser.add("--gltf_file", type=str, default="smplx.glb")
    parser.add("--hand", type=str, choices=["left", "right"], default="left")
    parser.add("--threshold", type=float, default=0.001)
    parser.add("--num_steps", type=int, default=50)
    parser.add("--learning_rate", type=float, default=0.2)
    parser.add("--subpoints", type=int, default=1)
    parser.add("--port", type=int, default=7860)
    parser.add("--share", action="store_true", help="Create shareable link")
    args = parser.parse_args()

    jax.config.update("jax_default_device", "cpu")

    app = IKGradioApp(args)
    interface = app.create_interface()
    interface.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
