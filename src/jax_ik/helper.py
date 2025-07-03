import numpy as np
from pygltflib import GLTF2
import jax.numpy as jnp
import urchin
import trimesh


def quaternion_to_matrix(q):
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return rot


def get_node_transform(node):
    if node.matrix is not None and any(node.matrix):
        return np.array(node.matrix, dtype=np.float32).reshape((4, 4)).T
    else:
        t = np.array(node.translation, dtype=np.float32) if node.translation is not None else np.zeros(3, dtype=np.float32)
        r = np.array(node.rotation, dtype=np.float32) if node.rotation is not None else np.array([0, 0, 0, 1], dtype=np.float32)
        s = np.array(node.scale, dtype=np.float32) if node.scale is not None else np.ones(3, dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = t
        R = quaternion_to_matrix(r)
        S = np.diag(np.append(s, 1.0))
        return T @ R @ S


def load_skeleton_from_gltf(gltf_file):
    gltf = GLTF2().load(gltf_file)

    root_index = None
    for i, node in enumerate(gltf.nodes):
        if node.name == "pelvis":
            root_index = i
            break
    if root_index is None:
        raise ValueError("Skeleton root node 'pelvis' not found in glTF file.")

    bones_by_index = {}

    def build_bone(node_index, parent_name=None):
        node = gltf.nodes[node_index]
        bone_name = node.name if node.name is not None else f"bone_{node_index}"
        local_transform = get_node_transform(node)
        #print(f"Node {node_index}: {bone_name} - Local Transform:\n{local_transform}")
        bone = {
            "name": bone_name,
            "local_transform": local_transform,
            "children": [],
            "bone_length": 0.0,
            "parent": parent_name,
        }
        bones_by_index[node_index] = bone
        if node.children:
            for child_index in node.children:
                child_node = gltf.nodes[child_index]
                child_name = child_node.name if child_node.name is not None else f"bone_{child_index}"
                bone["children"].append(child_name)
                build_bone(child_index, bone_name)

    build_bone(root_index, parent_name=None)

    global_rest = {}

    def compute_global(node_index, parent_transform=np.eye(4, dtype=np.float32)):
        bone = bones_by_index[node_index]
        global_transform = parent_transform @ bone["local_transform"]
        global_rest[node_index] = global_transform
        node = gltf.nodes[node_index]
        if node.children:
            for child_index in node.children:
                compute_global(child_index, global_transform)

    compute_global(root_index)

    for bone_index, bone in bones_by_index.items():
        head = global_rest[bone_index][:3, 3]
        if gltf.nodes[bone_index].children:
            lengths = []
            for child_index in gltf.nodes[bone_index].children:
                child_head = global_rest[child_index][:3, 3]
                lengths.append(np.linalg.norm(child_head - head))
            bone["bone_length"] = np.mean(lengths)
        else:
            bone["bone_length"] = 0.1

    skeleton = {bone["name"]: bone for bone in bones_by_index.values()}
    return skeleton


def load_skeleton_from_urdf(urdf_file):
    """Loads a skeleton and joint limits from a URDF file."""
    robot = urchin.URDF.load(urdf_file)
    skeleton = {}
    limits = {}

    # Create coordinate system transformation matrix:
    # URDF (Z-up, X-forward) to visualization (Y-up, Z-forward)
    # First rotate -90° around X to convert Z-up to Y-up
    # Then rotate 90° around Y to convert X-forward to Z-forward
    # Finally add translation to recenter the robot
    rot_x = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    rot_y = np.array([
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Add translation to move robot down to center it
    translation = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],  # Move down by 1 unit
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    coord_transform = translation @ rot_y @ rot_x

    # First, create all links as potential bones
    link_to_joint = {}  # Maps child link to the joint that creates it

    # Build a mapping from child links to their creating joints
    for joint in robot.joints:
        link_to_joint[joint.child] = joint

    # Create skeleton structure
    for link in robot.links:
        skeleton[link.name] = {
            "name": link.name,
            "local_transform": np.eye(4, dtype=np.float32),
            "children": [],
            "parent": None,
            "bone_length": 0.1,  # Default length
        }

    # Build parent-child relationships and set transforms
    for joint in robot.joints:
        parent_link = joint.parent
        child_link = joint.child

        if parent_link in skeleton and child_link in skeleton:
            # Set parent-child relationship
            skeleton[child_link]["parent"] = parent_link
            skeleton[parent_link]["children"].append(child_link)

            # The local transform of the child is the joint's origin transform
            # This positions the child relative to the parent
            skeleton[child_link]["local_transform"] = joint.origin.astype(np.float32)

            # Store joint limits if available with more restrictive defaults
            if joint.limit is not None and joint.joint_type in ["revolute", "prismatic", "continuous"]:
                if joint.joint_type == "continuous":
                    # Continuous joints get reasonable limits to prevent wild movements
                    limits[child_link] = (-90, 90)
                else:
                    # Convert to degrees and apply more conservative limits for safety
                    lower_deg = max(np.rad2deg(joint.limit.lower), -180)
                    upper_deg = min(np.rad2deg(joint.limit.upper), 180)
                    limits[child_link] = (lower_deg, upper_deg)
            else:
                # Fixed joints or joints without limits get very restrictive bounds
                if joint.joint_type == "fixed":
                    limits[child_link] = (0, 0)  # No movement for fixed joints
                else:
                    limits[child_link] = (-30, 30)  # Conservative default

    # Apply coordinate system transformation to root links
    root_links = [name for name, bone in skeleton.items() if bone["parent"] is None]
    for root_name in root_links:
        skeleton[root_name]["local_transform"] = coord_transform @ skeleton[root_name]["local_transform"]

    # Calculate bone lengths based on the distance to children
    for link_name, bone in skeleton.items():
        if bone["children"]:
            # Calculate distance to each child
            distances = []
            for child_name in bone["children"]:
                child_transform = skeleton[child_name]["local_transform"]
                child_position = child_transform[:3, 3]
                distance = np.linalg.norm(child_position)
                if distance > 0.001:  # Only consider significant distances
                    distances.append(distance)

            if distances:
                bone["bone_length"] = np.mean(distances)
            else:
                bone["bone_length"] = 0.05  # Small default for leaf nodes

    print(f"URDF loaded with {len(skeleton)} links")
    print(f"Root links: {root_links}")
    print(f"Joint limits found for: {list(limits.keys())}")
    print("Applied coordinate transformation: Z-up,X-forward -> Y-up,Z-forward with recentering")

    return skeleton, limits


def load_mesh_data_from_urdf(urdf_file, fk_solver, reduction_factor=0.5):
    """Loads mesh data from a URDF file for rigid skinning."""
    robot = urchin.URDF.load(urdf_file)

    # Create coordinate system transformation matrix:
    # URDF (Z-up, X-forward) to visualization (Y-up, Z-forward)
    # Include the same translation offset as in skeleton loading
    rot_x = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    rot_y = np.array([
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Add translation to move robot down to center it
    translation = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],  # Move down by 1 unit
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    coord_transform = translation @ rot_y @ rot_x

    meshes = []
    mesh_to_link = {}  # Track which link each mesh belongs to
    vertex_to_bone_map = []  # Track which bone each vertex belongs to

    #print("Loading meshes using urchin.Mesh...")

    # Build forward kinematics manually for mesh positioning
    link_transforms = {}

    def compute_link_transform(link_name, visited=None):
        if visited is None:
            visited = set()
        if link_name in visited:
            return np.eye(4)  # Avoid cycles
        visited.add(link_name)

        if link_name in link_transforms:
            return link_transforms[link_name]

        # Find the joint that has this link as child
        parent_joint = None
        for joint in robot.joints:
            if joint.child == link_name:
                parent_joint = joint
                break

        if parent_joint is None:
            # This is a root link - apply coordinate transformation
            transform = coord_transform.astype(np.float32)
        else:
            # Get parent transform and apply joint transform
            parent_transform = compute_link_transform(parent_joint.parent, visited.copy())
            transform = parent_transform @ parent_joint.origin

        link_transforms[link_name] = transform
        return transform

    # Create a mapping from link names to bone indices
    link_to_bone_idx = {name: idx for idx, name in enumerate(fk_solver.bone_names)}

    # Load meshes for each link
    vertex_offset = 0
    for link in robot.links:
        link_transform = compute_link_transform(link.name)

        # Try visual meshes first (changed priority)
        geometries_to_try = []
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry and visual.geometry.mesh:
                    geometries_to_try.append((visual.geometry.mesh, visual.origin))

        # If no visual meshes, fall back to collision meshes
        if not geometries_to_try and link.collisions:
            for collision in link.collisions:
                if collision.geometry and collision.geometry.mesh:
                    geometries_to_try.append((collision.geometry.mesh, collision.origin))

        link_meshes = []
        for urdf_mesh, geom_origin in geometries_to_try:
            try:
                if hasattr(urdf_mesh, 'meshes') and urdf_mesh.meshes:
                    # Already loaded meshes
                    trimesh_meshes = urdf_mesh.meshes
                else:
                    # Load from filename
                    import os
                    urdf_dir = os.path.dirname(urdf_file)
                    mesh_path = urdf_mesh.filename

                    # Handle relative paths
                    if not os.path.isabs(mesh_path):
                        possible_paths = [
                            os.path.join(urdf_dir, mesh_path),
                            os.path.join(urdf_dir, "..", mesh_path),
                            os.path.join(urdf_dir, "meshes", os.path.basename(mesh_path)),
                            mesh_path
                        ]
                    else:
                        possible_paths = [mesh_path]

                    trimesh_meshes = None
                    for full_path in possible_paths:
                        if os.path.exists(full_path):
                            try:
                                loaded_mesh = trimesh.load(full_path)
                                if hasattr(loaded_mesh, 'vertices'):
                                    trimesh_meshes = [loaded_mesh]
                                elif hasattr(loaded_mesh, 'geometry'):
                                    # Scene object
                                    trimesh_meshes = list(loaded_mesh.geometry.values())
                                break
                            except Exception as e:
                                print(f"Failed to load mesh {full_path}: {e}")
                                continue

                if trimesh_meshes:
                    for mesh in trimesh_meshes:
                        if hasattr(mesh, 'vertices') and mesh.vertices.shape[0] > 0:
                            mesh_copy = mesh.copy()

                            # Apply scaling if specified
                            if urdf_mesh.scale is not None:
                                scale_matrix = np.eye(4)
                                scale_matrix[:3, :3] = np.diag(urdf_mesh.scale)
                                mesh_copy.apply_transform(scale_matrix)

                            # Apply transforms: link transform, then geometry origin
                            final_transform = link_transform @ geom_origin
                            mesh_copy.apply_transform(final_transform)

                            link_meshes.append(mesh_copy)
                            #print(f"Loaded visual mesh for {link.name} with {mesh_copy.vertices.shape[0]} vertices")

            except Exception as e:
                print(f"Failed to process mesh for {link.name}: {e}")
                continue

        # Add meshes for this link and track vertex-to-bone mapping
        if link_meshes:
            bone_idx = link_to_bone_idx.get(link.name, -1)
            for mesh in link_meshes:
                meshes.append(mesh)
                mesh_to_link[len(meshes)-1] = link.name
                # Map all vertices of this mesh to this bone
                num_vertices = mesh.vertices.shape[0]
                vertex_to_bone_map.extend([bone_idx] * num_vertices)
        elif link.name in link_to_bone_idx:
            # Create a simple box if no mesh is found but the link exists as a bone
            bone_idx = link_to_bone_idx[link.name]
            transform = compute_link_transform(link.name)
            box = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
            box.apply_transform(transform)
            meshes.append(box)
            mesh_to_link[len(meshes)-1] = link.name
            num_vertices = box.vertices.shape[0]
            vertex_to_bone_map.extend([bone_idx] * num_vertices)

    # Create simple geometry if no meshes found
    if not meshes:
        print("No meshes found. Creating simple geometry...")
        for i, link in enumerate(robot.links):
            if link.name in link_to_bone_idx:
                bone_idx = link_to_bone_idx[link.name]
                transform = compute_link_transform(link.name)
                box = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
                box.apply_transform(transform)
                meshes.append(box)
                mesh_to_link[i] = link.name
                num_vertices = box.vertices.shape[0]
                vertex_to_bone_map.extend([bone_idx] * num_vertices)

        print(f"Created {len(meshes)} simple box meshes")

    if not meshes:
        print("Error: No meshes could be loaded or created.")
        return None

    # Combine all meshes
    #print(f"Combining {len(meshes)} meshes...")
    if len(meshes) == 1:
        combined_mesh = meshes[0]
    else:
        try:
            combined_mesh = trimesh.util.concatenate(meshes)
        except Exception as e:
            print(f"Failed to concatenate meshes: {e}")
            combined_mesh = meshes[0]

    #print(f"Combined mesh has {combined_mesh.vertices.shape[0]} vertices and {combined_mesh.faces.shape[0]} faces")

    # Note: Mesh simplification removed - always use original mesh quality
    vertices = combined_mesh.vertices

    # Convert vertex-to-bone mapping to numpy array
    vertex_to_bone_array = np.array(vertex_to_bone_map, dtype=np.int32)

    # Handle any unmapped vertices (bone_idx = -1)
    unmapped_mask = vertex_to_bone_array == -1
    if np.any(unmapped_mask):
        print(f"Warning: {np.sum(unmapped_mask)} vertices could not be mapped to bones. Assigning to bone 0.")
        vertex_to_bone_array[unmapped_mask] = 0

    #print(f"Vertex-to-bone mapping: {len(vertex_to_bone_array)} vertices mapped to bones")
    #print(f"Bone index range: {vertex_to_bone_array.min()} to {vertex_to_bone_array.max()}")

    N = vertices.shape[0]
    mesh_data = {
        "vertices": jnp.array(np.hstack([vertices, np.ones((N, 1))]), dtype=jnp.float32),
        "vertex_assignment": jnp.array(vertex_to_bone_array, dtype=jnp.int32),
        "faces": combined_mesh.faces,
        "is_urdf": True,  # Flag to indicate this is URDF mesh data
    }

    #print(f"Successfully created mesh data with {N} vertices")
    return mesh_data


def load_mesh_data_from_gltf(gltf_file, fk_solver, reduction_factor=0.5):
    """
    Loads mesh data from a GLTF or GLB file for skinning.
    This version uses pygltflib to handle binary GLTF files and extract skinning data.
    """
    gltf = GLTF2().load(gltf_file)

    # Helper to get buffer bytes
    def get_buffer_bytes(buffer_view):
        buffer = gltf.buffers[buffer_view.buffer]
        offset = buffer_view.byteOffset or 0
        length = buffer_view.byteLength
        # Get the raw buffer data
        if buffer.uri is not None:
            raw = gltf.get_data_from_buffer_uri(buffer.uri)
        else:
            raw = gltf.binary_blob()
        return raw[offset : offset + length]

    # Find the mesh associated with the skin
    skin = gltf.skins[0] if gltf.skins else None
    if skin is None:
        print("Warning: No skin found in GLTF file. Falling back to rigid vertex assignment.")
        return _load_mesh_data_rigid(gltf_file, fk_solver, reduction_factor)

    skinned_node_idx = -1
    for i, node in enumerate(gltf.nodes):
        if node.skin == 0:  # Assuming first skin
            skinned_node_idx = i
            break

    if skinned_node_idx == -1:
        print("Warning: A skin was found, but no node uses it. Falling back.")
        return _load_mesh_data_rigid(gltf_file, fk_solver, reduction_factor)

    mesh_idx = gltf.nodes[skinned_node_idx].mesh
    if mesh_idx is None:
        print("Warning: Skinned node has no mesh. Falling back.")
        return _load_mesh_data_rigid(gltf_file, fk_solver, reduction_factor)

    primitive = gltf.meshes[mesh_idx].primitives[0]
    attributes = primitive.attributes  # Access attributes directly

    # --- FIX: Use hasattr/getattr instead of "in"/[] for Attributes object ---
    if not (hasattr(attributes, "JOINTS_0") and hasattr(attributes, "WEIGHTS_0")):
        print("Warning: Skinned mesh primitive lacks JOINTS_0 or WEIGHTS_0. Falling back.")
        return _load_mesh_data_rigid(gltf_file, fk_solver, reduction_factor)

    # Decode vertex attributes
    joints_accessor_idx = getattr(attributes, "JOINTS_0")
    weights_accessor_idx = getattr(attributes, "WEIGHTS_0")
    position_accessor_idx = getattr(attributes, "POSITION")

    # --- JOINTS_0 ---
    joints_accessor = gltf.accessors[joints_accessor_idx]
    joints_buffer_view = gltf.bufferViews[joints_accessor.bufferView]
    joints_bytes = get_buffer_bytes(joints_buffer_view)
    if joints_accessor.componentType == 5121:  # UNSIGNED_BYTE
        joints = np.frombuffer(joints_bytes, dtype=np.uint8)
    elif joints_accessor.componentType == 5123:  # UNSIGNED_SHORT
        joints = np.frombuffer(joints_bytes, dtype=np.uint16)
    else:
        raise ValueError("Unsupported JOINTS_0 componentType")
    joints = joints.reshape(-1, joints_accessor.type.count("VEC") and 4 or 1).copy()  # <-- .copy() for writeable

    # --- WEIGHTS_0 ---
    weights_accessor = gltf.accessors[weights_accessor_idx]
    weights_buffer_view = gltf.bufferViews[weights_accessor.bufferView]
    weights_bytes = get_buffer_bytes(weights_buffer_view)
    if weights_accessor.componentType == 5126:  # FLOAT
        weights = np.frombuffer(weights_bytes, dtype=np.float32)
    elif weights_accessor.componentType == 5121:  # UNSIGNED_BYTE
        weights = np.frombuffer(weights_bytes, dtype=np.uint8) / 255.0
    elif weights_accessor.componentType == 5123:  # UNSIGNED_SHORT
        weights = np.frombuffer(weights_bytes, dtype=np.uint16) / 65535.0
    else:
        raise ValueError("Unsupported WEIGHTS_0 componentType")
    weights = weights.reshape(-1, weights_accessor.type.count("VEC") and 4 or 1).copy()  # <-- .copy() for writeable

    # --- POSITION ---
    vertices_accessor = gltf.accessors[position_accessor_idx]
    vertices_buffer_view = gltf.bufferViews[vertices_accessor.bufferView]
    vertices_bytes = get_buffer_bytes(vertices_buffer_view)
    vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)

    # --- FACES ---
    faces_accessor = gltf.accessors[primitive.indices]
    faces_buffer_view = gltf.bufferViews[faces_accessor.bufferView]
    faces_bytes = get_buffer_bytes(faces_buffer_view)
    if faces_accessor.componentType == 5123:  # UNSIGNED_SHORT
        faces = np.frombuffer(faces_bytes, dtype=np.uint16).reshape(-1, 3)
    elif faces_accessor.componentType == 5125:  # UNSIGNED_INT
        faces = np.frombuffer(faces_bytes, dtype=np.uint32).reshape(-1, 3)
    else:
        raise ValueError("Unsupported indices componentType")

    # Map GLTF joint indices to FK solver bone indices
    gltf_joint_names = [gltf.nodes[i].name for i in skin.joints]
    solver_bone_to_idx = {name: i for i, name in enumerate(fk_solver.bone_names)}
    gltf_to_solver_map = np.array([solver_bone_to_idx.get(name, -1) for name in gltf_joint_names], dtype=np.int32)
    remapped_skin_joints = gltf_to_solver_map[joints].copy()  # <-- .copy() for writeable

    unmapped_mask = remapped_skin_joints == -1
    if np.any(unmapped_mask):
        print("Warning: Some skin joints could not be mapped to FK solver bones. Their weights will be zeroed.")
        weights[unmapped_mask] = 0.0
        remapped_skin_joints[unmapped_mask] = 0

    # Normalize weights
    weight_sum = np.sum(weights, axis=1, keepdims=True)
    weight_sum[weight_sum == 0] = 1.0
    skin_weights_normalized = weights / weight_sum

    N = vertices.shape[0]
    mesh_data = {
        "vertices": jnp.array(np.hstack([vertices, np.ones((N, 1))]), dtype=jnp.float32),
        "skin_joints": jnp.array(remapped_skin_joints, dtype=jnp.int32),
        "skin_weights": jnp.array(skin_weights_normalized, dtype=jnp.float32),
        "faces": faces,
    }
    return mesh_data


def deform_mesh(angle_vector, fk_solver, mesh_data):
    """
    Deforms the mesh vertices using Linear Blend Skinning (LBS) if skinning
    data is available, otherwise falls back to rigid assignment.
    For URDF meshes, uses per-bone rigid assignment.
    """
    if "skin_joints" in mesh_data and "skin_weights" in mesh_data:
        #print("Using Linear Blend Skinning (LBS) for mesh deformation.")
        current_fk = fk_solver.compute_fk_from_angles(angle_vector)
        bind_fk_inv = jnp.linalg.inv(fk_solver.bind_fk)
        bone_transforms = jnp.matmul(current_fk, bind_fk_inv)

        vertices = mesh_data["vertices"]
        skin_joints = mesh_data["skin_joints"]
        skin_weights = mesh_data["skin_weights"]

        vertex_bone_transforms = bone_transforms[skin_joints]
        weighted_transforms = vertex_bone_transforms * skin_weights[..., None, None]
        final_transforms = jnp.sum(weighted_transforms, axis=1)

        vertices_exp = jnp.expand_dims(vertices, axis=-1)
        deformed_vertices_hom = jnp.matmul(final_transforms, vertices_exp)
        deformed_vertices = jnp.squeeze(deformed_vertices_hom, axis=-1)
        return deformed_vertices[:, :3]
    elif "vertex_assignment" in mesh_data:
        if not mesh_data.get("is_urdf", False):
            print("Warning: No skinning data found, using rigid vertex assignment.")

        current_fk = fk_solver.compute_fk_from_angles(angle_vector)
        bind_fk_inv = jnp.linalg.inv(fk_solver.bind_fk)
        bone_transforms = jnp.matmul(current_fk, bind_fk_inv)
        vertices = mesh_data["vertices"]
        vertex_assignment = mesh_data["vertex_assignment"]
        vertex_transforms = bone_transforms[vertex_assignment]
        vertices_exp = jnp.expand_dims(vertices, axis=-1)
        deformed_vertices_hom = jnp.matmul(vertex_transforms, vertices_exp)
        deformed_vertices = jnp.squeeze(deformed_vertices_hom, axis=-1)
        return deformed_vertices[:, :3]
    else:
        raise ValueError("mesh_data does not contain valid skinning or assignment information.")


def _load_mesh_data_rigid(gltf_file, fk_solver, reduction_factor, scene=None):
    """Helper for rigid vertex assignment based on bone proximity."""
    import trimesh

    if scene is None:
        scene = trimesh.load(gltf_file, force="scene")

    mesh_key = list(scene.geometry.keys())[0]
    mesh_trimesh = scene.geometry[mesh_key]

    mesh_transform = np.eye(4)
    # Correctly get transform from scene graph using the geometry key
    if mesh_key in scene.graph.geometry_nodes:
        node_name = scene.graph.geometry_nodes[mesh_key][0]
        mesh_transform = scene.graph.get(node_name)
    else:
        print("Warning: No scene graph node found for the mesh. Using identity transform.")

    target_face_count = int(mesh_trimesh.faces.shape[0] * reduction_factor)
    if target_face_count > 0 and target_face_count < mesh_trimesh.faces.shape[0]:
        mesh_trimesh = mesh_trimesh.simplify_quadric_decimation(target_face_count)

    vertices = mesh_trimesh.vertices
    ones = np.ones((vertices.shape[0], 1))
    vertices_hom = np.hstack([vertices, ones])
    vertices_transformed = (mesh_transform @ vertices_hom.T).T[:, :3]

    bone_positions = []
    for i in range(len(fk_solver.bone_names)):
        bone_positions.append(np.asarray(fk_solver.bind_fk[i][:3, 3]))
    bone_positions = np.array(bone_positions)

    dists = np.linalg.norm(vertices_transformed[:, None, :] - bone_positions[None, :, :], axis=2)
    vertex_assignment = np.argmin(dists, axis=1)

    N = vertices_transformed.shape[0]

    mesh_data = {
        "vertices": jnp.array(np.hstack([vertices_transformed, np.ones((N, 1))]), dtype=jnp.float32),
        "vertex_assignment": jnp.array(vertex_assignment, dtype=jnp.int32),
        "faces": mesh_trimesh.faces,
    }
    return mesh_data
