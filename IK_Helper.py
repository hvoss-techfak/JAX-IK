
import numpy as np
from pygltflib import GLTF2
import tensorflow as tf

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
        # print(f"Node {node_index}: {bone_name} - Local Transform:\n{local_transform}")
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
        mesh_trimesh = mesh_trimesh.simplify_quadratic_decimation(target_face_count)

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
        "vertices": tf.constant(np.hstack([vertices_transformed, np.ones((N, 1))]), dtype=tf.float32),
        "vertex_assignment": tf.constant(vertex_assignment, dtype=tf.int32),
        "faces": mesh_trimesh.faces,
    }
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
        "vertices": tf.constant(np.hstack([vertices, np.ones((N, 1))]), dtype=tf.float32),
        "skin_joints": tf.constant(remapped_skin_joints, dtype=tf.int32),
        "skin_weights": tf.constant(skin_weights_normalized, dtype=tf.float32),
        "faces": faces,
    }
    return mesh_data


def deform_mesh(angle_vector, fk_solver, mesh_data):
    """
    Deforms the mesh vertices using Linear Blend Skinning (LBS) if skinning
    data is available, otherwise falls back to rigid assignment.
    """
    if "skin_joints" in mesh_data and "skin_weights" in mesh_data:
        # --- LBS Path ---
        current_fk = fk_solver.compute_fk_from_angles(angle_vector)
        bind_fk_inv = tf.linalg.inv(fk_solver.bind_fk)
        bone_transforms = tf.matmul(current_fk, bind_fk_inv)

        vertices = mesh_data["vertices"]
        skin_joints = mesh_data["skin_joints"]
        skin_weights = mesh_data["skin_weights"]

        vertex_bone_transforms = tf.gather(bone_transforms, skin_joints)
        weighted_transforms = vertex_bone_transforms * skin_weights[..., None, None]
        final_transforms = tf.reduce_sum(weighted_transforms, axis=1)

        vertices_exp = tf.expand_dims(vertices, axis=-1)
        deformed_vertices_hom = tf.matmul(final_transforms, vertices_exp)
        deformed_vertices = tf.squeeze(deformed_vertices_hom, axis=-1)
        return deformed_vertices[:, :3]
    elif "vertex_assignment" in mesh_data:
        # --- Rigid Assignment Path ---
        current_fk = fk_solver.compute_fk_from_angles(angle_vector)
        bind_fk_inv = tf.linalg.inv(fk_solver.bind_fk)
        bone_transforms = tf.matmul(current_fk, bind_fk_inv)
        vertices = mesh_data["vertices"]
        vertex_assignment = mesh_data["vertex_assignment"]
        vertex_transforms = tf.gather(bone_transforms, vertex_assignment)
        vertices_exp = tf.expand_dims(vertices, axis=-1)
        deformed_vertices_hom = tf.matmul(vertex_transforms, vertices_exp)
        deformed_vertices = tf.squeeze(deformed_vertices_hom, axis=-1)
        return deformed_vertices[:, :3]
    else:
        raise ValueError("mesh_data does not contain valid skinning or assignment information.")