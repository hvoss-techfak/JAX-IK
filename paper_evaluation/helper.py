import numpy as np

# We add trimesh for loading and processing the mesh.
import trimesh
from pygltflib import GLTF2


def load_mesh_colliders(gltf_file):
    """
    Loads the mesh from the given glTF file using trimesh,
    performs a convex decomposition, and returns a tuple of collider
    equations (each a numpy array of shape (n_planes, 4), where each row
    is [a, b, c, d] such that for any point x inside the collider: a*x + b*x + c*x + d <= 0).

    Args:
        gltf_file (str): Path to the glTF file.

    Returns:
        tuple: A tuple of numpy arrays containing the collider equations.
    """

    # Load the entire file with trimesh (it can load glTF files)
    mesh = trimesh.load(gltf_file, force="mesh")
    # Perform convex decomposition. This returns a list of convex meshes.
    # (Make sure you have a VHACD backend installed if needed.)
    colliders = mesh.convex_decomposition()
    collider_eqs = []
    for collider in colliders:
        # The 'equations' property is available for convex meshes.
        # They are in the form [a, b, c, d] for each plane.
        collider_eqs.append(collider.equations.astype(np.float32))
    # Return as a tuple so it can be used as a static argument.
    return tuple(collider_eqs)


# ---------------------------------------------------------------------
# Your existing helper functions for converting quaternions and transforms
# ---------------------------------------------------------------------
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
        t = (
            np.array(node.translation, dtype=np.float32)
            if node.translation is not None
            else np.zeros(3, dtype=np.float32)
        )
        r = (
            np.array(node.rotation, dtype=np.float32)
            if node.rotation is not None
            else np.array([0, 0, 0, 1], dtype=np.float32)
        )
        s = (
            np.array(node.scale, dtype=np.float32)
            if node.scale is not None
            else np.ones(3, dtype=np.float32)
        )
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
                child_name = (
                    child_node.name
                    if child_node.name is not None
                    else f"bone_{child_index}"
                )
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
