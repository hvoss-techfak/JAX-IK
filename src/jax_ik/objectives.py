from __future__ import annotations

from abc import ABCMeta

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_ik.helper import inverse_skin_points


def _safe_norm(x: jnp.ndarray, axis=None, eps: float = 1e-6) -> jnp.ndarray:
    """L2 norm whose gradient stays finite at x == 0.

    jnp.linalg.norm(x) has an exact-zero-at-the-origin singularity: its
    gradient is x / ||x||, which is 0/0 (nan) when x is exactly zero. That's
    not a corner case here -- e.g. a trajectory's consecutive-frame
    differences start out exactly zero whenever every frame is initialized
    to the same pose (the common case), so an objective built on
    jnp.linalg.norm can NaN out on the very first optimization step. Adding
    eps^2 inside the sqrt instead of eps after it keeps the same forward
    value at x == 0 (sqrt(eps^2) == eps) while keeping the gradient bounded
    everywhere.
    """
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps * eps)


def _safe_arccos(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """arccos whose gradient stays finite at the domain boundary.

    jnp.arccos'(x) = -1/sqrt(1-x^2), which is infinite at x == +-1 -- and two
    unit vectors landing exactly parallel/antiparallel (cos similarity == 1
    or -1) is not a rare corner case for a "look at" style objective: it's
    exactly what happens whenever a bone's rest-pose direction already
    matches the target direction, e.g. at an all-zero initial pose. Clipping
    strictly inside (-1, 1) instead of to the closed interval keeps the
    gradient bounded everywhere.
    """
    return jnp.arccos(jnp.clip(x, -1.0 + eps, 1.0 - eps))


def get_config(X: jnp.ndarray) -> jnp.ndarray:
    """
    Get the last configuration from a trajectory or return the configuration itself if 1D.

    Args:
        X (jnp.ndarray): Joint angle array, shape (T, D) or (D,).

    Returns:
        jnp.ndarray: The last configuration (D,).
    """
    if X.ndim == 1:  # fixed: was X.shape.ndims (invalid)
        return X
    else:
        return X[-1]


class ObjectiveFunction(metaclass=ABCMeta):
    # Opt-in marker: set to True on a subclass whose __call__ *always*
    # reduces X down to just the trajectory's last frame (via
    # `X if X.ndim == 1 else X[-1]`), regardless of any of its instance
    # parameters. The solver loop (_solve_ik_core in ik.py) uses this to
    # route such objectives through a shared last-frame FK computation
    # instead of each recomputing FK independently -- see
    # `_LastFrameFKCache` in ik.py for why that requires this explicit
    # opt-in rather than being automatic. Leave False (the default) unless
    # an objective's frame selection is unconditionally "last frame only";
    # e.g. DistanceObjTraj sometimes reduces to the last frame (a single
    # target point) but not always (multiple target points sample other
    # frames too), so it does not opt in.
    LAST_FRAME_ONLY = False

    def referenced_bones(self):
        """Which bones this objective instance's __call__ actually reads.

        Used by InverseKinematicsSolver to automatically prune FK down to
        just the bones any given set of objectives needs (see
        FKSolver.bones_of_interest / FKSolver._pruned_view): computing FK
        for bones nothing ever queries is pure waste, and for a typical
        skeleton, most bones (the other arm, legs, unrelated fingers, ...)
        are never on the path to anything a given problem's objectives
        care about.

        Returns:
            - () if this objective doesn't use FK at all (pure
              joint-angle-space, e.g. a trajectory smoothness term).
            - A tuple of specific bone names if it only ever reads those
              bones (plus their ancestors, which the pruning logic adds
              automatically) -- the common case.
            - None (the default here) if it needs the *complete,
              unpruned* skeleton, e.g. because it walks every bone via
              fk_solver.bone_names/parent_list, or because it depends on
              fk_solver.sdf/mesh_data/bind_fk, which are only valid for
              the bone indexing of the FKSolver they were computed from.
              None is also the safe default for any custom objective that
              hasn't overridden this: pruning only ever kicks in once
              every active objective has explicitly opted in.
        """
        return None

    def _split_fields(self):
        """Like before but treat ints as auxiliary (static) to avoid tracer in conditionals."""
        leaves, aux = {}, {}
        for k, v in self.__dict__.items():
            if isinstance(v, bool):
                aux[k] = v
                continue
            # ints considered static metadata
            if isinstance(v, int):
                aux[k] = v
                continue
            if isinstance(v, (jnp.ndarray, np.ndarray, jnp.generic, float)):
                leaves[k] = v
            else:
                aux[k] = v
        return leaves, aux

    def tree_flatten(self):
        """
        Flatten the objective function into its pytree representation.
        Returns:
            leaves (tuple): Leaf data (attributes that can change).
            aux (tuple): Auxiliary data (non-leaf attributes).
        """
        leaves, aux = self._split_fields()
        return tuple(leaves.values()), (tuple(leaves.keys()), aux)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        """
        Reconstruct the objective function from its pytree representation.
        Args:
            aux (tuple): Auxiliary data (non-leaf attributes).
            leaves (tuple): Leaf data (attributes that can change).
        Returns:
            ObjectiveFunction: Reconstructed objective function instance.
        """
        leaf_keys, aux_dict = aux
        obj = cls.__new__(cls)  # create without __init__
        for k, v in zip(leaf_keys, leaves):
            setattr(obj, k, v)
        for k, v in aux_dict.items():
            setattr(obj, k, v)
        return obj

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective function as a dictionary.

        Returns:
            dict: Dictionary with the current order, weight, and next_frames.
        """
        leaves, aux = self._split_fields()
        params = {}
        for k, v in {**leaves, **aux}.items():
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                params[k] = np.asarray(v).tolist()
            elif isinstance(v, jnp.generic):
                params[k] = float(v)
            else:
                params[k] = v
        return params

    def update_params(self, params: dict) -> None:
        """
        Update parameters of the objective function.
        This method allows dynamic adjustment of the weight and next_frames.

        Args:
            params_dict (dict): Dictionary of parameters to update.
        """
        for k, v in params.items():
            if hasattr(self, k):
                cur = getattr(self, k)
                if isinstance(cur, (jnp.ndarray, np.ndarray, jnp.generic)):
                    v = jnp.asarray(v, jnp.float32)
                setattr(self, k, v)

    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Evaluate the objective function.

        Args:
            X (jnp.ndarray): Joint angles or trajectory.
            fk_solver: Forward kinematics solver.

        Returns:
            jnp.ndarray: Objective value.
        """
        pass


@register_pytree_node_class
class DistanceObjTraj(ObjectiveFunction):
    """
    Mean-squared distance between a bone end (head/tail) and a sparse set of
    target points along the trajectory.
    """

    def __init__(
        self,
        bone_name: str,
        target_points: np.ndarray,
        use_head: bool = False,
        weight: float = 1.0,
    ):
        """
        Args:
            bone_name (str): Name of the bone to track.
            target_points (np.ndarray): Target points (M,3) or (3,).
            use_head (bool): If True, use bone head; else use tail.
            weight (float): Weight for the objective.
        """
        self.bone_name = bone_name
        self.use_head = bool(use_head)

        self.target_points = jnp.asarray(target_points, jnp.float32)
        if self.target_points.ndim == 1:
            self.target_points = self.target_points[None, :]
        if self.target_points.shape[-1] != 3:
            raise ValueError("target_points must have shape (M,3) or (3,)")

        self.weight = jnp.asarray(weight, jnp.float32)

    def referenced_bones(self):
        return (self.bone_name,)

    def _bone_point(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Get the world-space position of the bone head or tail.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: 3D position.
        """
        fk = fk_solver.compute_fk_from_angles(cfg)
        head, tail = fk_solver.get_bone_head_tail_from_fk(fk, self.bone_name)
        # use python bool (static) so safe
        return head if self.use_head else tail

    # --------------------- main loss ----------------------------------------
    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the MSE between the bone tip (head or tail) and the target points.

        Args:
            X (jnp.ndarray): Joint angles or trajectory.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Weighted mean squared error.
        """
        # Make sure X is 2-D: (T, D)
        X_traj = X.reshape(1, -1) if X.ndim == 1 else X

        T = X_traj.shape[0]
        M = self.target_points.shape[0]
        if T == 0 or M == 0:
            return jnp.asarray(0.0, jnp.float32)

        # Which frame each of the M target points is compared against. This
        # depends only on the static shapes T and M (not on any traced
        # value), so it can be computed with plain numpy at trace time
        # instead of being carried as a jnp array leaf.
        ks = np.arange(1, M + 1, dtype=np.float64)
        idx = np.rint(ks / M * (T - 1)).astype(np.int64)  # (M,)

        # Several target points can map to the same frame (e.g. a single
        # target point always maps to the last frame). Run FK only on the
        # distinct frames that are actually needed instead of vmapping over
        # every frame in the trajectory.
        unique_idx, inverse = np.unique(idx, return_inverse=True)

        if unique_idx.shape[0] == 1:
            # Common case (e.g. a single target point): call FK directly on
            # the one needed frame instead of through vmap. A non-vmapped
            # call has the same HLO shape as the plain (unbatched) FK calls
            # other objectives make on that same frame (e.g.
            # BoneRelativeLookObj), which lets XLA's CSE pass merge them
            # into a single FK evaluation instead of recomputing it once per
            # objective.
            frame_idx = int(unique_idx[0])
            if frame_idx == T - 1:
                # When the one needed frame is the trajectory's last frame,
                # reuse the exact same cfg object _solve_ik_core's FK cache
                # is keyed on (rather than slicing X_traj ourselves, which
                # would produce an equal-valued but distinct object and miss
                # the cache) so this objective's FK forward+backward is
                # fully shared with every LAST_FRAME_ONLY objective too,
                # instead of only sharing the forward value via CSE.
                shared_cfg = getattr(fk_solver, "shared_last_frame_cfg", None)
                cfg = shared_cfg if shared_cfg is not None else X_traj[frame_idx]
            else:
                cfg = X_traj[frame_idx]
            bone_pt = self._bone_point(cfg, fk_solver)
            bone_pts = jnp.broadcast_to(bone_pt[None, :], (M, 3))
        else:
            bone_pts_unique = jax.vmap(lambda cfg: self._bone_point(cfg, fk_solver))(
                X_traj[unique_idx]
            )  # (U, 3)
            bone_pts = bone_pts_unique[inverse]  # (M, 3)

        diff = bone_pts - self.target_points  # (M, 3)
        return jnp.mean(jnp.square(diff)) * self.weight


@register_pytree_node_class
class BoneRelativeLookObj(ObjectiveFunction):
    """
    Penalise the angle between a bone vector and a user-tweaked target point.
    `modifications` is a list of (index, delta) tuples applied to that point.
    """

    LAST_FRAME_ONLY = True

    def __init__(
        self, bone_name: str, use_head: bool, modifications: list, weight: float = 1.0
    ):
        """
        Args:
            bone_name (str): Name of the bone.
            use_head (bool): If True, use head as reference; else tail.
            modifications (list): List of (index, delta) tuples to tweak the target.
            weight (float): Weight for the objective.
        """
        self.bone_name = bone_name
        self.use_head = bool(use_head)

        mods = modifications or []
        self.mod_idx = jnp.asarray([m[0] for m in mods], jnp.int32)
        self.mod_delta = jnp.asarray([m[1] for m in mods], jnp.float32)

        self.weight = jnp.asarray(weight, jnp.float32)

    def referenced_bones(self):
        return (self.bone_name,)

    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the squared angle between the bone vector and the tweaked target vector.

        Args:
            X (jnp.ndarray): Joint angles or trajectory.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Weighted squared angle error.
        """
        # get a single configuration
        cfg = X if X.ndim == 1 else X[-1]

        # FK
        fk = fk_solver.compute_fk_from_angles(cfg)
        head, tail = fk_solver.get_bone_head_tail_from_fk(fk, self.bone_name)

        # target point with user tweaks
        adjusted_target = head if self.use_head else tail
        if self.mod_idx.size > 0:
            adjusted_target = adjusted_target.at[self.mod_idx].add(self.mod_delta)

        # compute the angle between the bone vector and the target vector
        bone_vec = tail - head  # head → tail
        bone_vec = bone_vec / _safe_norm(bone_vec)

        tgt_vec = adjusted_target - head  # head → target
        tgt_vec = tgt_vec / _safe_norm(tgt_vec)

        cos_th = jnp.dot(bone_vec, tgt_vec)
        misalign = _safe_arccos(cos_th) ** 2
        return misalign * self.weight

    def update_params(self, params: dict) -> None:  # custom handling for modifications
        if "modifications" in params:
            mods = params["modifications"] or []
            self.mod_idx = jnp.asarray([m[0] for m in mods], jnp.int32)
            self.mod_delta = jnp.asarray([m[1] for m in mods], jnp.float32)
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)
        if "use_head" in params:
            self.use_head = bool(params["use_head"])
        # ignore unknown keys (no-op)


@register_pytree_node_class
class EndEffectorOrientationObj(ObjectiveFunction):
    """Penalize end-effector orientation error for a URDF link.

    Uses stable quaternion geodesic distance on SO(3):
    angle(R_target^T R_current) = 2*acos(|w|)

    """

    LAST_FRAME_ONLY = True

    def __init__(
        self, bone_name: str, target_transform: np.ndarray, weight: float = 1.0
    ):
        import jax.numpy as jnp

        self.bone_name = bone_name
        self.target_R = jnp.asarray(
            np.asarray(target_transform, dtype=np.float32)[:3, :3], dtype=jnp.float32
        )
        self.weight = jnp.asarray(weight, dtype=jnp.float32)

    def referenced_bones(self):
        return (self.bone_name,)

    def tree_flatten(self):
        return (self.target_R, self.weight), {"bone_name": self.bone_name}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        import jax.numpy as jnp

        target_R, weight = children
        obj = cls.__new__(cls)
        obj.bone_name = aux_data["bone_name"]
        obj.target_R = jnp.asarray(target_R, dtype=jnp.float32)
        obj.weight = jnp.asarray(weight, dtype=jnp.float32)
        return obj

    @staticmethod
    def _quat_w_from_R(R):
        """Return the absolute w component of unit quaternion for rotation matrix R.

        Stable formula: w = sqrt(max(0, (trace(R)+1)/4)).
        We only need w for the SO(3) angle.
        """
        import jax.numpy as jnp

        tr = jnp.trace(R)
        w2 = jnp.maximum(jnp.float32(0.0), (tr + jnp.float32(1.0)) * jnp.float32(0.25))
        w = jnp.sqrt(w2)
        # In case numerical drift makes w slightly > 1, clip.
        return jnp.clip(jnp.abs(w), jnp.float32(0.0), jnp.float32(1.0))

    def __call__(self, X, fk_solver):
        import jax.numpy as jnp

        cfg = X if X.ndim == 1 else X[-1]
        fk = fk_solver.compute_fk_from_angles(cfg)
        idx = fk_solver.bone_names.index(self.bone_name)
        R = fk[idx][:3, :3]

        R_rel = self.target_R.T @ R

        w = self._quat_w_from_R(R_rel)
        angle = jnp.float32(2.0) * _safe_arccos(w)  # radians in [0, pi]
        return jnp.square(angle) * self.weight

@register_pytree_node_class
class DerivativeObj(ObjectiveFunction):
    """Velocity (1), acceleration (2) or jerk (3) regulariser on the trajectory."""

    def __init__(self, order: int, weight: float, next_frames: np.ndarray = None):
        """
        Args:
            order (int): Derivative order (1=velocity, 2=acceleration, 3=jerk).
            weight (float): Weight for the regularization.
            next_frames (np.ndarray): Optional extra frames for continuity.
        """
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        self.order = int(order)
        self.weight = jnp.asarray(weight, jnp.float32)
        if next_frames is None:
            self.next_frames = jnp.zeros((0, 53), dtype=jnp.float32)
        else:
            self.next_frames = jnp.asarray(next_frames, jnp.float32)

    def referenced_bones(self):
        return ()

    # loss -------------------------------------------------------------------
    def __call__(self, X: jnp.ndarray, fk_solver=None) -> jnp.ndarray:
        """
        Compute the mean squared value of the specified derivative order.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: Not used.

        Returns:
            jnp.ndarray: Weighted mean squared derivative.
        """
        if X.ndim == 1:
            return jnp.array(0.0, jnp.float32)

        traj = X
        if self.next_frames.shape[0] > 0:
            traj = jnp.concatenate([X, self.next_frames], axis=0)

        if self.order == 1:
            diff = jnp.diff(traj, n=1, axis=0)
        elif self.order == 2:
            if traj.shape[0] < 3:
                return jnp.array(0.0, jnp.float32)
            diff = traj[2:] - 2 * traj[1:-1] + traj[:-2]
        else:
            if traj.shape[0] < 4:
                return jnp.array(0.0, jnp.float32)
            diff = traj[3:] - 3 * traj[2:-1] + 3 * traj[1:-2] - traj[:-3]

        return jnp.mean(jnp.square(diff)) * self.weight

    def update_params(self, params: dict) -> None:
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)
        # order treated static; ignore updates


@register_pytree_node_class
class CombinedDerivativeObj(ObjectiveFunction):
    """
    Combined velocity, acceleration and jerk regulariser on the trajectory.

    Computes all derivative orders from 1 up to max_order and combines them
    with individual weights or a single weight applied to all.
    """

    def __init__(
        self,
        max_order: int,
        weight: float = 1.0,
        weights: list = None,
        next_frames: np.ndarray = None,
    ):
        """
        Args:
            max_order (int): Maximum derivative order (1, 2, or 3).
            weight (float): Weight for all orders if 'weights' is None.
            weights (list): List of weights for each order.
            next_frames (np.ndarray): Optional extra frames for continuity.
        """
        if max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2 or 3")
        self.max_order = int(max_order)

        # If specific weights for each order are provided, use them
        # Otherwise use the same weight for all orders
        if weights is not None:
            if len(weights) != max_order:
                raise ValueError(
                    f"weights must have length {max_order} for max_order {max_order}"
                )
            self.weights = jnp.asarray(weights, jnp.float32)
        else:
            self.weights = jnp.full(max_order, weight, dtype=jnp.float32)

        if next_frames is None:
            self.next_frames = jnp.zeros((0, 53), dtype=jnp.float32)
        else:
            self.next_frames = jnp.asarray(next_frames, jnp.float32)

    def referenced_bones(self):
        return ()

    # loss -------------------------------------------------------------------
    def __call__(self, X: jnp.ndarray, fk_solver=None) -> jnp.ndarray:
        """
        Compute the combined mean squared values of all derivatives up to max_order.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: Not used.

        Returns:
            jnp.ndarray: Weighted sum of mean squared derivatives.
        """
        if X.ndim == 1:
            return jnp.array(0.0, jnp.float32)

        traj = X
        if self.next_frames.shape[0] > 0:
            traj = jnp.concatenate([X, self.next_frames], axis=0)

        total_loss = jnp.array(0.0, jnp.float32)

        # Compute losses for all orders up to max_order
        for order in range(1, self.max_order + 1):
            if order == 1:
                if traj.shape[0] < 2:
                    continue
                diff = jnp.diff(traj, n=1, axis=0)
            elif order == 2:
                if traj.shape[0] < 3:
                    continue
                diff = traj[2:] - 2 * traj[1:-1] + traj[:-2]
            elif order == 3:
                if traj.shape[0] < 4:
                    continue
                diff = traj[3:] - 3 * traj[2:-1] + 3 * traj[1:-2] - traj[:-3]

            order_loss = jnp.mean(jnp.square(diff)) * self.weights[order - 1]
            total_loss += order_loss

        return total_loss

    def update_params(self, params: dict) -> None:
        if "weights" in params:
            w = params["weights"]
            self.weights = jnp.asarray(w, jnp.float32)
        elif "weight" in params:
            self.weights = jnp.full(self.max_order, params["weight"], dtype=jnp.float32)


@register_pytree_node_class
class InitPoseObj(ObjectiveFunction):
    """
    Anchor the first or last pose (or the whole trajectory) to `init_rot`.
    """

    def __init__(
        self,
        init_rot: np.ndarray,
        full_trajectory: bool = False,
        last_position: bool = False,
        weight: float = 1.0,
        mask: np.ndarray = None,
    ):
        """
        Args:
            init_rot (np.ndarray): Target pose to anchor to.
            full_trajectory (bool): If True, anchor all frames.
            last_position (bool): If True, anchor only the last frame.
            weight (float): Weight for the objective.
            mask (np.ndarray): Optional mask for which angles to anchor.
        """
        self.init_rot = jnp.asarray(init_rot, jnp.float32).reshape(-1)
        self.full_trajectory = bool(full_trajectory)
        self.last_position = bool(last_position)
        self.weight = jnp.asarray(weight, jnp.float32)

        self.mask = (
            jnp.ones_like(self.init_rot)
            if mask is None
            else jnp.asarray(mask, jnp.float32).reshape(-1)
        )

    def referenced_bones(self):
        return ()

    def __call__(self, X: jnp.ndarray, fk_solver=None) -> jnp.ndarray:
        """
        Compute the mean squared error between the selected poses and the target.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: Not used.

        Returns:
            jnp.ndarray: Weighted mean squared error.
        """
        X = jnp.reshape(X, [-1, X.shape[-1]])

        if not self.full_trajectory:
            if self.last_position:
                X = X[-1:]  # Take only the last pose (slice to keep 2D)
            else:
                X = X[:1]  # Take only the first pose (slice to keep 2D)
        # If full_trajectory is True, use all poses (X unchanged)

        # Every selected pose has the same number of (masked) angles, so the
        # mean-of-per-pose-means jax.vmap used to compute is exactly the
        # flat mean over all selected poses' angles -- no need for vmap.
        diff = (X - self.init_rot) * self.mask
        return jnp.mean(jnp.square(diff)) * self.weight


@register_pytree_node_class
class EqualDistanceObj(ObjectiveFunction):
    """
    Keep consecutive poses equally spaced in joint-angle space.
    """

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight (float): Weight for the objective.
        """
        self.weight = jnp.asarray(weight, jnp.float32)

    def referenced_bones(self):
        return ()

    def __call__(self, X: jnp.ndarray, fk_solver=None) -> jnp.ndarray:
        """
        Compute the penalty for unequal spacing between consecutive poses.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: Not used.

        Returns:
            jnp.ndarray: Weighted penalty.
        """
        if X.ndim == 1:
            return jnp.array(0.0, jnp.float32)

        diffs = X[1:] - X[:-1]
        distances = _safe_norm(diffs, axis=1)
        mean_dist = jnp.mean(distances)
        penalty = jnp.mean(jnp.square(distances - mean_dist))
        return penalty * self.weight


@register_pytree_node_class
class SphereCollisionPenaltyObjTraj(ObjectiveFunction):
    """
    Keep every bone segment outside a sphere collider.
    """

    def __init__(self, sphere_collider: dict, weight: float = 1.0, min_clearance: float = 0.05, segment_radius: float = 0.02):
        """
        Args:
            sphere_collider (dict): Dictionary with 'center' and 'radius' keys.
            weight (float): Weight for the penalty.
            min_clearance (float): Minimum allowed clearance from the sphere.
            segment_radius (float): Radius of the bone segment.
        """
        self.center = jnp.asarray(sphere_collider["center"], jnp.float32)
        self.radius = jnp.asarray(sphere_collider["radius"], jnp.float32)
        self.min_clearance = jnp.asarray(min_clearance, jnp.float32)
        self.segment_radius = jnp.asarray(segment_radius, jnp.float32)
        self.weight = jnp.asarray(weight, jnp.float32)

    def referenced_bones(self):
        # Walks every bone (fk_solver.parent_list) to check every segment
        # against the collider, so it needs the complete, unpruned skeleton.
        return None

    def _penalty_single(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the penalty for a single configuration.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Penalty value.
        """
        fk = fk_solver.compute_fk_from_angles(cfg)  # (N,4,4)
        heads = fk[:, :3, 3]  # (N,3)
        parents = jnp.asarray(fk_solver.parent_list, jnp.int32)  # (N,)
        seg_mask = (parents >= 0).astype(jnp.float32)  # (N,)
        safe_parent_indices = jnp.where(parents >= 0, parents, 0)
        p_head = heads[safe_parent_indices]
        c_head = heads
        v = c_head - p_head
        dot_vv = jnp.sum(v * v, axis=1) + 1e-6
        eff_rad = self.radius + self.min_clearance + self.segment_radius
        vc = self.center - p_head
        t = jnp.clip(jnp.sum(vc * v, axis=1) / dot_vv, 0.0, 1.0)
        closest = p_head + t[:, None] * v
        dist = _safe_norm(self.center - closest, axis=1)
        penetration = jnp.maximum(0.0, eff_rad - dist)
        return jnp.sum((penetration ** 2) * seg_mask)

    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the mean penalty over a trajectory.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Weighted mean penalty.
        """
        if X.ndim == 1:
            loss = self._penalty_single(X, fk_solver)
        else:
            loss = jnp.mean(jax.vmap(lambda c: self._penalty_single(c, fk_solver))(X))
        return loss * jnp.float32(self.weight)

    def update_params(self, params: dict) -> None:
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)

@register_pytree_node_class
class BoneDirectionObjective(ObjectiveFunction):
    """
    Penalize deviation of a bone's direction from a desired direction.
    """

    def __init__(
        self,
        bone_name: str,
        use_head: bool = True,
        directions: list = None,
        weight: float = 1.0,
    ):
        """
        Args:
            bone_name (str): Name of the bone.
            use_head (bool): If True, use head-to-tail; else tail-to-head.
            directions (list): List of desired direction vectors.
            weight (float): Weight for the objective.
        """
        self.bone_name = bone_name
        self.use_head = use_head

        if directions is not None:
            self.raw_directions = directions
            self.directions = jnp.asarray(directions, dtype=jnp.float32)
        else:
            self.raw_directions = [[0, 1, 0]]
            self.directions = jnp.array([[0, 1, 0]], dtype=jnp.float32)
        self.weight = jnp.asarray(weight, dtype=jnp.float32)

    def referenced_bones(self):
        return (self.bone_name,)

    def _loss_single(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the normalized squared angle between the bone and desired direction.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Normalized squared angle error.
        """
        fk = fk_solver.compute_fk_from_angles(cfg)
        head, tail = fk_solver.get_bone_head_tail_from_fk(fk, self.bone_name)

        if self.use_head:
            bone_vector = head - tail
        else:
            bone_vector = tail - head

        bone_vector_norm = _safe_norm(bone_vector)
        bone_vector_normalized = bone_vector / bone_vector_norm

        # Combine directions: sum then normalize
        combined_direction = jnp.sum(self.directions, axis=0)
        desired_direction = combined_direction / _safe_norm(combined_direction)

        # Calculate dot product and angle
        dot_product = jnp.sum(bone_vector_normalized * desired_direction)
        angle_difference = _safe_arccos(dot_product)

        # Normalize error by pi^2
        normalized_error = jnp.square(angle_difference) / (jnp.pi**2)
        return normalized_error

    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the mean direction penalty over a trajectory.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Weighted mean penalty.
        """
        # Handle both single config and trajectory
        X = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X[None, :]
        losses = jax.vmap(lambda c: self._loss_single(c, fk_solver))(X)
        return jnp.mean(losses) * self.weight


@register_pytree_node_class
class BoneZeroRotationObj(ObjectiveFunction):
    """
    Shrink every Euler angle toward zero (optionally masked).
    """

    def __init__(self, weight: float = 1.0, mask: np.ndarray = None):
        """
        Args:
            weight (float): Weight for the objective.
            mask (np.ndarray): Optional mask for which angles to penalize.
        """
        self.weight = jnp.asarray(weight, jnp.float32)
        self.mask = (
            jnp.ones([1], jnp.float32)
            if mask is None
            else jnp.asarray(mask, jnp.float32)
        )

    def referenced_bones(self):
        return ()

    def __call__(self, X: jnp.ndarray, fk_solver=None) -> jnp.ndarray:
        """
        Compute the mean squared norm penalty over a trajectory.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: Not used.

        Returns:
            jnp.ndarray: Weighted mean penalty.
        """
        # self.mask broadcasts against the last (angle) dim whether it's a
        # single shared value (shape (1,)) or one per angle (shape (D,)), so
        # no need to branch on its size. Every pose has the same number of
        # (masked) angles, so the mean-of-per-pose-means jax.vmap used to
        # compute is exactly the flat mean over all poses' angles.
        poses = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X[None, :]
        return jnp.mean(jnp.square(poses * self.mask)) * self.weight


@register_pytree_node_class
class SDFCollisionPenaltyObj(ObjectiveFunction):
    """
    Penalize points for being inside a pre-computed SDF grid.
    """

    def __init__(
        self, bone_name: str, sdf: dict, num_samples: int = 10, weight: float = 1.0
    ):
        """
        Args:
            bone_name (str): Name of the bone.
            sdf (dict): SDF dictionary with 'grid', 'origin', 'spacing'.
            num_samples (int): Number of samples along the bone.
            weight (float): Weight for the penalty.
        """
        self.bone_name = bone_name
        self.num_samples = int(num_samples)
        self.weight = jnp.asarray(weight, jnp.float32)
        self.sdf_grid = sdf["grid"]
        self.sdf_origin = sdf["origin"]
        self.sdf_spacing = sdf["spacing"]

    def referenced_bones(self):
        # sdf is an explicit, external grid (not fk_solver.sdf), so this
        # only ever needs self.bone_name's own FK, not the whole skeleton.
        return (self.bone_name,)

    def _get_sdf_value(self, points: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate SDF values at given points.

        Args:
            points (jnp.ndarray): Points to query, shape (N, 3).

        Returns:
            jnp.ndarray: SDF values at the points.
        """
        coords = (points - self.sdf_origin) / self.sdf_spacing
        # Use JAX's map_coordinates for interpolation
        return jax.scipy.ndimage.map_coordinates(
            self.sdf_grid, coords.T, order=1, mode="constant", cval=jnp.inf
        )

    def _penalty_single(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the SDF penetration penalty for a single configuration.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Penalty value.
        """
        fk = fk_solver.compute_fk_from_angles(cfg)
        head, tail = fk_solver.get_bone_head_tail_from_fk(fk, self.bone_name)

        # Sample points along the bone segment
        ts = jnp.linspace(0.0, 1.0, self.num_samples)
        points = jax.vmap(lambda t: head + t * (tail - head))(ts)

        distances = self._get_sdf_value(points)
        penetration = jnp.maximum(0.0, -distances)
        return jnp.mean(jnp.square(penetration))

    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the mean SDF penalty over a trajectory.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Weighted mean penalty.
        """
        X = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X[None, :]
        losses = jax.vmap(lambda c: self._penalty_single(c, fk_solver))(X)
        return jnp.mean(losses) * self.weight

    def update_params(self, params: dict) -> None:
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)


@register_pytree_node_class
class SDFSelfCollisionPenaltyObj(ObjectiveFunction):
    """
    Penalize self-collision using a pre-computed mesh SDF.
    """

    def __init__(
        self,
        bone_names: list,
        num_samples_per_bone: int = 5,
        min_dist: float = 0.0,
        weight: float = 1.0,
    ):
        """
        Args:
            bone_names (list): List of bone names to check for collision.
            num_samples_per_bone (int): Number of samples per bone.
            min_dist (float): Minimum allowed distance from the mesh surface.
            weight (float): Weight for the penalty.
        """
        self.bone_names = tuple(bone_names)
        self.num_samples_per_bone = int(num_samples_per_bone)
        self.min_dist = jnp.float32(min_dist)
        self.weight = jnp.asarray(weight, jnp.float32)

    def referenced_bones(self):
        # Needs fk_solver.sdf/mesh_data/bind_fk, which are only valid (and
        # only computed at all) for the *unpruned* FKSolver they came from
        # -- a pruned view has none of them. Also, inverse_skin_points
        # relies on skin-joint indices that reference bone positions in the
        # full skeleton's own numbering.
        return None

    def _get_sdf_value(self, points: jnp.ndarray, sdf: dict) -> jnp.ndarray:
        """
        Interpolate SDF values at given points.

        Args:
            points (jnp.ndarray): Points to query, shape (N, 3).
            sdf (dict): SDF dictionary.

        Returns:
            jnp.ndarray: SDF values at the points.
        """
        coords = (points - sdf["origin"]) / sdf["spacing"]
        return jax.scipy.ndimage.map_coordinates(
            sdf["grid"], coords.T, order=1, mode="constant", cval=jnp.inf
        )

    def _penalty_single(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the self-collision penalty for a single configuration.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Penalty value.
        """
        if (
            not hasattr(fk_solver, "sdf")
            or fk_solver.sdf is None
            or not hasattr(fk_solver, "mesh_data")
            or fk_solver.mesh_data is None
        ):
            return 0.0

        fk = fk_solver.compute_fk_from_angles(cfg)

        # Collect sample points from all specified bones
        all_points = []
        for bone_name in self.bone_names:
            head, tail = fk_solver.get_bone_head_tail_from_fk(fk, bone_name)
            ts = jnp.linspace(0.0, 1.0, self.num_samples_per_bone)
            points = jax.vmap(lambda t: head + t * (tail - head))(ts)
            all_points.append(points)

        if not all_points:
            return 0.0

        query_points_world = jnp.concatenate(all_points, axis=0)

        # Transform points back to rest-pose local space
        query_points_local = inverse_skin_points(
            query_points_world, fk_solver, fk_solver.mesh_data, fk
        )

        # Query SDF
        distances = self._get_sdf_value(query_points_local, fk_solver.sdf)

        # Penalize if distance is less than min_dist
        # This ignores the surface itself and only penalizes deep penetrations
        penetration = jnp.maximum(0.0, self.min_dist - distances)
        return jnp.mean(jnp.square(penetration))

    def __call__(self, X: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the mean self-collision penalty over a trajectory.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Weighted mean penalty.
        """
        X = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X[None, :]
        losses = jax.vmap(lambda c: self._penalty_single(c, fk_solver))(X)
        return jnp.mean(losses) * self.weight

    def update_params(self, params: dict) -> None:
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)
        if "min_dist" in params:
            self.min_dist = jnp.float32(params["min_dist"])
