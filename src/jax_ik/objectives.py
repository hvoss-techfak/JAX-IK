from __future__ import annotations

from abc import ABCMeta, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from jax_ik.helper import inverse_skin_points


def get_config(X: jnp.ndarray) -> jnp.ndarray:
    """
    Get the last configuration from a trajectory or return the configuration itself if 1D.

    Args:
        X (jnp.ndarray): Joint angle array, shape (T, D) or (D,).

    Returns:
        jnp.ndarray: The last configuration (D,).
    """
    if X.shape.ndims == 1:
        return X
    else:
        return X[-1]


class ObjectiveFunction(metaclass=ABCMeta):
    def update_params(self, params_dict: dict) -> None:
        """
        Update parameters of the objective function.
        This method allows dynamic adjustment of the weight and next_frames.

        Args:
            params_dict (dict): Dictionary of parameters to update.
        """
        pass

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective function as a dictionary.

        Returns:
            dict: Dictionary with the current order, weight, and next_frames.
        """
        pass

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
        self._update_ratios()

    def tree_flatten(self):
        leaves = (self.target_points, self.weight, self.rat)
        aux = (self.bone_name, self.use_head)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        bone_name, use_head = aux
        target_points, weight, rat = leaves
        obj = cls(bone_name, target_points, use_head, weight)
        obj.rat = rat
        return obj

    def update_params(self, params: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            params (dict): Dictionary with keys 'bone_name', 'use_head', 'target_points', 'weight'.
        """
        if "bone_name" in params:
            self.bone_name = params["bone_name"]
        if "use_head" in params:
            self.use_head = bool(params["use_head"])
        if "target_points" in params:
            pts = jnp.asarray(params["target_points"], jnp.float32)
            self.target_points = pts[None, :] if pts.ndim == 1 else pts
            self._update_ratios()
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(
            bone_name=self.bone_name,
            use_head=self.use_head,
            target_points=np.asarray(self.target_points).tolist(),
            weight=float(self.weight),
        )

    def _update_ratios(self) -> None:
        """
        Pre-compute the fractions k/M that decide which frames to sample.

        Handles the corner-case M == 0 to avoid divide-by-zero.
        """
        M = int(self.target_points.shape[0])
        if M == 0:
            self.rat = jnp.zeros((0,), jnp.float32)
            return

        ks = jnp.arange(1, M + 1, dtype=jnp.float32)  # 1 … M
        self.rat = ks / jnp.float32(M)  # (M,)

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

        # FK for every frame
        bone_pts = jax.vmap(lambda cfg: self._bone_point(cfg, fk_solver))(X_traj)  # (T, 3)

        T = bone_pts.shape[0]
        if T == 0 or self.rat.size == 0:
            return jnp.asarray(0.0, jnp.float32)

        # For non-negative values this is identical to floor(x + 0.5).
        idx = jnp.rint(self.rat * jnp.float32(T - 1)).astype(jnp.int32)

        diff = bone_pts[idx] - self.target_points  # (M, 3)
        return jnp.mean(jnp.square(diff)) * self.weight


@register_pytree_node_class
class BoneRelativeLookObj(ObjectiveFunction):
    """
    Penalise the angle between a bone vector and a user-tweaked target point.
    `modifications` is a list of (index, delta) tuples applied to that point.
    """

    def __init__(
        self,
        bone_name: str,
        use_head: bool,
        modifications: list,
        weight: float = 1.0
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

    # pytree -----------------------------------------------------------------
    def tree_flatten(self):
        leaves = (self.mod_idx, self.mod_delta, self.weight)
        aux = (self.bone_name, self.use_head)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        bone_name, use_head = aux
        mod_idx, mod_delta, w = leaves
        mods = list(zip(np.asarray(mod_idx).tolist(), np.asarray(mod_delta).tolist()))
        return cls(bone_name, use_head, mods, w)

    # API --------------------------------------------------------------------
    def update_params(self, params: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            params (dict): Dictionary with keys 'bone_name', 'use_head', 'modifications', 'weight'.
        """
        if "bone_name" in params:
            self.bone_name = params["bone_name"]
        if "use_head" in params:
            self.use_head = bool(params["use_head"])
        if "modifications" in params:
            mods = params["modifications"] or []
            self.mod_idx = jnp.asarray([m[0] for m in mods], jnp.int32)
            self.mod_delta = jnp.asarray([m[1] for m in mods], jnp.float32)
        if "weight" in params:
            self.weight = jnp.asarray(params["weight"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        mods = list(zip(np.asarray(self.mod_idx).tolist(), np.asarray(self.mod_delta).tolist()))
        return dict(
            bone_name=self.bone_name,
            use_head=self.use_head,
            modifications=mods,
            weight=float(self.weight),
        )

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
        bone_vec = bone_vec / (jnp.linalg.norm(bone_vec) + 1e-6)

        tgt_vec = adjusted_target - head  # head → target
        tgt_vec = tgt_vec / (jnp.linalg.norm(tgt_vec) + 1e-6)

        cos_th = jnp.clip(jnp.dot(bone_vec, tgt_vec), -1.0, 1.0)
        misalign = jnp.arccos(cos_th) ** 2
        return misalign * self.weight


@register_pytree_node_class
class DerivativeObj(ObjectiveFunction):
    """Velocity (1), acceleration (2) or jerk (3) regulariser on the trajectory."""

    def __init__(
        self,
        order: int,
        weight: float,
        next_frames: np.ndarray = None
    ):
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

    # pytree -----------------------------------------------------------------
    def tree_flatten(self):
        return (self.weight, self.next_frames), (self.order,)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (order,) = aux
        w, nxt = leaves
        obj = cls(order, w)
        obj.next_frames = nxt
        return obj

    # API --------------------------------------------------------------------
    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'order', 'weight', 'next_frames'.
        """
        if "order" in p:
            if p["order"] not in (1, 2, 3):
                raise ValueError("order must be 1, 2 or 3")
            self.order = int(p["order"])
        if "weight" in p:
            self.weight = jnp.asarray(p["weight"], jnp.float32)
        if "next_frames" in p:
            if p["next_frames"] is None:
                self.next_frames = jnp.zeros((0, 53), dtype=jnp.float32)
            else:
                self.next_frames = jnp.asarray(p["next_frames"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(
            order=self.order,
            weight=float(self.weight),
            next_frames=np.asarray(self.next_frames) if self.next_frames.shape[0] > 0 else None,
        )

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
        next_frames: np.ndarray = None
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
                raise ValueError(f"weights must have length {max_order} for max_order {max_order}")
            self.weights = jnp.asarray(weights, jnp.float32)
        else:
            self.weights = jnp.full(max_order, weight, dtype=jnp.float32)

        if next_frames is None:
            self.next_frames = jnp.zeros((0, 53), dtype=jnp.float32)
        else:
            self.next_frames = jnp.asarray(next_frames, jnp.float32)

    # pytree -----------------------------------------------------------------
    def tree_flatten(self):
        return (self.weights, self.next_frames), (self.max_order,)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (max_order,) = aux
        weights, nxt = leaves
        obj = cls(max_order, weight=1.0, weights=weights)
        obj.next_frames = nxt
        return obj

    # API --------------------------------------------------------------------
    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'max_order', 'weight', 'weights', 'next_frames'.
        """
        if "max_order" in p:
            if p["max_order"] not in (1, 2, 3):
                raise ValueError("max_order must be 1, 2 or 3")
            old_max_order = self.max_order
            self.max_order = int(p["max_order"])
            # Adjust weights array if max_order changed
            if self.max_order != old_max_order:
                if self.max_order > old_max_order:
                    # Extend weights with the last weight value
                    last_weight = self.weights[-1] if len(self.weights) > 0 else 1.0
                    new_weights = jnp.concatenate([
                        self.weights,
                        jnp.full(self.max_order - old_max_order, last_weight, dtype=jnp.float32)
                    ])
                    self.weights = new_weights
                else:
                    # Truncate weights
                    self.weights = self.weights[:self.max_order]

        if "weight" in p:
            # Set all weights to the same value
            self.weights = jnp.full(self.max_order, p["weight"], dtype=jnp.float32)

        if "weights" in p:
            if len(p["weights"]) != self.max_order:
                raise ValueError(f"weights must have length {self.max_order} for max_order {self.max_order}")
            self.weights = jnp.asarray(p["weights"], jnp.float32)

        if "next_frames" in p:
            if p["next_frames"] is None:
                self.next_frames = jnp.zeros((0, 53), dtype=jnp.float32)
            else:
                self.next_frames = jnp.asarray(p["next_frames"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(
            max_order=self.max_order,
            weights=np.asarray(self.weights).tolist(),
            next_frames=np.asarray(self.next_frames) if self.next_frames.shape[0] > 0 else None,
        )

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

        self.mask = jnp.ones_like(self.init_rot) if mask is None else jnp.asarray(mask, jnp.float32).reshape(-1)

    # pytree -----------------------------------------------------------------
    def tree_flatten(self):
        leaves = (self.init_rot, self.mask, self.weight)
        aux = (self.full_trajectory, self.last_position)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        full_traj, last_pos = aux
        init_rot, mask, w = leaves
        return cls(init_rot, full_traj, last_pos, w, mask)

    # API --------------------------------------------------------------------
    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'init_rot', 'weight', 'mask', 'full_trajectory', 'last_position'.
        """
        if "init_rot" in p:
            self.init_rot = jnp.asarray(p["init_rot"], jnp.float32).reshape(-1)
        if "weight" in p:
            self.weight = jnp.asarray(p["weight"], jnp.float32)
        if "mask" in p:
            self.mask = jnp.asarray(p["mask"], jnp.float32).reshape(-1)
        if "full_trajectory" in p:
            self.full_trajectory = bool(p["full_trajectory"])
        if "last_position" in p:
            self.last_position = bool(p["last_position"])

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(
            init_rot=np.asarray(self.init_rot),
            full_trajectory=self.full_trajectory,
            last_position=self.last_position,
            weight=float(self.weight),
            mask=np.asarray(self.mask),
        )

    def _loss_single(self, pose: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the masked mean squared error between a pose and the target.

        Args:
            pose (jnp.ndarray): Pose to compare.

        Returns:
            jnp.ndarray: Masked mean squared error.
        """
        # Apply mask and compute MSE
        diff = (pose - self.init_rot) * self.mask
        return jnp.mean(jnp.square(diff))

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

        # Compute loss for selected poses
        losses = jax.vmap(self._loss_single)(X)
        return jnp.mean(losses) * self.weight


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

    def tree_flatten(self):
        return (self.weight,), ()

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (w,) = leaves
        return cls(w)

    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with key 'weight'.
        """
        if "weight" in p:
            self.weight = jnp.asarray(p["weight"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(weight=float(self.weight))

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
        distances = jnp.linalg.norm(diffs, axis=1) + 1e-6
        mean_dist = jnp.mean(distances)
        penalty = jnp.mean(jnp.square(distances - mean_dist))
        return penalty * self.weight


@register_pytree_node_class
class SphereCollisionPenaltyObjTraj(ObjectiveFunction):
    """
    Keep every bone segment outside a sphere collider.
    """

    def __init__(
        self,
        sphere_collider: dict,
        weight: float = 1.0,
        min_clearance: float = 0.05,
        segment_radius: float = 0.02
    ):
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

    def tree_flatten(self):
        leaves = (
            self.center,
            self.radius,
            self.min_clearance,
            self.segment_radius,
            self.weight,
        )
        return leaves, ()

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        c, r, mc, sr, w = leaves
        return cls(dict(center=c, radius=r), w, mc, sr)

    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'sphere_collider', 'center', 'radius', 'min_clearance', 'segment_radius', 'weight'.
        """
        if "sphere_collider" in p:
            collider = p["sphere_collider"]
            if "center" in collider:
                self.center = jnp.asarray(collider["center"], jnp.float32)
            if "radius" in collider:
                self.radius = jnp.asarray(collider["radius"], jnp.float32)
        if "center" in p:
            self.center = jnp.asarray(p["center"], jnp.float32)
        if "radius" in p:
            self.radius = jnp.asarray(p["radius"], jnp.float32)
        if "min_clearance" in p:
            self.min_clearance = jnp.asarray(p["min_clearance"], jnp.float32)
        if "segment_radius" in p:
            self.segment_radius = jnp.asarray(p["segment_radius"], jnp.float32)
        if "weight" in p:
            self.weight = jnp.asarray(p["weight"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(
            sphere_collider=dict(center=np.asarray(self.center).tolist(), radius=float(self.radius)),
            min_clearance=float(self.min_clearance),
            segment_radius=float(self.segment_radius),
            weight=float(self.weight),
        )

    def _penalty_single(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the penalty for a single configuration.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Penalty value.
        """
        fk = fk_solver.compute_fk_from_angles(cfg)
        total = 0.0
        eff_rad = self.radius + self.min_clearance + self.segment_radius

        for i, parent in enumerate(fk_solver.parent_list):
            if parent < 0:
                continue
            pb = fk_solver.bone_names[parent]
            cb = fk_solver.bone_names[i]

            p_head, _ = fk_solver.get_bone_head_tail_from_fk(fk, pb)
            c_head, _ = fk_solver.get_bone_head_tail_from_fk(fk, cb)

            v = c_head - p_head
            dot_vv = jnp.dot(v, v) + 1e-6
            t = jnp.clip(jnp.dot(self.center - p_head, v) / dot_vv, 0.0, 1.0)
            closest = p_head + t * v

            dist = jnp.linalg.norm(self.center - closest)
            penetration = jnp.maximum(0.0, eff_rad - dist)
            total += penetration**2
        return total

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
        return loss * self.weight


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
        weight: float = 1.0
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

    def tree_flatten(self):
        return (self.directions, self.weight), (self.bone_name, self.use_head, self.raw_directions)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        bone_name, use_head, raw_directions = aux
        directions, w = leaves
        # Use the stored raw_directions instead of converting JAX arrays
        return cls(bone_name, use_head, raw_directions, w)


    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'bone_name', 'use_head', 'directions', 'weight'.
        """
        if "bone_name" in p:
            self.bone_name = p["bone_name"]
        if "use_head" in p:
            self.use_head = bool(p["use_head"])
        if "directions" in p:
            self.raw_directions = p["directions"]
            self.directions = jnp.asarray(p["directions"], dtype=jnp.float32)
        if "weight" in p:
            self.weight = jnp.asarray(p["weight"], dtype=jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(
            bone_name=self.bone_name,
            use_head=self.use_head,
            directions=self.raw_directions,
            weight=float(self.weight),
        )

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

        bone_vector_norm = jnp.linalg.norm(bone_vector) + 1e-6
        bone_vector_normalized = bone_vector / bone_vector_norm

        # Combine directions: sum then normalize
        combined_direction = jnp.sum(self.directions, axis=0)
        desired_direction = combined_direction / (jnp.linalg.norm(combined_direction) + 1e-6)

        # Calculate dot product and angle
        dot_product = jnp.sum(bone_vector_normalized * desired_direction)
        dot_product_clipped = jnp.clip(dot_product, -1.0, 1.0)
        angle_difference = jnp.arccos(dot_product_clipped)

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
        self.mask = jnp.ones([1], jnp.float32) if mask is None else jnp.asarray(mask, jnp.float32)

    def tree_flatten(self):
        return (self.weight, self.mask), ()

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        w, m = leaves
        return cls(w, m)

    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'weight', 'mask'.
        """
        if "weight" in p:
            self.weight = jnp.asarray(p["weight"], jnp.float32)
        if "mask" in p:
            self.mask = jnp.asarray(p["mask"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return dict(weight=float(self.weight), mask=np.asarray(self.mask))

    def _masked_sq_norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the masked mean squared norm of x.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Masked mean squared norm.
        """
        #if mask is shape (1,) then it is applied to every element of x
        if self.mask.ndim == 1 and self.mask.shape[0] == 1:
            return jnp.mean(jnp.square(x)) * self.mask[0]
        return jnp.mean(jnp.square(x * self.mask))

    def __call__(self, X: jnp.ndarray, fk_solver=None) -> jnp.ndarray:
        """
        Compute the mean squared norm penalty over a trajectory.

        Args:
            X (jnp.ndarray): Trajectory of joint angles.
            fk_solver: Not used.

        Returns:
            jnp.ndarray: Weighted mean penalty.
        """
        poses = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X[None, :]
        return jnp.mean(jax.vmap(self._masked_sq_norm)(poses)) * self.weight

@register_pytree_node_class
class SDFCollisionPenaltyObj(ObjectiveFunction):
    """
    Penalize points for being inside a pre-computed SDF grid.
    """

    def __init__(
        self,
        bone_name: str,
        sdf: dict,
        num_samples: int = 10,
        weight: float = 1.0
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

    def tree_flatten(self):
        leaves = (self.weight, self.sdf_grid, self.sdf_origin, self.sdf_spacing)
        aux = (self.bone_name, self.num_samples)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        bone_name, num_samples = aux
        weight, grid, origin, spacing = leaves
        sdf = {"grid": grid, "origin": origin, "spacing": spacing}
        return cls(bone_name, sdf, num_samples, weight)

    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'bone_name', 'num_samples', 'weight', 'sdf'.
        """
        if "bone_name" in p: self.bone_name = p["bone_name"]
        if "num_samples" in p: self.num_samples = int(p["num_samples"])
        if "weight" in p: self.weight = jnp.asarray(p["weight"], jnp.float32)
        if "sdf" in p:
            self.sdf_grid = p["sdf"]["grid"]
            self.sdf_origin = p["sdf"]["origin"]
            self.sdf_spacing = p["sdf"]["spacing"]

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return {
            "bone_name": self.bone_name,
            "num_samples": self.num_samples,
            "weight": float(self.weight),
            "sdf": {
                "grid": np.asarray(self.sdf_grid),
                "origin": np.asarray(self.sdf_origin),
                "spacing": float(self.sdf_spacing),
            },
        }

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
        return jax.scipy.ndimage.map_coordinates(self.sdf_grid, coords.T, order=1, mode='constant', cval=jnp.inf)

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
        weight: float = 1.0
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

    def tree_flatten(self):
        leaves = (self.min_dist, self.weight)
        aux = (self.bone_names, self.num_samples_per_bone)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        bone_names, num_samples = aux
        min_dist, weight = leaves
        return cls(bone_names, num_samples, min_dist, weight)

    def update_params(self, p: dict) -> None:
        """
        Update the parameters of the objective.

        Args:
            p (dict): Dictionary with keys 'bone_names', 'num_samples_per_bone', 'min_dist', 'weight'.
        """
        if "bone_names" in p: self.bone_names = tuple(p["bone_names"])
        if "num_samples_per_bone" in p: self.num_samples_per_bone = int(p["num_samples_per_bone"])
        if "min_dist" in p: self.min_dist = jnp.float32(p["min_dist"])
        if "weight" in p: self.weight = jnp.asarray(p["weight"], jnp.float32)

    def get_params(self) -> dict:
        """
        Get the current parameters of the objective.

        Returns:
            dict: Dictionary of parameters.
        """
        return {
            "bone_names": self.bone_names,
            "num_samples_per_bone": self.num_samples_per_bone,
            "min_dist": float(self.min_dist),
            "weight": float(self.weight),
        }

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
        return jax.scipy.ndimage.map_coordinates(sdf["grid"], coords.T, order=1, mode='constant', cval=jnp.inf)

    def _penalty_single(self, cfg: jnp.ndarray, fk_solver) -> jnp.ndarray:
        """
        Compute the self-collision penalty for a single configuration.

        Args:
            cfg (jnp.ndarray): Joint angles.
            fk_solver: FK solver.

        Returns:
            jnp.ndarray: Penalty value.
        """
        if not hasattr(fk_solver, 'sdf') or fk_solver.sdf is None or \
           not hasattr(fk_solver, 'mesh_data') or fk_solver.mesh_data is None:
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
        query_points_local = inverse_skin_points(query_points_world, fk_solver, fk_solver.mesh_data, fk)

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

