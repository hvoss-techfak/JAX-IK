# JAX-IK
Official implementation for the paper: "Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters" 

[More Information](https://hvoss-techfak.github.io/TF-JAX-IK/)

# Demo

You can try out the JAX-IK library in this [demo app](https://experiments.scs.techfak.uni-bielefeld.de/jaxik/)

# Install

You can install the library simply using pip:
```bash
pip install jax-ik
```

# Example code 

This is a small example code to solve the arm position on the smplx skeleton. \
Sadly, we don't have the rights to add the SMPL-X skeleton to this repository. Therefore, you will have to obtain a .glb file of the smplx skeleton yourself. \
For this install the [smplx blender addon](https://github.com/Meshcapade/SMPL_blender_addon) in blender, press on the "add" button in the Blender SMPL-X drawer. \
Finally, go to File -> Export -> glTF 2.0 and follow the export dialog to save your .glb file.

```python
import numpy as np
from jax_ik.objectives import DistanceObjTraj
from jax_ik.ik import InverseKinematicsSolver

# === Setup ===
hand = "left"
controlled_bones = [f"{hand}_collar", f"{hand}_shoulder", f"{hand}_elbow", f"{hand}_wrist"]

# Define bounds in degrees
angle_bounds_deg = {
    "left_collar": ([-10, -10, -10], [10, 10, 10]),
    "left_shoulder": ([-120, -140, -65], [70, 50, 25]),
    "left_elbow": ([-100, -180, -10], [90, 10, 10]),
    "left_wrist": ([-120, -70, -70], [90, 60, 80]),
}

# Flatten and convert to radians
bounds = []
for bone in controlled_bones:
    lower, upper = angle_bounds_deg[bone]
    for l, u in zip(lower, upper):
        bounds.append((np.radians(l), np.radians(u)))

# === Initialize Solver ===
model_file = "smplx.glb"  # Path to your SMPL-X .glb file
ik_solver = InverseKinematicsSolver(
    model_file=model_file,
    controlled_bones=controlled_bones,
    bounds=bounds,
    threshold=0.005,
    num_steps=1000,
)

# === Target Setup ===
end_effector_bone = f"{hand}_wrist"
random_target = np.array([0.3, 0.2, 0.5])  # Some reachable 3D point

mandatory_objectives = [
    DistanceObjTraj(
        target_points=[random_target],
        bone_name=end_effector_bone,
        use_head=True,
        weight=1.0,
    )
]

# === Solve IK ===
initial_angles = np.zeros(len(controlled_bones) * 3, dtype=np.float32)
solved_angles, obj_value, steps = ik_solver.solve(
    initial_rotations=initial_angles,
    learning_rate=0.2,
    mandatory_objective_functions=mandatory_objectives,
    ik_points=5,
    patience=200,
)

print(f"Solved in {steps} steps. Final objective: {obj_value:.4f}")

# === Render Result ===
ik_solver.render(angle_vector=solved_angles[-1], target_pos=[random_target], interactive=True)
```

More examples, and some open source models will be added in the following weeks.

# LICENSE
<a href="https://github.com/hvoss-techfak/TF-JAX-IK">Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters</a> © 2025 by <a href="https://github.com/hvoss-techfak">Hendric Voss</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
