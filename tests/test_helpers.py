import numpy as np


def test_fk_and_bone_head_tail(solver):
    fk_solver = solver.fk_solver
    import jax.numpy as jnp
    zero = jnp.zeros(len(fk_solver.controlled_indices)*3, dtype=jnp.float32)
    fk = fk_solver.compute_fk_from_angles(zero)
    assert fk.shape[1:] == (4,4)
    # Pick first controlled bone if available, else a known name
    bone_name = solver.controlled_bones[0]
    h, t = fk_solver.get_bone_head_tail_from_fk(fk, bone_name)
    assert h.shape == (3,)
    assert t.shape == (3,)


def test_quaternion_to_matrix_roundtrip():
    from jax_ik.helper import quaternion_to_matrix
    # identity quaternion
    q = np.array([0,0,0,1], dtype=np.float32)
    M = quaternion_to_matrix(q)
    assert M.shape == (4,4)
    R = M[:3,:3]
    should_be_I = R @ R.T
    assert np.allclose(should_be_I, np.eye(3), atol=1e-5)


def test_inverse_skin_points_fallback():
    import jax.numpy as jnp

    from jax_ik.helper import inverse_skin_points
    # points
    pts = jnp.array([[0.0,0.0,0.0],[0.01,0.0,0.0]], dtype=jnp.float32)
    class DummyFK:
        bind_fk = jnp.eye(4)[None,:,:]
    out = inverse_skin_points(pts, DummyFK(), {}, jnp.eye(4)[None,:,:])
    # Should return input unchanged due to early fallback
    assert out.shape == pts.shape
    assert np.allclose(out, pts)


