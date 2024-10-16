import jax
import jax.numpy as jnp


def angle_axis_to_quaternion(angle_axis):
    """
    Converts an angle-axis vector to a quaternion.

    Args:
        angle_axis: A JAX array of shape (3,) representing the angle-axis vector.

    Returns:
        A JAX array of shape (4,) representing the quaternion [w, x, y, z].
    """
    a0, a1, a2 = angle_axis
    theta = jnp.linalg.norm(angle_axis)

    def not_zero(theta, angle_axis):
        half_theta = 0.5 * theta
        k = jnp.sin(half_theta) / theta
        w = jnp.cos(half_theta)
        xyz = angle_axis * k
        quaternion = jnp.concatenate([jnp.array([w]), xyz])
        return quaternion

    def is_zero(theta, angle_axis):
        k = 0.5
        w = 1.0
        xyz = angle_axis * k
        quaternion = jnp.concatenate([jnp.array([w]), xyz])
        return quaternion

    quaternion = jax.lax.cond(
        theta > 1e-8,  # A small threshold to handle numerical precision
        not_zero,
        is_zero,
        theta,
        angle_axis
    )
    return quaternion


def quaternion_to_angle_axis(quaternion):
    """
    Converts a quaternion to an angle-axis vector.

    Args:
        quaternion: A JAX array of shape (4,) representing the quaternion [w, x, y, z].

    Returns:
        A JAX array of shape (3,) representing the angle-axis vector.
    """
    q1, q2, q3 = quaternion[1], quaternion[2], quaternion[3]
    sin_theta = jnp.linalg.norm(quaternion[1:4])

    def not_zero(sin_theta, quaternion):
        cos_theta = quaternion[0]

        two_theta = 2.0 * jnp.where(
            cos_theta < 0.0,
            jnp.arctan2(-sin_theta, -cos_theta),
            jnp.arctan2(sin_theta, cos_theta)
        )

        k = two_theta / sin_theta
        angle_axis = quaternion[1:4] * k
        return angle_axis

    def is_zero(sin_theta, quaternion):
        k = 2.0
        angle_axis = quaternion[1:4] * k
        return angle_axis

    angle_axis = jax.lax.cond(
        sin_theta > 1e-8,  # A small threshold to handle numerical precision
        not_zero,
        is_zero,
        sin_theta,
        quaternion
    )
    return angle_axis
