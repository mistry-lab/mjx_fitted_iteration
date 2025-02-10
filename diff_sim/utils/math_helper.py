import jax
import jax.numpy as jnp

def random_quaternion(key, batch_size):
    """
    Generate a random unit quaternion for each element in the batch.
    """
    q = jax.random.normal(key, (batch_size, 4))  # Normal distribution for quaternion
    q /= jnp.linalg.norm(q, axis=-1, keepdims=True)  # Normalize to get unit quaternion
    return q

def quaternion_conjugate(q):
    """
    Computes the conjugate (inverse) of a quaternion.
    Args:
        q: A JAX array of shape (4,) representing the quaternion [w, x, y, z].

    Returns:
        A JAX array of shape (4,) representing the conjugate quaternion [w, -x, -y, -z].
    """
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.
    Args:
        q1: A JAX array of shape (4,) representing the first quaternion [w, x, y, z].
        q2: A JAX array of shape (4,) representing the second quaternion [w, x, y, z].

    Returns:
        A JAX array of shape (4,) representing the product of q1 and q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quaternion_difference(q1, q2):
    """
    Computes the relative rotation (difference) between two quaternions.
    Args:
        q1: A JAX array of shape (4,) representing the reference quaternion [w, x, y, z].
        q2: A JAX array of shape (4,) representing the current quaternion [w, x, y, z].

    Returns:
        A JAX array of shape (4,) representing the quaternion difference.
    """
    q1_conjugate = quaternion_conjugate(q1)
    return quaternion_multiply(q2, q1_conjugate)


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


def quat2vel(quat, dt):
    """
    Equivalent to mju_quat2Vel but returns the result instead of modifying an array.
    quat: (4,) array-like, quaternion in [w, x, y, z] format
    dt: scalar
    """
    # Extract the axis from quaternion imaginary part
    axis = quat[1:]
    # Compute the magnitude (sin_a_2) and normalize
    sin_a_2 = jnp.linalg.norm(axis)
    # Avoid division by zero by conditionally normalizing
    axis = jnp.where(sin_a_2 > 1e-15, axis / sin_a_2, axis)
    # Compute angle speed
    speed = 2.0 * jnp.arctan2(sin_a_2, quat[0])
    # When axis-angle is larger than pi, adjust
    speed = jnp.where(speed > jnp.pi, speed - 2.0 * jnp.pi, speed)
    # Divide by dt
    speed = speed / dt
    return axis * speed


def sub_quat(qa, qb):
    """
    Equivalent to mju_subQuat(res, qa, qb), but returns res.
    qb * quat(res) = qa  =>  quat(res) = neg(qb) * qa
    The 3D velocity is then computed from the 'difference' quaternion.
    """
    # qdif = neg(qb) * qa
    qdif = quaternion_multiply(quaternion_conjugate(qb), qa)
    # Convert difference quaternion to velocity
    return quat2vel(qdif, 1.0)