from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

# Register DataClass as a PyTree node
@jax.tree_util.register_pytree_node_class
@dataclass
class DataClass:
    qpos: jnp.ndarray
    qvel: jnp.ndarray

    # Define how to flatten the class into its children (fields)
    def tree_flatten(self):
        # Return a tuple of children and auxiliary data
        children = (self.qpos, self.qvel)
        aux_data = None  # No auxiliary data needed
        return children, aux_data

    # Define how to reconstruct the class from its children
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        qpos, qvel = children
        return cls(qpos, qvel)


# Function to generate new data based on initial qpos and qvel
def make_data(init_qpos: jnp.ndarray, init_qvel: jnp.ndarray) -> DataClass:
    # For simplicity, let's add random noise to the initial positions and velocities
    new_qpos = init_qpos + jnp.ones(init_qpos.shape) * 0.1
    new_qvel = init_qvel + jnp.ones(init_qvel.shape) * 0.2
    return DataClass(qpos=new_qpos, qvel=new_qvel)


# Function to remove selected indices and append new data
def remove_and_append(data: DataClass, indices_to_remove: jnp.ndarray, new_data: DataClass) -> DataClass:
    # Create a mask to select elements to keep
    batch_size = data.qpos.shape[0]
    mask = ~jnp.isin(jnp.arange(batch_size), indices_to_remove)

    # Define a function to process each field
    def process_field(field, new_field):
        # Remove selected indices
        field = field[mask]
        # Append new data
        field = jnp.concatenate([field, new_field], axis=0)
        return field

    # Apply the process to each field in the dataclass
    new_data_class = jax.tree_map(process_field, data, new_data)
    return new_data_class


# Example usage
def main():
    # Set random seed for reproducibility
    np.random.seed(0)

    # Create initial data with batch size of 10 and state size of 3
    batch_size = 10
    state_size = 3
    qpos = jnp.array(np.random.rand(batch_size, state_size))
    qvel = jnp.array(np.random.rand(batch_size, state_size))
    data = DataClass(qpos=qpos, qvel=qvel)

    print("Original Data:")
    print("qpos:", data.qpos)
    print("qvel:", data.qvel)
    print()

    # Indices to remove (e.g., remove indices 2, 5, 7, 8)
    indices_to_remove = jnp.array([2, 5, 7, 8])

    # Initial conditions for new data (using the removed indices' qpos and qvel)
    init_qpos = data.qpos[indices_to_remove]
    init_qvel = data.qvel[indices_to_remove]

    # Generate new data (batch size matches the number of indices removed)
    new_data = make_data(init_qpos, init_qvel)

    # Update data by removing selected indices and appending new data
    updated_data = remove_and_append(data, indices_to_remove, new_data)

    print("Updated Data:")
    print("qpos:", updated_data.qpos)
    print("qvel:", updated_data.qvel)
    print()

    # Verify the batch size remains the same
    print("Original batch size:", data.qpos.shape[0])
    print("Updated batch size:", updated_data.qpos.shape[0])


if __name__ == "__main__":
    main()
