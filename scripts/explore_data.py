# scripts/explore_data.py
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import jax
import jax.numpy as jnp
from waymax import config, dataloader, visualization
import matplotlib.pyplot as plt

print(f"JAX device: {jax.devices()}")

# Explicitly point to the correct GCS path
data_config = config.DatasetConfig(
    path='gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000',
    max_num_objects=8,
)

scenarios = dataloader.simulator_state_generator(data_config)
scenario = next(scenarios)

print(f"Num objects: {scenario.num_objects}")
print(f"Timesteps: {scenario.sim_trajectory.num_timesteps}")
print(f"SDC index: {jnp.where(scenario.object_metadata.is_sdc)[0]}")

# Render and save
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# visualization.plot_simulator_state(scenario, use_log_traj=True, ax=ax)
# plt.savefig("scenario_render.png", dpi=150, bbox_inches="tight")
# print("Saved scenario_render.png")

# Replace the visualization block with this
img = visualization.plot_simulator_state(scenario, use_log_traj=True)
plt.imshow(img)
plt.axis('off')
plt.savefig("scenario_render.png", dpi=150, bbox_inches="tight")
print("Saved scenario_render.png")