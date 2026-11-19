"""Small observation audit for the existing simulation runner."""
import hashlib
import numpy as np
import random
from collections.abc import Mapping


def seed_environment(seed):
    # An unseeded reset must preserve the existing RNG stream, not use entropy.
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def observation_snapshot(observations, lane):
    # AsyncVectorEnv returns one dict per lane; also accept stacked dicts.
    if isinstance(observations, Mapping):
        observation = {port: values[lane] for port, values in observations.items()}
    else:
        observation = observations[lane]
    if not isinstance(observation, Mapping):
        raise TypeError('Expected a dictionary of observations for each environment')
    result = {'sha256_by_port': {}, 'robot_state': {}}
    for port in sorted(observation):
        value = np.ascontiguousarray(observation[port])
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
        result['sha256_by_port'][port] = digest.hexdigest()
        if port in ('joint_positions', 'gripper_joint_positions'):
            result['robot_state'][port] = value.tolist()
    return result
