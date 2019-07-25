
from gym.envs.registration import register

register(
    id='my-gridworld-v0',
    entry_point='custom_gym_envs.envs:GridWorld'
)

register(
        id='my-basic-gridworld-v0',
        entry_point='custom_gym_envs.envs:BasicGridWorld'
)

register(
        id='forage-world-v0',
        entry_point='custom_gym_envs.envs:ForageWorld'
)

register(
        id='hunters-gatherers-v0',
        entry_point='custom_gym_envs.envs:HuntersAndGatherers'
)

register(
        id='hunters-gatherers-multi-v0',
        entry_point='custom_gym_envs.envs:HuntersAndGatherersMultiPlayer'
)