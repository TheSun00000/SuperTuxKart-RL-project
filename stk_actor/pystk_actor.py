from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor
from .custom_wrapper import ActionOnlyFlattenerWrapper, ActionTimeExtentionWrapper, ExpertObservationWrapper
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations

#: The base environment name
env_name = "supertuxkart/full-v0"
# env_name = 

#: Player name
player_name = "Naxim"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = Actor(observation_space, action_space)

    # Returns a dummy actor
    # state = None
    if state is None:
        # print(action_space)
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        # lambda env: MyWrapper(env, option="1")
        lambda env: ConstantSizedObservations(env, state_paths=20, state_items=5, state_karts=5),
        lambda env: ActionOnlyFlattenerWrapper(env),
        lambda env: ActionTimeExtentionWrapper(env, n=5),
        lambda env: ExpertObservationWrapper(env)
    ]
