import copy
import numpy as np

import libs

from agent import Agent
from gridworld import GridWorld
from constants import Direction

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()
logger.info('init')

filename='worlds/gridworld_4-4.txt'
env = GridWorld(filename=filename)

initial_state = 1
states = env.get_states()
actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

agent = Agent(
    initial_state = initial_state,
    states=states,
    actions=actions)


current_value_function = {s : 0.0 for s in states}
updated_value_function = copy.deepcopy(current_value_function)

num_iterations = 100

for k in range(num_iterations):
    for state in states:
        if state == -1:
            pass
        else:
            updated_value_function[state] = float(
                np.round(libs.value_function_at_s(
                    state=state,
                    agent=agent,
                    env=env,
                    current_value_function=current_value_function),
                    3)
            )

    logger.debug(f"Handling iteration {k} of {num_iterations}")
    logger.debug("Current value function : ")
    logger.debug(current_value_function)

    current_value_function = copy.deepcopy(updated_value_function)


logger.info("Current value function : ")
logger.info(current_value_function)

updated_policy = libs.get_updated_policy(
    value_function=current_value_function,
    states=states,
    env=env)

logger.info("Updated policy : ")
logger.info(updated_policy) # this policy optimal after about 100 iterations