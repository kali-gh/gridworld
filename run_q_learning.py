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

# inits
initial_state = 1
states = env.get_states()
actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

agent = Agent(
    initial_state = initial_state,
    states=states,
    actions=actions)

num_trajectories = 100
for k in range(num_trajectories):
    libs.sample_trajectory(env, agent)

logger.info("Current state value function")
logger.info(agent.state_value_function)

logger.info("Current policy")
logger.info(agent.policy)