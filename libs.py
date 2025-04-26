from typing import List, Tuple, Dict

from agent import Agent
from gridworld import GridWorld
from constants import Direction


import logging
logger = logging.getLogger(__name__)

def value_function_at_s(
        state : int, 
        agent : Agent, 
        env : GridWorld,
        current_value_function : Dict[int, float]) -> int:
    """
    Calculates value function at s - see sutton/ barto p.94

    Args:
        state (int): request state
        agent (Agent): agent 
        env (GridWorld): enviornment
        current_value_function (List[int]): current value estimates

    Returns:
        int: the value of being in state
    """

    policy = agent.policy

    prob_action_in_state = policy[state]

    next_states = env.get_next_states(state)
    
    value=0
    for action, prob in prob_action_in_state.items():
        
        inner_term = 0
        for next_state in next_states:
        
            transition_prob =  env.get_transition_prob(state=state,new_state = next_state, action=action)
            
            immediate_reward = \
                transition_prob * \
                env.get_reward(state=state, new_state=next_state, action=action)
            
            long_term_reward = agent.gamma * transition_prob * current_value_function[next_state]
            
            inner_term += (immediate_reward + long_term_reward)
            
        value += prob * inner_term
        
    return value


def get_greedy_policy_from_value_function_at_s(
        value_function : List[float],
        state : int,
        env : GridWorld) -> Dict[Direction, float]:
    """
    Get the greedy policy from value_function

    :param value_function: current value function
    :param state: state we need updated policy for
    :param env: gridworld
    :return: value states for each direction (candidate policy)
    """
    cell, neighbor_cells = env.get_neighbors_dict_and_self(state)

    #print(neighbor_cells)
    mapping = {}
    for direction, neighbor_cell in neighbor_cells.items():
        if neighbor_cell is None: # wall bump
            mapping.update({direction : value_function[cell.state]})
        else:
            mapping.update({direction : value_function[neighbor_cell.state]})

    return mapping

def get_updated_policy(
        value_function : Dict[int, float],
        states : List[int],
        env : GridWorld) -> Dict[int, Dict[Direction, float]]:
    """
    Get the updated greedy policy using the value function for all states
    :param value_function: current value function
    :param states: target states
    :param env: GridWorld
    :return: updated policy
    """
    policy = {}

    for state in states:
        if state == -1:
            continue
        else:
            pass

        logger.info(f"Processing {state}")

        mapping = get_greedy_policy_from_value_function_at_s(value_function=value_function, state=state, env=env)

        # get largest values
        max_value = max(mapping.values())
        greedy_policy_actions = {key for key, value in mapping.items() if value == max_value}

        logger.info(greedy_policy_actions)

        total_actions = len(greedy_policy_actions)
        prob_action = 1/total_actions

        policy[state] = {}
        for action in greedy_policy_actions:
            policy[state].update({action : prob_action})

    return policy



def sample_trajectory(
        env, 
        agent, 
        max_steps=10):
        
    trajectory = list()
    actions = list()
    rewards = list()
    
    
    terminal=False

    current_cell = env.get_initial_cell()

    import copy

    for i in range(max_steps):

        trajectory.append(current_cell)
        
        state = current_cell.state
        
        action = agent.get_action(state)
        actions.append(action)
        
        logger.debug(f"action: {action}")
        
        new_cell, _ = env.step(
            current_coordinates=current_cell.coordinates,
            action=action)
        
        logger.debug(f"new coordinates: {new_cell.coordinates}")

        reward = env.get_reward(
            state = state, 
            new_state = new_cell.state,
            action = action)
        rewards.append(reward)

        terminal = new_cell.is_terminal
        logger.debug(f"terminal: {terminal}")
        current_cell = copy.deepcopy(new_cell)
        
        if terminal:
            trajectory.append(current_cell)
            break
        else:
            pass

        agent.update(
            state=state,
            action=action,
            reward=reward,
            state_new=new_cell.state)

        # state = state_new

        if i == max_steps-1:
            logger.debug(f"max steps reached")
            break
        else:
            pass
        
    return {"trajectory" : trajectory, "actions" : actions, 'rewards' : rewards}
