
import copy
import random
import itertools
import numpy as np

from typing import List, Tuple, Dict

from constants import Direction

import logging
logger = logging.getLogger(__name__)


class Agent:
    gamma = 1
    
    def __init__(
            self, 
            initial_state : int,
            states : List[int],
            actions : List[Direction],
            alpha : float = 0.05,
            policy : Dict = None,):
        """
        Agent with equiprobable random policy by default

        Args:
            initial_state (int): starting state
            states (List[int]): list of valid states
            actions (List[Direction]): list of valid actions
            policy (Dict, optional): policy we will follow. Defaults to None. If none does equiprobable random policy.
        """
        
        self.initial_state = initial_state
        self.states = copy.deepcopy(states)
        self.actions = actions
        self.alpha = alpha

        self._set_policy(policy)

        self.state_value_function = {}
        for state_action in itertools.product(states, actions):
            self.state_value_function[state_action] = 0.0
        logger.info(self.state_value_function)            

    @property
    def policy(self) -> Dict:
        """
        Returns:
            Dict: returs policy as dict
        """
        return self._policy
    
    def _set_policy(self, policy):
        """
        Sets the policy with given policy or defines random equiprobable

        Args:
            policy (Dict): input policy. if None does random equiprob.
        """
        
        _policy = {}
        if policy is not None:
            self._policy = policy
        else:
            # random equiprobable
            for state in self.states:
                n_actions = len(self.actions)
                prob_per_action = 1 / n_actions
                random_policy = {action : prob_per_action for action in self.actions}
                _policy.update({state: random_policy})
                
            self._policy = _policy

        
    def get_action(self, state : int) -> Direction:
        """
        Gets action based on current policy in given state

        Args:
            state (int): state we are in

        Returns:
            Direction: our action
        """
        
        probs = self.policy[state] # dict of actions
        
        actions, weights = list(probs.keys()), list(probs.values())
        
        choice = random.choices(population=actions, weights=weights, k=1)[0]

        epsilon_draw = np.random.uniform(low=0.0, high=1.0, size=1)
        epsilon = 0.05
        if epsilon < epsilon_draw:
            choice = random.choice(actions) # random choice epsilon % of time
        else:
            pass

        return choice

    
    def update(
            self,
            state : int,
            action : Direction,
            reward : float,
            state_new : int):
        """
        Update the agent's state-value function with q-learning

        :param state: current state
        :param action: action we are taking
        :param reward: reward we saw
        :param state_new: next state we are going to
        :return: none
        """

        self.update_state_value_function(state=state, action=action, reward=reward, state_new=state_new)

        self.update_policy()

    def update_state_value_function(
            self,
            state: int,
            action: Direction,
            reward: float,
            state_new : int):
        """
        Update the state value function for the given observation using Q-learning

        Args:
            state (int): current state
            action (Direction): action we took
            reward (float): reward we observed
            state_new (int): new sate
        """

        # Q-value in current state and action
        Q_s_a = self.state_value_function[(state, action)]

        # Calculate the optimal action based on state-value in next state . if multiple matches for optimal action choose random
        # Q_sprime is the set of state-action values in the next state for each action.
        Q_sprime = {sa : v for sa,v in self.state_value_function.items() if sa[0] == state_new}

        max_value = max(Q_sprime.values())
        best_next_state_actions = [key for key, value in Q_sprime.items() if value == max_value]

        opt_next_action = random.choice(best_next_state_actions) # several choices match for optimal state value

        max_Q_sprime_aprime = Q_sprime[opt_next_action] # this is the best state-action value in the next state.

        # update the q state-value for the current state by taking alpha step:
        #   in direction of : 
        #       immediate reward + long term reward attenuated by gamma (net of what we currently think the state-value is)
        self.state_value_function[(state, action)] = Q_s_a + self.alpha * (
            (reward + self.gamma * max_Q_sprime_aprime) -
            Q_s_a
        )

    def update_policy(self) -> None:
        """
        Update the policy based on the state value function with Q-learning. 
        
        See Sutton & Barto p.149
        """
        
        for state in self.states:

            Q_s = {sa: v for sa, v in self.state_value_function.items() if sa[0] == state}

            ZERO_THRESHOLD = 1e-16

            max_value = max(Q_s.values())
            best_state_actions = [key for key, value in Q_s.items() if abs(value - max_value) < ZERO_THRESHOLD]

            best_actions = [a for s,a in best_state_actions]

            num_best_actions = len(best_actions)
            best_action_prob = 1 / num_best_actions

            prob_per_action = {}
            for action in self.actions:
                if action in best_actions:
                    prob_per_action[action] = best_action_prob
                else:
                    prob_per_action[action] = 0.0

            self._policy.update({state: prob_per_action})
