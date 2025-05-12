### Background

This repository implements various reinforcement learning algorithms from Sutton & Barto's "Reinforcement Learning : An Introduction" (1998).

We focus on the "Gridworld" example as outlined in example 4.1 on page 92.

Currently the repository implements:

- Value iteration
- Q-learning

### Structure

The code is structured to be extensible.

Core behaviors are implemented in objects. For instance:

- The environment is implemented in gridworld.py : GridWorld and GridCell.
- The agent is implemented in agent.py : Agent

Algorithms are implemented procedurally in files, using those objects.

- Value iteration : run_value_iteration.py using helpers in libs.py
- Q-learning: run_q_learning.py using helpers in libs.py

### Set up

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Testing

To run value iteration

```
python run_value_iteration.py
```

To run q-learning

```
python run_q_learning.py
```

The code was tested as shown in run_q_learning.py and run_value_iteration.py.

- Value Iteration: The value iteration appears to match closely with Sutton & Barto's development. The learned policy matches the optimal policy found by Sutton and Barto. Note that we fix the random policy the whole way rather than update which might explain small differences in the value function at each iteration.
- Q-Learning: The learned policy matches closely with Sutton and Barto, except in states 6 and 9 where the optimal policy is random. This needs more investigation, otherwise the policy matches the optimal policy as reported in Sutton and Barto.
