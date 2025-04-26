import random
import numpy as np
import copy

from constants import Direction
from typing import Dict, List, Tuple

import logging

logger = logging.getLogger(__name__)


class GridCell:

    def __init__(
        self, 
        *,
        state : int, 
        coordinates : Tuple,
        cell_params : Dict
    ):
        """
        Initializes a GridWorld cell. Each GridWorld is composed of n_x by n_y cells.
        
        After init basic facts about the cell are saved as shown

        Args:
            state (int): id number for the cell
            coordinates (tuple): cell coordinates in the world
            cell_params (dict): params for the cell
        """
        
        self.state = state    
        self.is_terminal = (state == -1)
        self.coordinates = coordinates
        
        self.world_n_x = cell_params["world_n_x"]
        self.world_n_y = cell_params["world_n_y"]
        
        self.up_neighbor_coordinates = cell_params['up']
        self.right_neighbor_coordinates = cell_params['right']
        self.down_neighbor_coordinates = cell_params['down']
        self.left_neighbor_coordinates = cell_params['left']
    
        self.has_border_up = (self.up_neighbor_coordinates[0] <= -1)
        self.has_border_right = (self.right_neighbor_coordinates[1] >= self.world_n_x)
        self.has_border_down = (self.down_neighbor_coordinates[0] >= self.world_n_y)
        self.has_border_left = (self.left_neighbor_coordinates[1] <= -1)
    
        
    def is_terminal(self):
        """
        Whether the state is teriminal

        Returns:
            bool: True when terminal
        """
        return self.is_terminal
        
    

class GridWorld:
    
    def __init__(
            self, 
            filename : str,
            initial_coordinates : Tuple =(0,1)):
        """
        Initializes a GridWorld. Each world is composed of n_x by n_y GridCells.
        
        example 4x4 gridworld structure (-1 is the terminal state)
        
            -1 1 2 3
            4 5 6 7
            8 9 10 11
            12 13 14 -1

        Args:
            filename (str, optional): name to load the gridworld from.
            initial_coordinates (tuple, optional): initial coordinates for the agent. Defaults to (0,1).
        """
        
        self.filename = filename
        
        self.initial_coordinates = initial_coordinates
        
        self.n_x = None
        self.n_y = None
                
        self.TERMINAL_STATE = -1 
    
        self.states = list()
        
        self._init_from_file(self.filename)
    
    def get_states(self) -> List[int]:
        """
        Returns:
            List: Returns the states in the gridworld
        """
        return np.unique([int(s.state) for s in self.states]).tolist()
    
    def get_initial_cell(self) -> GridCell:
        """
        Returns:
            GridCell: cell for the initial state
        """
        return self.world[self.initial_coordinates[0]][self.initial_coordinates[1]]
    
    def _init_from_file(self, filename):
        """
        Initializes the gridworld from a file. For example see init function.

        Args:
            filename (str): file to init from.
        """
        
        logger.info(f'loading gridworld from file: {filename}: ')
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line.split() for line in lines]
            lines = [[int(x) for x in line] for line in lines]
            
            logger.info("World structure: ")
            logger.info(lines)
        
        self.states = list()
            
        self.n_y = len(lines)
        self.n_x = len(lines[0])
        
        logger.info(f"World size : n_x : {self.n_x}, n_y : {self.n_y}")
        
        world = list()
        
        for row_idx, line in enumerate(lines):
            world.append([])
            
            for col_idx, cell in enumerate(line):                
                coordinates_up = (row_idx - 1, col_idx)
                coordinates_right = (row_idx, col_idx + 1)
                coordinates_down = (row_idx + 1, col_idx)
                coordinates_left = (row_idx, col_idx - 1)            
                    
                cell = GridCell(
                    state=int(cell),
                    coordinates = (row_idx, col_idx),
                    cell_params = {
                        "world_n_x" : self.n_x,
                        "world_n_y" : self.n_y,
                        "up" : coordinates_up,
                        "right" : coordinates_right,
                        "down" : coordinates_down,
                        "left" : coordinates_left
                    })
                
                self.states.append(cell)
                world[row_idx].append(cell)
                            
        self.world = world                
        
    def size(self) -> Tuple:
        """
        Returns:
            Tuple: Size of the gridworld n_x by n_y
        """
        return (self.n_x, self.n_y)
    

    def step(
            self, 
            current_coordinates : Tuple, 
            action : Direction) -> Tuple[GridCell, bool]:
        """
        Takes a step for the agent in the given action's direction from current_coordinates and returns the new cell and whether we hit a wall

        Args:
            current_coordinates (Tuple): starting coordinates
            action (Direction): what direction we want to go

        Raises:
            ValueError: if we are passed an invalid action

        Returns:
            tuple:
                [0] GridCell we end up in
                [1] if we hit a wall
        """
        
        if action == Direction.UP:
            new_coordinates = (current_coordinates[0] - 1, current_coordinates[1])
        elif action == Direction.RIGHT:
            new_coordinates = (current_coordinates[0], current_coordinates[1] + 1)
        elif action == Direction.DOWN:
            new_coordinates = (current_coordinates[0] + 1, current_coordinates[1])
        elif action == Direction.LEFT:
            new_coordinates = (current_coordinates[0], current_coordinates[1] - 1)
        else:
            raise ValueError(f"Invalid action {action}")
        
        hit_wall = False
        if new_coordinates[0] <= -1 or new_coordinates[0] >= self.n_y or new_coordinates[1] <= -1 or new_coordinates[1] >= self.n_x:
            # we hit a wall
            logger.debug("Hit a wall")
            new_coordinates = current_coordinates
            hit_wall = True
        else:
            pass
        
        return self.world[new_coordinates[0]][new_coordinates[1]], hit_wall
    
    def state_to_cell(self, state : int) -> GridCell:
        """
        Converts a state id to a GridCell for a non-terminal cell
        
        Note that terminal state has multiple cells so we only allow this with non-terminal

        Args:
            state (int): cell id

        Raises:
            ValueError: when requesting teriminal state cell (never needed by the agent)
            AssertionError: if somehow terminal state is mislabelled and exists multiple times in the world. (Should never happen unless world is misconfigured)

        Returns:
            GridCell: Cell cooresponding to state id
        """
        if state == -1:
            raise ValueError("termimal state matches multiple cells")
        else:
            pass
        
        cells = list()
        for row_idx, line in enumerate(self.world):            
            for col_idx, cell in enumerate(line):       
                if cell.state == state:
                    cells.append(cell) # can have multiple terminal state cells. done like this in case impl. needs to change somewhere in future
                else:
                    pass
                
        assert(len(cells) == 1)
        
        return cells[0]
    
    def get_next_states(self, state : int) -> List[int]:
        """
        Get the states next an input state as list of int for a non-terminal state
        
        Agent never needs to know who is next to a terimal state since the episode must end when reaching it

        Args:
            state (int): state requested

        Raises:
            ValueError: if we request teriminal state

        Returns:
            List[int]: states next to the input state
        """
        if state == -1:
            raise ValueError("terminal state has no neighbors.")
        else:
            pass
        
        neighbors = self._get_next_cells(state)
        neighbors_states = [n.state for n in neighbors]
            
        return neighbors_states
    
    def _get_next_cells(self, state : int) -> List[GridCell]:
        """
        Gets cells next to the input state
        
        Returns a list of references.  Can't be called with terminal cell

        Args:
            state (int): requested state

        Returns:
            List[GridCell]: List of cells next to the input state
        """

        cell, neighbors_dict = self.get_neighbors_dict_and_self(state)

        neighbors_all = list(neighbors_dict.values())

        neighbors = [n for n in neighbors_all if n is not None]
        
        if cell.has_border_up or cell.has_border_right or cell.has_border_down or cell.has_border_left:
            neighbors.append(cell) # current state is a neighbor if it has borders since the agent can bump into walls
        else:
            pass
        
        return neighbors

    def get_neighbors_dict_and_self(self, state : int) -> Tuple[GridCell, Dict[int, GridCell]]:
        cell = self.state_to_cell(state)

        up = self.get_cell_with_coordinates(cell.up_neighbor_coordinates)
        right = self.get_cell_with_coordinates(cell.right_neighbor_coordinates)
        down = self.get_cell_with_coordinates(cell.down_neighbor_coordinates)
        left = self.get_cell_with_coordinates(cell.left_neighbor_coordinates)

        neighbors_dict = {}

        neighbors_dict.update({Direction.UP : up})
        neighbors_dict.update({Direction.RIGHT : right})
        neighbors_dict.update({Direction.DOWN : down})
        neighbors_dict.update({Direction.LEFT : left})

        return cell, neighbors_dict

    def get_cell_with_coordinates(self, coordinates : Tuple) -> GridCell:
        """
        Get a GridCell given coordinates

        Args:
            coordinates (Tuple): tuple x y in the grid world we want the cell for

        Returns:
            GridCell: the cooresponding cell, or None if the coordinates don't exist in the world
        """
        if coordinates[0] <= -1 or coordinates[0] >= self.n_y or coordinates[1] <= -1 or coordinates[1] >= self.n_x:
            logger.debug(f"Requested cell coordinates {(coordinates[0], coordinates[1])} do not exist in the world")
            return None # doesn't exist in the world
        else:
            return self.world[coordinates[0]][coordinates[1]]
        

        
    def get_reward(self, state : int, new_state : int, action : Direction) -> int:
        """
        Get reward. Always -1 unless terminal state then 0.

        Args:
            state (int): input state we start at
            new_state (int): unused
            action (Direction): unused

        Returns:
            int:  Reward. Always -1 unless terminal state then 0.
        """
        
        if state == -1:
            return 0
        else:
            return -1

    def get_transition_prob(
            self, state : int, 
            new_state :int, 
            action : Direction) -> float:
        
        """
        Get prob of transition to a new state
        Bew state can be next to us, or us if we bump into a wall. 

        Args:
            state (int): current state
            new_state (int): target state
            action (Direction): direction we want to go in

        Returns:
            float: probability of transition. 
                always 1 if we go in the direction we are targeting, otherwise 0. 
                if we bump into a wall, we need to want to go in that wall direction and be "targeting ourselves". See Sutton/Barto for more info p.94
        """
        
        cell_state = self.state_to_cell(state)
        
        new_cell_known, hit_wall = self.step(current_coordinates = cell_state.coordinates, action=action)
        
        if hit_wall:
            if state == new_state:
                logger.debug("hit wall")
                return 1 # 1 if the query state matches our state.
            else:
                pass
        else:
            pass
        
        next_cells = self._get_next_cells(state)
        
        logger.debug([c.state for c in next_cells])

        cells_new_state = [c for c in next_cells if c.state == new_state]
        
        logger.debug([c.state for c in cells_new_state])
        
        assert(len(cells_new_state)==1)
        
        cell_new_state = cells_new_state[0]
                
        matches = (
            (new_cell_known.coordinates[0] == cell_new_state.coordinates[0]) and
            (new_cell_known.coordinates[1] == cell_new_state.coordinates[1]) 
        )
        
        if matches:
            return 1 # the query state matches the state we should reach when we take action so prob is 1.
        else:
            return 0

        