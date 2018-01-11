import numpy as np
import gym
from gym import spaces
from PIL import Image
import itertools
from envs.util import utils
import math

class Manipulator(gym.Env):

    # consts
    EMPTY = 0
    OCCUPIED = 1
    UNKNOWN = 2
    GOAL = 10

    # channel IDs
    NUM_CHANNELS = 3
    CH_GRAPH = 0
    CH_GOAL = 1
    CH_2D = 2

    # action types
    ACT_QUERY = 0
    ACT_MOVE = 1

    # rewards
    REW_GOAL = 1             # reward for reaching goal
    REW_STEP = -0.01            # reward for each step
    REW_BAD_QUERY = -0.01  # -10          # reward for querying a "bad" state
    REW_QUERY = -0.01
    REW_MOVE_COLLISION = -1
    REW_MOVE_UNKNOWN = -1

    def __init__(self, state_space_dims, d_theta, lengths, map_args,
                 hop=1, partial_obs=False, map_data=None, cached_maps=None):
        '''Args -
        state_space_dims -- dim of state space
        d_theta -- step in angle space for each dof. MUST be a factor of 2pi
        lengths -- np array of link lengths. Should be in (0, 1)
        hop -- Determines the hop distance until which neighbors can be added to the frontier.
                e.g. hop=1 implies that only the immediate 8 neighbors will be added to the
                frontier in a 2-D grid.
        map_args -- A dictionary of arguments for map generation, must have 'type' key
                e.g. {'type':'image','filename':'XYZ.jpg'}
        partial_obs -- specifies whether the environment is partially observed
        '''
        # set up observation space
        state_space_size = int(2*math.pi / d_theta)
        self.d_theta = d_theta
        self.dims = state_space_dims
        self.dim_size = state_space_size
        self.lengths = lengths

        # set up no. of channels
        self.n_channels = Manipulator.NUM_CHANNELS

        # set up start and goal coords
        self.goal_coords = np.array([state_space_size-1] * self.dims, dtype=np.int)
        self.start_coords = np.array([0] * self.dims, dtype=np.int)

        self.hop = hop
        self.partial_obs = partial_obs

        self.map_args = map_args

        # set up actions and list of states
        self.setup_actions()
        self.setup_states_list()
        self.maps = cached_maps
        self.map_data = map_data

        # Reset
        self.reset()

    def setup_states_list(self):
        self.states_list = []
        state_coords = [range(0, self.dim_size)] * self.dims
        state_list = self.cartesian(state_coords)
        for state in state_list:
            self.states_list.append(state)        

    def setup_actions(self):
        '''
        Sets up the action space for this grid.
        Actions are of the form (act_type, dim1_hop, dim2_hop, ....)
        ACT_QUERY -- queries a given edge, and recovers its cost
        ACT_MOVE  -- moves to a particular state, and updates the frontier
        These actions can be specified for nodes upto +/- 'hop' nodes away.
        '''
        if self.partial_obs:
            # the state is partially observed, so we must both explore (query), and move
            action_types = [Manipulator.ACT_QUERY, Manipulator.ACT_MOVE]
        else:
            # the state is fully observed, so the only action is to move
            action_types = [Manipulator.ACT_MOVE]
        actions = [range(-self.hop, self.hop+1)] * self.dims
        arr_list = [action_types]
        arr_list.extend(actions)

        action_list = self.cartesian(arr_list)
        action_idx = 0
        self.actions = {}
        self.action_id_lookup = {}
        for action in action_list:
            temp = np.array(action)
            temp[0] = 0
            if not np.all(temp == 0):
                # check for degenerate actions that stay in the same place
                self.actions[action_idx] = action
                self.action_id_lookup[tuple(action)] = action_idx
                action_idx += 1

        self.n = action_idx

    def _reset(self):
        '''
        Internally, state is maintained as a list of np arrays, with each np array representing one
        channel. The coordinates of the state are also maintained separately.

        The state that is reported to the outside world is a tuple of two elements-
        -- the first element represents the list of np arrays described above.
        -- the second element contains the state coordinates as a numpy array.
        '''

        if self.partial_obs:
            # maintain two distinct copies of the state. self.state is partially observed
            # while self.full_state holds the complete state, but is only accessible internally
            self.state = [np.full([self.dim_size]*self.dims, Manipulator.UNKNOWN, dtype=np.uint8) for x in range(self.n_channels)]
            self.full_state = [np.full([self.dim_size]*self.dims, Manipulator.UNKNOWN, dtype=np.uint8) for x in range(self.n_channels)]
        else:
            # no distinction between self.state and self.full_state now.
            self.full_state = [np.full([self.dim_size]*self.dims, Manipulator.UNKNOWN, dtype=np.uint8) for x in range(self.n_channels)]
            self.state = self.full_state

        # mark all nodes as empty on the goal channel
        self.state[Manipulator.CH_GOAL] = np.full([self.dim_size]*self.dims, Manipulator.EMPTY, dtype=np.uint8)
        self.full_state[Manipulator.CH_GOAL] = np.full([self.dim_size]*self.dims, Manipulator.EMPTY, dtype=np.uint8)

        # generate obstacle map using arguments for map
        self.generate_map()

        self.state_coords = np.copy(self.start_coords)

        # add frontier nodes for start state
        self.update_frontier(self.get_neighbors(self.state_coords))

        return (self.state, self.state_coords)

    def _step(self, action_idx):
        action = self.actions[action_idx]

        coord = self.state_coords + action[1:]
        coord_tup = tuple(coord)
        reward = 0
        is_terminal = False
        debug_info = {}

        if action[0] == Manipulator.ACT_MOVE:
            if self.check_valid_move(coord):
                # check if it's a valid state to move into
                # update state and frontier
                if np.all(coord == self.goal_coords):
                    # check for goal
                    reward = Manipulator.REW_GOAL
                    is_terminal = True
                else:
                    self.update_frontier(self.get_neighbors(self.state_coords))
                    reward = Manipulator.REW_STEP
                    is_terminal = False

                self.state_coords = coord
            elif self.check_collision(coord):
                reward = Manipulator.REW_MOVE_COLLISION
                is_terminal = True
            else:
                # we've tried to move into an unknown state
                reward = Manipulator.REW_MOVE_UNKNOWN
                is_terminal = True
        elif action[0] == Manipulator.ACT_QUERY:
            # Check if a valid state is queried
            if any(coord < 0) or any(coord >= self.dim_size):
                reward = Manipulator.REW_BAD_QUERY
                is_terminal = False
            # check if we've already queried this node, and if it's in the frontier
            elif self.state[Manipulator.CH_GRAPH][coord_tup] != Manipulator.UNKNOWN or coord_tup not in self.frontier:
                reward = Manipulator.REW_BAD_QUERY
                # @avemula : Can be changed to True (?)
                is_terminal = False
            else:
                # query node and update state
                self.query(coord_tup)
                reward = Manipulator.REW_QUERY
                is_terminal = False

        return (self.state, self.state_coords), reward, is_terminal, debug_info

    def check_valid_move(self, state_coords):
        '''
        Checks if this is a valid state to move into.
        '''
        if any(state_coords < 0) or any(state_coords >= self.dim_size):
            return False

        if self.state[Manipulator.CH_GRAPH][tuple(state_coords)] == Manipulator.EMPTY:
            return True
        else:
            return False

    def check_collision(self, coord_tup):
        '''
        Checks if coord_tup is in collision, or going beyond the edge of the map
        '''
        if any(coord_tup < 0) or any(coord_tup >= self.dim_size):
            return True

        if self.full_state[Manipulator.CH_GRAPH][tuple(coord_tup)] == Manipulator.OCCUPIED:
            return True

        return False

    def set_map_args(self, map_args):
        self.map_args = map_args

    def set_maps_data(self, data):
        self.maps = data

    def angle_in_pi(self, angle):
        ''' Restricts 'angle' to lie between -pi and pi '''
        if angle > math.pi:
            angle -= 2*math.pi

        return angle

    def coord_to_angle(self, coords):
        ''' Converts coords to np array of angles in (-pi, pi) '''
        angles = coords * self.d_theta

        return map(self.angle_in_pi, angles)

    def generate_map(self, ):
        '''
        Generates the obstacle map whenever the environment is reset.
        '''

        if self.map_data is not None:
            self.start_coords = self.map_data[0]
            self.goal_coords = self.map_data[1]
            self.full_state[Manipulator.CH_GRAPH] = self.map_data[2]
            self.full_state[Manipulator.CH_GOAL] = self.map_data[3]
            if not self.partial_obs:
                self.state = self.full_state
            self.obstacles = self.map_data[4]
            self.shortest_path = self.map_data[5]
            self.action_history = self.map_data[6]
        elif self.maps is not None:
            # randomly sample a map from the dataset
            num_maps = self.maps.shape[0]
            map_data = self.maps[np.random.randint(0, num_maps)]
            self.start_coords = map_data[0]
            self.goal_coords = map_data[1]
            self.full_state[Manipulator.CH_GRAPH] = map_data[2]
            self.full_state[Manipulator.CH_GOAL] = map_data[3]
            self.obstacles = map_data[4]
            self.shortest_path = map_data[5]
            self.action_history = map_data[6]
        else:
            # Randomly sample start and goal coordinates
            self.start_coords = np.array([np.random.randint(1, self.dim_size-1) 
                for x in range(self.dims)], dtype=np.int)
            self.goal_coords = np.array([np.random.randint(1, self.dim_size-1) 
                for x in range(self.dims)], dtype=np.int)

            # If start is same as goal, re-sample
            while np.array_equal(self.start_coords, self.goal_coords):
                self.start_coords = np.array([np.random.randint(1, self.dim_size-1) 
                    for x in range(self.dims)], dtype=np.int)


            # Use arguments for map type
            if self.map_args['type'] == 'generator':
                num_obs = self.map_args['num_obs']
                max_width = self.map_args['max_width']
                min_width = self.map_args['min_width']
                start_angles = self.coord_to_angle(self.start_coords)
                goal_angles = self.coord_to_angle(self.goal_coords)

                # Get list of obstacles based on params
                obstacles = utils.generate_obstacles_continuous(num_obs, min_width, max_width,
                    self.lengths, start_angles, goal_angles)
                self.obstacles = obstacles

                # Mark all as free first
                self.full_state[Manipulator.CH_GRAPH] = np.full(
                    [self.dim_size]*self.dims, Manipulator.EMPTY, dtype=np.uint8)

                # Iterate through ALL states and check for C-space collisions. Sigh.
                count = 1
                for state in self.states_list:
                    # print("checking")
                    angle = self.coord_to_angle(state)
                    if utils.is_manip_in_self_collision(self.lengths, angle):
                        self.full_state[Manipulator.CH_GRAPH][tuple(state)] = Manipulator.OCCUPIED
                        count = count + 1
                    elif utils.is_manip_in_env_collision(self.lengths, angle, obstacles):
                        self.full_state[Manipulator.CH_GRAPH][tuple(state)] = Manipulator.OCCUPIED
                        # count = count + 1
                # print count
            elif self.map_args['type'] == 'cached':
                # use map provided by map_args, and randomly set start and goal
                self.full_state[Manipulator.CH_GRAPH] = np.copy(self.map_args['cached_map'])

                # randomly sample start coordinates, and sample until it is collision-free
                self.start_coords = np.array([np.random.randint(1, self.dim_size-1) 
                        for x in range(self.dims)], dtype=np.int)
                while self.check_collision(self.start_coords):
                    self.start_coords = np.array([np.random.randint(1, self.dim_size-1) 
                        for x in range(self.dims)], dtype=np.int)

                self.obstacles = self.map_args['cached_obstacles']

            # randomly reveal some portion of the map
            if self.partial_obs and 'reveal_factor' in self.map_args:
                num_reveal = int((self.dim_size ** self.dims) * self.map_args['reveal_factor'])
                for i in range(0, num_reveal):
                    # randomly sample coords to reveal
                    coords = tuple(np.random.randint(0, self.dim_size, self.dims))
                    self.state[Manipulator.CH_GRAPH][coords] = self.full_state[Manipulator.CH_GRAPH][coords]

            if 'difficulty' in self.map_args:
                # set goal state according to curriculum
                self.goal_coords, self.shortest_path = utils.get_random_node_at_depth(
                    self, self.start_coords, self.map_args['difficulty'])
                if self.goal_coords is None:
                    # recurse until we generate valid configs
                    # print("recursing")
                    self.generate_map()

                # set up action history
                self.action_history = []
                for i in range(1, len(self.shortest_path)):
                    action = self.shortest_path[i] - self.shortest_path[i-1]
                    action = np.insert(action, 0, Manipulator.ACT_MOVE)
                    self.action_history.append(self.action_id_lookup[tuple(action)])

        # Mark start and goal states as empty
        self.state[Manipulator.CH_GRAPH][tuple(self.start_coords)] = Manipulator.EMPTY
        if self.partial_obs:
            self.state[Manipulator.CH_GRAPH][tuple(self.goal_coords)] = Manipulator.UNKNOWN
        else:
            self.state[Manipulator.CH_GRAPH][tuple(self.goal_coords)] = Manipulator.EMPTY

        # mark goal state on the goal channel
        self.state[Manipulator.CH_GOAL][tuple(self.goal_coords)] = Manipulator.GOAL
        self.full_state[Manipulator.CH_GOAL][tuple(self.goal_coords)] = Manipulator.GOAL


    def get_neighbors(self, state_coords):
        '''
        Returns the neighbors for an arbitrary state as a list of tuples,
        where each tuple represents the coordinates of that node in the grid.
        '''
        neighbors = []
        if self.partial_obs:
            high = int(self.n/2)
        else:
            high = self.n

        for i in range(0, high):
            action = self.actions[i]
            new_coord = state_coords + action[1:]
            if np.all(new_coord >= 0) and np.all(new_coord < self.dim_size):
                # valid coordinates
                neighbors.append(new_coord)

        return neighbors

    def query(self, coords_tup):
        '''
        Queries whether this node is in collision, and updates the state.
        Accepts a tuple.
        '''
        self.state[Manipulator.CH_GRAPH][coords_tup] = self.full_state[Manipulator.CH_GRAPH][coords_tup]

    def update_frontier(self, neighbors):
        '''
        Updates the frontier with the current set of neighbors
        '''
        # for now, just flush the current frontier and add the new neighbors
        self.frontier = {}
        for n in neighbors:
            self.frontier[tuple(n)] = True

    def _render(self, **kwargs):
        pass

    def render_2D(self, grid_size):
        pass

    def cartesian(self, arrays, out=None):
        """
        Generate a cartesian product of input arrays.

        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])

        http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = np.int

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = int(n / arrays[0].size)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
        return out
