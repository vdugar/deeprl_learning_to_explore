import numpy as np
import gym
from gym import spaces
from PIL import Image
import itertools
from envs.util import utils


class NDGrid(gym.Env):

    # consts
    EMPTY = 0
    OCCUPIED = 1
    UNKNOWN = 2
    GOAL = 10

    # channel IDs
    NUM_CHANNELS = 2
    CH_GRAPH = 0
    CH_GOAL = 1

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

    def __init__(self, state_space_dims, state_space_size, map_args,
                 hop=1, partial_obs=True):
        '''Args -
        state_space_dims -- dim of state space
        state_space_size -- size of each dim. Assumes grid is symmetric.
        hop -- Determines the hop distance until which neighbors can be added to the frontier.
                e.g. hop=1 implies that only the immediate 8 neighbors will be added to the
                frontier in a 2-D grid.
        map_args -- A dictionary of arguments for map generation, must have 'type' key
                e.g. {'type':'image','filename':'XYZ.jpg'}
        partial_obs -- specifies whether the environment is partially observed
        '''
        # set up observation space
        self.observation_space = spaces.MultiDiscrete([[0, state_space_size-1]] * state_space_dims)
        self.dims = state_space_dims
        self.dim_size = state_space_size

        # set up no. of channels
        self.n_channels = NDGrid.NUM_CHANNELS

        # set up start and goal coords
        self.goal_coords = np.array([state_space_size-1] * self.dims, dtype=np.int)
        self.start_coords = np.array([0] * self.dims, dtype=np.int)

        self.hop = hop
        self.partial_obs = partial_obs

        self.map_args = map_args

        # set up actions
        self.setup_actions()

        # Reset
        self.reset()

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
            action_types = [NDGrid.ACT_QUERY, NDGrid.ACT_MOVE]
        else:
            # the state is fully observed, so the only action is to move
            action_types = [NDGrid.ACT_MOVE]
        actions = [range(-self.hop, self.hop+1)] * self.dims
        arr_list = [action_types]
        arr_list.extend(actions)

        action_list = self.cartesian(arr_list)
        action_idx = 0
        self.actions = {}
        for action in action_list:
            temp = np.array(action)
            temp[0] = 0
            if not np.all(temp == 0):
                # check for degenerate actions that stay in the same place
                self.actions[action_idx] = action
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
            self.state = [np.full([self.dim_size]*self.dims, NDGrid.UNKNOWN, dtype=np.uint8) for x in range(self.n_channels)]
            self.full_state = [np.full([self.dim_size]*self.dims, NDGrid.UNKNOWN, dtype=np.uint8) for x in range(self.n_channels)]
        else:
            # no distinction between self.state and self.full_state now.
            self.full_state = [np.full([self.dim_size]*self.dims, NDGrid.UNKNOWN, dtype=np.uint8) for x in range(self.n_channels)]
            self.state = self.full_state

        # mark all nodes as empty on the goal channel
        self.state[NDGrid.CH_GOAL] = np.full([self.dim_size]*self.dims, NDGrid.EMPTY, dtype=np.uint8)
        self.full_state[NDGrid.CH_GOAL] = np.full([self.dim_size]*self.dims, NDGrid.EMPTY, dtype=np.uint8)

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

        if action[0] == NDGrid.ACT_MOVE:
            if self.check_valid_move(coord):
                # check if it's a valid state to move into
                # update state and frontier
                if np.all(coord == self.goal_coords):
                    # check for goal
                    reward = NDGrid.REW_GOAL
                    is_terminal = True
                else:
                    self.update_frontier(self.get_neighbors(self.state_coords))
                    reward = NDGrid.REW_STEP
                    is_terminal = False

                self.state_coords = coord
            elif self.check_collision(coord):
                reward = NDGrid.REW_MOVE_COLLISION
                is_terminal = True
            else:
                # we've tried to move into an unknown state
                reward = NDGrid.REW_MOVE_UNKNOWN
                is_terminal = True
        elif action[0] == NDGrid.ACT_QUERY:
            # Check if a valid state is queried
            if any(coord < 0) or any(coord >= self.dim_size):
                reward = NDGrid.REW_BAD_QUERY
                is_terminal = False
            # check if we've already queried this node, and if it's in the frontier
            elif self.state[NDGrid.CH_GRAPH][coord_tup] != NDGrid.UNKNOWN or coord_tup not in self.frontier:
                reward = NDGrid.REW_BAD_QUERY
                # @avemula : Can be changed to True (?)
                is_terminal = False
            else:
                # query node and update state
                self.query(coord_tup)
                reward = NDGrid.REW_QUERY
                is_terminal = False

        return (self.state, self.state_coords), reward, is_terminal, debug_info

    def check_valid_move(self, state_coords):
        '''
        Checks if this is a valid state to move into.
        '''
        if any(state_coords < 0) or any(state_coords >= self.dim_size):
            return False

        if self.state[NDGrid.CH_GRAPH][tuple(state_coords)] == NDGrid.EMPTY:
            return True
        else:
            return False

    def check_collision(self, coord_tup):
        '''
        Checks if coord_tup is in collision, or going beyond the edge of the map
        '''
        if any(coord_tup < 0) or any(coord_tup >= self.dim_size):
            return True

        if self.full_state[NDGrid.CH_GRAPH][tuple(coord_tup)] == NDGrid.OCCUPIED:
            return True

        return False

    def set_map_args(self, map_args):
        self.map_args = map_args

    def generate_map(self, ):
        '''
        Generates the obstacle map whenever the environment is reset.
        Assumes the start state is (0,0...) and the goal state is (dim_size-1,dim_size-1...)
        '''
        # making sure that start/goal are not generated at the boundaries
        if 'start_goal_dist' in self.map_args and self.map_args['start_goal_dist']:
            # set goal coord to lie at start_goal_dist of start coord in each dimension
            self.start_coords = np.array([np.random.randint(1, self.dim_size-1) for x in range(self.dims)], dtype=np.int)
            hops = [-self.map_args['start_goal_dist'], self.map_args['start_goal_dist']]
            self.goal_coords = np.copy(self.start_coords)
            for i in range(self.dims):
                self.goal_coords[i] += np.random.choice(hops)
                if self.goal_coords[i] < 1:
                    self.goal_coords[i] = 1
                if self.goal_coords[i] >= self.dim_size-2:
                    self.goal_coords[i] = self.dim_size-2
        else:
            # Randomly sample start and goal coordinates
            # self.start_coords = np.array([1 for x in range(self.dims)], dtype=np.int)
            self.start_coords = np.array([np.random.randint(1, self.dim_size-1) for x in range(self.dims)], dtype=np.int)
            # self.goal_coords = np.array([3 for x in range(self.dims)], dtype=np.int)
            self.goal_coords = np.array([np.random.randint(1, self.dim_size-1) for x in range(self.dims)], dtype=np.int)

            # If start is same as goal, re-sample
            while np.array_equal(self.start_coords, self.goal_coords):
                self.start_coords = np.array([np.random.randint(1, self.dim_size-1) for x in range(self.dims)], dtype=np.int)

        # Mark start and goal states as empty
        self.state[NDGrid.CH_GRAPH][tuple(self.start_coords)] = NDGrid.EMPTY
        # @avemula : Marking goal as unknown on grid map channel
        if self.partial_obs:
            # mark goal state as unknown
            self.state[NDGrid.CH_GRAPH][tuple(self.goal_coords)] = NDGrid.UNKNOWN
        else:
            self.state[NDGrid.CH_GRAPH][tuple(self.goal_coords)] = NDGrid.EMPTY

        # TODO: Generate obstacles and update self.full_state
        # Use arguments for map type
        map_type = self.map_args['type']
        if map_type == 'image':
            assert (self.dims == 2), "Image can only be used as map_gen for 2D problem!"

            # Get image filename and load image
            img_name = self.map_args['filename']
            pil_img = Image.open(img_name).convert('L')

            # Get image array 0 - 255
            img_arr_gscale = np.asarray(pil_img, dtype=np.uint8)

            assert ((self.dim_size, self.dim_size) == img_arr_gscale.shape), "Image must be square and equal to dims"

            # Now fill in coords of full_state based on pixel values of image - 0 or 255
            for coord in itertools.product(range(self.dim_size), self.dim_size):
                if img_arr_gscale[coord] == 255:  # Empty (white) pixel
                    self.full_state[NDGrid.CH_GRAPH][coord] = NDGrid.EMPTY
                else:                            # Occupied (black) pixel
                    self.full_state[NDGrid.CH_GRAPH][coord] = NDGrid.OCCUPIED

        elif map_type == 'generator':

            num_obs = self.map_args['num_obs']
            max_width = self.map_args['max_width']

            # Get list of obstacles based on params
            obstacles = utils.generate_obstacles(self.dim_size, self.dims, num_obs, max_width, self.start_coords, self.goal_coords)

            # Mark all as free first
            self.full_state[NDGrid.CH_GRAPH] = np.full([self.dim_size]*self.dims, NDGrid.EMPTY, dtype=np.uint8)

            # For all coordinates within obstacles, mark as occupied
            # obstacles is a list of (list1,list2)
            for obs in obstacles:

                low, high = obs

                # Slice with corresponding elements of low and high
                slc = [slice(None)]*self.dims

                for (i, (l, h)) in enumerate(zip(low, high)):
                    slc[i] = slice(l, h)

                # Set slice to occupied
                self.full_state[NDGrid.CH_GRAPH][slc] = NDGrid.OCCUPIED

            # Free everything along all dimensions of start_coord
            # TODO : Generalize for ND case
            # if self.dims == 2:
                # Set whole column and row of start_coords to be empty
                # self.full_state[NDGrid.CH_GRAPH][self.start_coords[0], :] = NDGrid.EMPTY
                # self.full_state[NDGrid.CH_GRAPH][:, self.start_coords[1]] = NDGrid.EMPTY

                # self.full_state[NDGrid.CH_GRAPH][self.goal_coords[0], :] = NDGrid.EMPTY
                # self.full_state[NDGrid.CH_GRAPH][:, self.goal_coords[1]] = NDGrid.EMPTY

        elif map_type == 'empty':
            self.full_state[NDGrid.CH_GRAPH] = np.full([self.dim_size]*self.dims, NDGrid.EMPTY, dtype=np.uint8)

            # generate obstacles at the edges
            if self.dims == 2:
                for i in range(0, self.dim_size):
                    self.full_state[NDGrid.CH_GRAPH][0, i] = NDGrid.OCCUPIED
                    self.full_state[NDGrid.CH_GRAPH][i, 0] = NDGrid.OCCUPIED
                    self.full_state[NDGrid.CH_GRAPH][i, self.dim_size-1] = NDGrid.OCCUPIED
                    self.full_state[NDGrid.CH_GRAPH][self.dim_size-1, i] = NDGrid.OCCUPIED

        # randomly reveal some portion of the map
        if 'reveal_factor' in self.map_args:
            num_reveal = int((self.dim_size ** self.dims) * self.map_args['reveal_factor'])
            for i in range(0, num_reveal):
                # randomly sample coords to reveal
                coords = tuple(np.random.randint(0, self.dim_size, self.dims))
                self.state[NDGrid.CH_GRAPH][coords] = self.full_state[NDGrid.CH_GRAPH][coords]

        if 'difficulty' in self.map_args:
            # set goal state according to curriculum
            difficulty = self.map_args['difficulty']
            if self.map_args['train']:
                # set difficulty to a random number between 1-max
                difficulty = np.random.randint(1, difficulty+1)

            self.goal_coords = utils.get_random_node_at_depth(self, self.start_coords,
                                                              difficulty)
            if self.goal_coords is None:
                # recurse until we generate valid configs
                self.generate_map()

        # mark goal state on the goal channel
        self.state[NDGrid.CH_GOAL][tuple(self.goal_coords)] = NDGrid.GOAL
        self.full_state[NDGrid.CH_GOAL][tuple(self.goal_coords)] = NDGrid.GOAL

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
        self.state[NDGrid.CH_GRAPH][coords_tup] = self.full_state[NDGrid.CH_GRAPH][coords_tup]

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
