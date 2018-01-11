import torch
import torch.utils.data as data
import numpy as np
import init_paths
from envs.util import utils
from envs.manipulator import Manipulator


class GridworldDataManip(data.Dataset):
    def __init__(self, data_path, imsize, max_difficulty, train=True):
        self.data_path = data_path
        self.imsize = imsize
        self.max_difficulty = max_difficulty
        self.train = train

        self.images, self.S1, self.S2, self.S3, self.labels = self._process(data_path, max_difficulty)

    def __getitem__(self, index):
        img = np.copy(self.images[index])
        obs_grid = img[0]
        goal_channel = img[1]

        img = np.array([obs_grid, goal_channel], dtype=np.float32)
        s1 = self.S1[index]
        s2 = self.S2[index]
        s3 = self.S3[index]

        label = self.labels[index]

        img = torch.from_numpy(img)

        return img, int(s1), int(s2), int(s3), int(label)

    def __len__(self):
        return self.images.shape[0]

    def _process(self, data_path, max_difficulty):
        images = []
        S1 = []
        S2 = []
        S3 = []
        labels = []

        for difficulty in range(max_difficulty):
            data = utils.load_manip_maps_for_difficulty(data_path=data_path, difficulty=difficulty+1)

            image_channels = np.copy(data[:, 2:4])
            path_histories = np.copy(data[:, 5])
            action_histories = np.copy(data[:, 6])

            if self.train:
                # First 4000 environments for training
                start_env = 0
                end_env = 3999
                # end_env = 1999
            else:
                # Last 1000 environments for testing
                start_env = 4000
                end_env = 4999

            # for env in range(image_channels.shape[0]):
            for env in range(start_env, end_env+1):
                image = np.copy(image_channels[env, :])
                path_history = np.copy(path_histories[env])
                action_history = np.copy(action_histories[env])

                for tstep in range(path_history.shape[0]-1):
                    s1_tstep = path_history[tstep][0]
                    s2_tstep = path_history[tstep][1]
                    s3_tstep = path_history[tstep][2]
                    action = action_history[tstep]

                    images.append(np.copy(image))
                    S1.append(s1_tstep)
                    S2.append(s2_tstep)
                    S3.append(s3_tstep)
                    labels.append(action)

        images = np.array(images)
        S1 = np.array(S1).astype(int)
        S2 = np.array(S2).astype(int)
        S3 = np.array(S3).astype(int)
        labels = np.array(labels).astype(int)

        return images, S1, S2, S3, labels


class GridworldDataManipRollout(object):
    def __init__(self, data_path, imsize, max_difficulty, train=False):
        self.data_path = data_path
        self.imsize = imsize
        self.max_difficulty = max_difficulty
        self.train = train

        self.images, self.start_positions, self.goal_positions, self.paths, self.rows, self.difficulties = self._process(data_path, max_difficulty)

        self.dims = 3
        self.lengths = [0.3] * self.dims
        self.dtheta = np.pi / 12
        self.num_obs = 4
        self.max_width = 0.5
        self.min_width = 0.3
        self.partial_obs = False
        self.num_per_obstacle_map = 10

        self.map_args = {
            'type': 'generator',
            'num_obs': self.num_obs,
            'max_width': self.max_width,
            'min_width': self.min_width,
            'difficulty': 0,
            'train': False
        }

    def __getitem__(self, index):
        img = np.copy(self.images[index])
        obs_grid = img[0]
        goal_channel = img[1]

        img = np.array([obs_grid, goal_channel], dtype=np.float32)
        start_pos = self.start_positions[index]
        goal_pos = self.goal_positions[index]
        path = self.paths[index]
        row = self.rows[index]
        difficulty = self.difficulties[index]

        img = torch.from_numpy(img)

        env = Manipulator(self.dims, self.dtheta, self.lengths, self.map_args, partial_obs=self.partial_obs, map_data=row)

        return img, start_pos, goal_pos, path, env, difficulty

    def __len__(self):
        return self.images.shape[0]

    def _process(self, data_path, max_difficulty):
        images = []
        start_positions = []
        goal_positions = []
        paths = []
        rows = []
        difficulties = []

        for difficulty in range(max_difficulty):
            data = utils.load_manip_maps_for_difficulty(data_path=data_path, difficulty=difficulty+1)

            image_channels = np.copy(data[:, 2:4])
            path_histories = np.copy(data[:, 5])
            # action_histories = np.copy(data[:, 6])
            start_pos = np.copy(data[:, 0])
            goal_pos = np.copy(data[:, 1])

            if self.train:
                # First 4000 environments for training
                start_env = 0
                end_env = 3999
            else:
                # Last 1000 environments for testing
                start_env = 4000
                # start_env = 9900
                end_env = 4999

            for env in range(start_env, end_env+1):
                image = np.copy(image_channels[env, :])
                path_history = np.copy(path_histories[env])
                # action_history = np.copy(action_histories[env])
                start = np.copy(start_pos[env])
                goal = np.copy(goal_pos[env])
                row = np.copy(data[env, :])

                images.append(image)
                start_positions.append(start)
                goal_positions.append(goal)
                paths.append(path_history)
                rows.append(row)
                difficulties.append(difficulty)

        return images, start_positions, goal_positions, paths, rows, difficulties
