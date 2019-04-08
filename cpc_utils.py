import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

class Particle2d(object):
    # 2D particle with horizontal / vertical obstacles obstacle
    def __init__(self, dx=0.1, xmin=-1, xmax=1, x0=0.0, dy=0.1, ymin=-1, ymax=1, y0=0.0):
        self.horizontal_obs = []
        self.vertical_obs = []
        self.dead_zones = []
        self.forbidden_transitions = []
        self.borders = [(xmin, ymin, xmax, ymax)]

        self.dx = dx
        self.xmin = xmin
        self.xmax = xmax
        self.x0 = x0
        self.x = self.x0
        self.dy = dy
        self.ymin = ymin
        self.ymax = ymax
        self.y0 = y0
        self.y = self.y0
        return

    @property
    def get_obstacles(self):
        # x1, y1, x2, y2
        return np.reshape(self.horizontal_obs + self.vertical_obs, (-1, 4))

    def add_horizontal_obstacle(self, x0, x1, y):
        self.horizontal_obs.append([[x0, y], [x1, y]])

    def add_vertical_obstacle(self, y0, y1, x):
        self.vertical_obs.append([[x, y0], [x, y1]])

    def check_horizontal_collision(self, point0, point1, obs):
        if point0[1] >= obs[0][1] and point1[1] >= obs[0][1]:
            return False
        if point0[1] <= obs[0][1] and point1[1] <= obs[0][1]:
            return False
        if point1[0] == point0[0]:
            if point0[0] > np.maximum(obs[0][0], obs[1][0]):
                return False
            if point0[0] < np.minimum(obs[0][0], obs[1][0]):
                return False
            return True
        slope = (point1[1] - point0[1]) / (point1[0] - point0[0])
        # line: y = point0[1] + slope * (x - point0[0])
        x_col = ((obs[0][1] - point0[1]) / slope) + point0[0]
        if x_col > np.maximum(obs[0][0], obs[1][0]):
            return False
        if x_col < np.minimum(obs[0][0], obs[1][0]):
            return False
        return True

    def check_vertical_collision(self, point0, point1, obs):
        if point0[0] >= obs[0][0] and point1[0] >= obs[0][0]:
            return False
        if point0[0] <= obs[0][0] and point1[0] <= obs[0][0]:
            return False
        slope = (point1[1] - point0[1]) / (point1[0] - point0[0])
        # line: y = point0[1] + slope * (x - point0[0])
        y_col = point0[1] + slope * (obs[0][0] - point0[0])
        if y_col > np.maximum(obs[0][1], obs[1][1]):
            return False
        if y_col < np.minimum(obs[0][1], obs[1][1]):
            return False
        return True

    def check_collisions(self, point0, point1):
        for obs in self.horizontal_obs:
            if self.check_horizontal_collision(point0, point1, obs):
                return True
        for obs in self.vertical_obs:
            if self.check_vertical_collision(point0, point1, obs):
                return True
        if point1[0] > self.xmax or point1[0] < self.xmin:
            return True
        if point1[1] > self.ymax or point1[1] < self.ymin:
            return True
        return False

    def reset(self):
        self.x = self.x0
        self.y = self.y0

    def step(self):
        collision = True
        count = 0
        while collision:
            x = self.x
            y = self.y
            d = np.random.uniform()
            x += d * self.dx if np.random.rand() < 0.5 else -d * self.dx
            d = np.random.uniform()
            y += d * self.dy if np.random.rand() < 0.5 else -d * self.dy
            collision = self.check_collisions([self.x, self.y], [x, y])
            count += 1
            assert (count < 1000)
        self.x = x
        self.y = y
        return self.x, self.y

    def forbid(self, x0, y0, x1, y1):
        assert x0 <= x1 and y0 < y1
        self.dead_zones.append([x0, y0, x1, y1])

    def forbid_transition(self, criteria):
        """criteria: A function that returns True if the transition is forbidden, False if allowed."""
        self.forbidden_transitions.append(criteria)

    def feasible(self, *path):
        if not path:
            return None
        for i, (x, y) in enumerate(path):
            for x0, y0, x1, y1 in self.dead_zones:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return False
            for x0, y0, x1, y1 in self.borders:
                if x0 >= x or x >= x1 and y0 >= y or y >= y1:
                    return False
            try:
                x1, y1 = path[i + 1]
                for fn in self.forbidden_transitions:
                    if fn(x, y, x1, y1):
                        return False
            except IndexError:
                pass
        return True

class Particle2dTunnel(Particle2d):
    '''
    2D particle moving with supervision.
    '''

    def __init__(self, dx=0.1, xmin=-1, xmax=1, x0=(-0.9, -0.9), dy=0.1, ymin=-1, ymax=1, y0=(0.9, -0.9), ministeps=1):
        self.inits = list(zip(x0, y0))
        self.reset()
        self.ministeps = ministeps
        super(Particle2dTunnel, self).__init__(dx, xmin, xmax, x0, dy, ymin, ymax, y0)

    def ministep(self, heading=None, noise=0.01):
        collision = True
        count = 0
        s = 1.0
        while collision:
            x = self.x
            y = self.y
            if heading is None:
                d = np.random.uniform()
                x += d * self.dx if np.random.rand() < 0.5 else -d * self.dx
                d = np.random.uniform()
                y += d * self.dy if np.random.rand() < 0.5 else -d * self.dy
            else:
                assert len(heading) == 2
                _dx = (heading[0] - x) + np.random.randn() * noise
                _dy = (heading[1] - y) + np.random.randn() * noise
                _norm = np.sqrt(_dx ** 2 + _dy ** 2)
                x += s * _dx / _norm * self.dx
                y += s * _dy / _norm * self.dy
            collision = self.check_collisions([self.x, self.y], [x, y])
            count += 1
            s /= 1.5
            assert (count < 1000)
        self.x = x
        self.y = y
        return self.x, self.y

    def step(self, heading=None, noise=0.01):
        for i in range(self.ministeps):
            self.ministep(heading, noise)
        return self.x, self.y

    def reset(self):
        x0, y0 = self.inits[np.random.choice(len(self.inits))]
        self.x = x0
        self.y = y0

class Particle2dKey(Particle2d):
    '''
    2D particle moving in a domain with obstacles, a key and a door.
    '''

    def __init__(self, dx=0.1, xmin=-1, xmax=1, x0=0.0, dy=0.1, ymin=-1, ymax=1, y0=0.0,
                 key_x=0.9, key_y=0.9, key_r=0.1):
        super(Particle2dKey, self).__init__(dx, xmin, xmax, x0, dy, ymin, ymax, y0)
        self.key_loc = np.array([key_x, key_y])
        self.key_rad = key_r
        self.has_key = False
        self.horizontal_doors = []
        self.vertical_doors = []

        # Add bouding box
        self.add_horizontal_obstacle(x0=xmin, x1=xmax, y=ymin)
        self.add_horizontal_obstacle(x0=xmin, x1=xmax, y=ymax)
        self.add_vertical_obstacle(y0=ymin, y1=ymax, x=xmin)
        self.add_vertical_obstacle(y0=ymin, y1=ymax, x=xmax)
        return

    # def check_reach_key(self):
    #     if np.linalg.norm(np.array([self.x, self.y]) - self.key_loc) < self.key_rad:
    #         return True
    #     return False
    def check_reach_key(self, x=None, y=None, feather=1.0):
        """

        :param x:
        :param y:
        :param feather: the fuzzy radius for checking the key, default is 1.0.
        :return:
        """
        assert feather >= 1.0, "feather radius has to be greater than 1.0"
        if x is None and y is None:
            x, y = self.x, self.y
        pos = np.array([x, y])
        if np.linalg.norm(pos - self.key_loc) < self.key_rad * feather:
            return True
        return False

    @property
    def get_key(self):
        return self.key_loc[0], self.key_loc[1], self.key_rad

    @property
    def get_doors(self):
        # x1, y1, x2, y2
        return np.reshape(self.horizontal_doors + self.vertical_doors, (-1, 4))

    def add_horizontal_door(self, x0, x1, y):
        door = [[x0, y], [x1, y]]
        self.horizontal_obs.append(door)
        self.horizontal_doors.append(door)

    def add_vertical_door(self, y0, y1, x):
        door = [[x, y0], [x, y1]]
        self.vertical_obs.append(door)
        self.vertical_doors.append(door)

    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.has_key = False
        for door in self.horizontal_doors:
            if door not in self.horizontal_obs:
                self.horizontal_obs.append(door)
        for door in self.vertical_doors:
            if door not in self.vertical_obs:
                self.vertical_obs.append(door)

    def step(self):
        collision = True
        count = 0
        while collision:
            x = self.x
            y = self.y
            d = np.random.uniform()
            x += d * self.dx if np.random.rand() < 0.5 else -d * self.dx
            d = np.random.uniform()
            y += d * self.dy if np.random.rand() < 0.5 else -d * self.dy
            collision = self.check_collisions([self.x, self.y], [x, y])
            count += 1
            assert (count < 1000)
        self.x = x
        self.y = y
        if self.check_reach_key() and not self.has_key:
            # If reach here, for the first time. The doors become open.
            for door in self.horizontal_doors:
                self.horizontal_obs.remove(door)
            for door in self.vertical_doors:
                self.vertical_obs.remove(door)
            self.has_key = True
        return self.x, self.y, float(self.has_key)

    def feasible(self, *path, verbose=False, feather_scale=2.0):
        # note: hard-coded radius for detecting the key
        p = np.array(path)
        # Only pass through the x, y position to tunnel domain.
        if not super(Particle2dKey, self).feasible(*p[:, :2]):
            return False

        # note: hard-coded single key and single door
        # todo: make more general, to have more doors
        *_, door_y = self.horizontal_doors[0][0]
        for i, (x, y, has_key) in enumerate(path):
            try:
                x1, y1, has_key1 = path[i + 1]
                # Can NOT transition from no key to has key
                if not has_key and has_key1 and not self.check_reach_key(x1, y1, feather_scale):
                    if verbose:
                        print(f"Can NOT transition from no key to has key {(x, y)} => {(x1, y1)}")
                    return False
                # passage without key from lower half to higher half is forbidden
                if (not has_key) and y < door_y < y1:
                    if verbose:
                        print('passage without key from lower half to higher half is forbidden')
                    return False
                # passage from top to bottom but bottom has no key is wrong.
                if (not has_key1) and y > door_y > y1:
                    if verbose:
                        print('passage from top to bottom but bottom has no key is wrong.')
                    return False
                # starting from bottom without key then have key is wrong
                if (not has_key) and y < door_y and has_key1:
                    if verbose:
                        print('starting from bottom without key then have key is wrong')
                    return False
                # when y is close to the border yet it crossed, this is infeasible.
                if -0.1 < y < 0.2 and (not has_key) and y1 < -0.1:
                    if verbose:
                        print('when y is close to the border yet it crossed, this is infeasible.')
                    return False
            except IndexError:
                pass
        return True

def get_map(name, **kwargs):
    if name == 'block':
        """
        One big block obstacle.
        """
        params = dict(dx=0.30, dy=0.30)
        params.update(kwargs)
        map2d = Particle2d(**params)
        map2d.add_horizontal_obstacle(x0=-0.8, x1=0.8, y=-0.1)
        map2d.add_horizontal_obstacle(x0=-0.8, x1=0.8, y=-0.4)
        map2d.add_vertical_obstacle(y0=-0.1, y1=-0.4, x=0.8)
        map2d.add_vertical_obstacle(y0=-0.1, y1=-0.4, x=-0.8)
        map2d.forbid(-0.8, -0.4, 0.8, -0.1)
    elif name == 'line':
        """
        Two line obstacles with a narrow opening through.
        """
        params = dict(dx=0.50, dy=0.50)
        params.update(kwargs)
        map2d = Particle2d(**params)
        map2d.add_horizontal_obstacle(x0=-1.0, x1=-.2, y=-0.1)
        map2d.add_horizontal_obstacle(x0=.2, x1=1.0, y=-0.1)
        map2d.forbid(-0.8, -0.4, 0.8, -0.1)
    elif 'tunnel' in name:
        """
        Small tunnels connecting room 1 to room 2, and room 3 to room 4.
        The initialization position is biased around the tunnel area. To override these initial positions, 
        set the x0 and y0 to None and None. This will allow the reset function to generate_pairs a uniform 
        distribution in the entire map.
        """
        params = dict(x0=(-0.3, -0.3, 0.3, 0.3), y0=(-0.15, 0.05, -0.15, 0.05), dx=0.05, dy=0.05, ministeps=5)
        params.update(kwargs)
        map2d = Particle2dTunnel(**params)

        # Room separation
        map2d.add_horizontal_obstacle(x0=-1.0, x1=1.0, y=-0.1)
        map2d.forbid_transition(lambda x0, y0, x1, y1: (y0 + 0.1) * (y1 + 0.1) <= 0)

        # Obstacles
        map2d.add_horizontal_obstacle(x0=-0.2, x1=0.2, y=0.0)
        map2d.add_vertical_obstacle(y0=0.0, y1=1.0, x=-0.2)
        map2d.add_vertical_obstacle(y0=0.0, y1=1.0, x=0.2)
        map2d.forbid(-0.2, 0, 0.2, 1)
        map2d.add_horizontal_obstacle(x0=-0.2, x1=0.2, y=-0.2)
        map2d.add_vertical_obstacle(y0=-0.2, y1=-1.0, x=-0.2)
        map2d.add_vertical_obstacle(y0=-0.2, y1=-1.0, x=0.2)
        map2d.forbid(-0.2, -1, 0.2, -0.2)
    elif name == 'door':
        """
        Need to get the key to open the door and go to the next room.
        todo: add door logic
        """
        params = dict(dx=0.30, dy=0.30)
        params.update(kwargs)
        map2d = Particle2dKey(**params)
        map2d.add_horizontal_door(x0=-0.2, x1=0.2, y=-0.1)
        map2d.add_horizontal_obstacle(x0=-1.0, x1=-.2, y=-0.1)
        map2d.add_horizontal_obstacle(x0=.2, x1=1.0, y=-0.1)
    elif name == 'no':
        params = dict(x0=(-0.3, -0.3, 0.3, 0.3), y0=(-0.15, 0.05, -0.15, 0.05), dx=0.05, dy=0.05, ministeps=5)
        params.update(kwargs)
        map2d = Particle2dTunnel(**params)
    else:
        raise ValueError('name should be one of ["line", "block", "*tunnel", "door"]')
    return map2d

def plot_data_density(data):
    from matplotlib import pyplot as plt
    x = data[:, 0]
    y = data[:, 1]
    xy = np.vstack([x, y])
    # z = gaussian_kde(xy)(xy)
    plt.gca().scatter(x, y, c='#23aaff', alpha=0.1, s=100, edgecolor='')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

def plot_map(map2d, key_color='red', key_alpha=0.9):
    from matplotlib import pyplot as plt
    plt.xlim((-1.2, 1.2))
    plt.ylim((-1.2, 1.2))
    if 'Key' in map2d.__class__.__name__:
        obstacles = map2d.get_obstacles
        doors = map2d.get_doors
        key_x, key_y, key_size = map2d.get_key
        for obstacle in obstacles:
            plt.plot(obstacle[::2], obstacle[1::2], c='black')
        for door in doors:
            plt.plot(door[::2], door[1::2], c='yellow')
        key_draw = plt.Circle((key_x, key_y), key_size, edgecolor=key_color, facecolor="none", linewidth=4,
                              alpha=key_alpha,
                              linestyle="-", label="key")
        plt.gca().add_artist(key_draw)
    else:
        obstacles = map2d.get_obstacles
        for obstacle in obstacles:
            plt.plot(obstacle[::2], obstacle[1::2], c='black')

def plot_clusters(clust_idx_map,
                  c_dim,
                  map2d):
    cmap = matplotlib.cm.get_cmap('hsv')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=c_dim)
    fig = plt.figure()
    plt.imshow(clust_idx_map, cmap=cmap, norm=norm, origin='lower', extent=[-1., 1., -1., 1.])
    plot_map(map2d)
    return fig

def binary_to_int(x, width):
    return x.dot(2 ** np.arange(width)[::-1])

def stochastic_binary_layer(x, tau=1.0):
    """
    x is (batch size)x(N) input of binary logits. Output is stochastic sigmoid, applied element-wise to the logits.
    """
    orig_shape = list(x.size())
    x0 = torch.zeros_like(x)
    x = torch.stack([x, x0], dim=2)
    x_flat = x.view(-1, 2)
    out = F.gumbel_softmax(x_flat, tau=tau, hard=True)[:, 0]
    return out.view(orig_shape)

def onehot_to_int(x):
    return np.where(x==1)[1]
