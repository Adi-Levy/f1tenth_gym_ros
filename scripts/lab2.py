import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import yaml
import os
import pathlib
from a_star import A_star
import matplotlib.pyplot as plt


def load_map_and_metadata(map_file):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    map_arr[map_arr < 220] = 1
    map_arr[map_arr >= 220] = 0
    map_arr = map_arr.astype(bool)
    map_hight = map_arr.shape[0]
    map_width = map_arr.shape[1]
    # TODO: load the map dimentions and resolution from yaml file
    with open(map_file.replace('.png', '.yaml'), 'r') as f:
        try:
            map_metadata = yaml.safe_load(f)
            map_resolution = map_metadata['resolution']
            map_origin = map_metadata['origin']
        except yaml.YAMLError as exc:
            print(exc)
    origin_x = map_origin[0]
    origin_y = map_origin[1]

    print(map_arr)
    return map_arr, map_hight, map_width, map_resolution, origin_x, origin_y


def pose2map_coordinates(map_resolution, origin_x, origin_y, x, y):
    x_map = int((x - origin_x) / map_resolution)
    y_map = int((y - origin_y) / map_resolution)
    return y_map, x_map


def map2pose_coordinates(map_resolution, origin_x, origin_y, x_map, y_map):
    x = x_map * map_resolution + origin_x
    y = y_map * map_resolution + origin_y
    return x, y


def collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, x, y, theta):
    """
    Checks if a point (x,y,theta) is in free space
    """
    x, y = pose2map_coordinates(map_resolution, origin_x, origin_y, x, y)
    return map_arr[y, x]


def collision_path(map_arr, map_resolution, origin_x, origin_y, X, Y):
    """
    This function checks if the straight line between two points (with a 4-nearest-neighbour pad), X and Y collides with restricted areas of a map.
    params:
      map_arr: Boolean map, False in free areas
      X: 2D point coordinates (x, y), ints
      Y: 2D point coordinates (x, y), ints
    returns:
      collision_free: Boolean, is straight line between X, Y collision free
    """

    # convert to map frame
    x1, y1 = pose2map_coordinates(map_resolution, origin_x, origin_y, X[0], X[1])
    x2, y2 = pose2map_coordinates(map_resolution, origin_x, origin_y, Y[0], Y[1])

    # Calculate the difference between x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Determine the number of steps based on the maximum difference
    num_steps = max(abs(dx), abs(dy))
    if num_steps == 0:
        return True
    # Calculate the step size for x and y coordinates
    step_x = dx / num_steps
    step_y = dy / num_steps

    # Initialize the coordinates list with the starting point
    coordinates = [(x1, y1)]

    # Generate the coordinates for the line
    for i in range(1, num_steps + 1):
        x = x1 + i * step_x
        y = y1 + i * step_y
        coordinates.append((x, y))

    collision_free = True
    for x, y in coordinates:
        x_int = int(x)
        y_int = int(y)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i * j:
                    continue
                if map_arr[y_int + j][x_int + i]:
                    collision_free = False

    return collision_free

    ##################################


def sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000,
                         dim=2, rand=False):
    """
    This function samples n_points_to_sample points of the form (x,y[,theta]) (depending on dim)
    from the relevant configuration options of the given map_arr.
    """

    def trim_boolean_matrix(map_arr, padding=10):
        """
        This function takes a map and trims away (upto padding) the irrelevant areas we can never reach.
        Returns the boundaries.
        """
        matrix = ~map_arr
        rows_with_true, cols_with_true = np.any(matrix, axis=1), np.any(matrix, axis=0)
        min_col, max_col = np.argmax(rows_with_true), len(rows_with_true) - np.argmax(np.flip(rows_with_true, axis=0))
        min_row, max_row = np.argmax(cols_with_true), len(cols_with_true) - np.argmax(np.flip(cols_with_true, axis=0))
        return min_row - padding, max_row + padding, min_col - padding, max_col + padding

    def get_effective_n(n):
        while int(np.sqrt(n)) ** 2 < n:
            n += 1
        return n

    def get_sampled_coordinates(min_y, max_y, min_x, max_x, num_points):
        effective_num_points = get_effective_n(num_points)
        num_points_x = int(np.sqrt(effective_num_points))
        num_points_y = int(np.sqrt(effective_num_points))
        x = np.linspace(min_x, max_x, num_points_x)
        y = np.linspace(min_y, max_y, num_points_y)
        X, Y = np.meshgrid(x, y)
        coordinates = np.stack((X.flatten(), Y.flatten()), axis=-1)
        return coordinates[0:num_points, :]

    def get_rand_sampled_coordinates(min_y, max_y, min_x, max_x, num_points):
        return np.random.uniform((min_y, min_x), (max_y, max_x), (num_points, 2))

    if dim not in [2, 3]:
        raise Exception("Sorry, dim must be 2 or 3")
    min_row, max_row, min_col, max_col = trim_boolean_matrix(map_arr)
    if rand:
        coordinates = get_rand_sampled_coordinates(min_row, max_row, min_col, max_col, n_points_to_sample)
    else:
        coordinates = get_sampled_coordinates(min_row, max_row, min_col, max_col, n_points_to_sample)
    if dim == 3:
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=(n_points_to_sample, 1))
        coordinates = np.concatenate((coordinates, theta), axis=1)

    # converting to the configuration space:
    for idx, coor in enumerate(coordinates):
        coordinates[idx, 0:2] = map2pose_coordinates(map_resolution, origin_x, origin_y, coor[0], coor[1])

    return coordinates





def build_graph(free_coordinates, radius, map_arr, map_resolution, origin_x, origin_y):
    # graph = {
    #     'nodes': [node1, node2, ...],
    #     'edges': [(node1_index, node2_index), (node2_index, node3_index), ...],
    #     'costs': {(node1_index, node2_index): cost1, (node2_index, node3_index): cost2, ...}
    # node = (x, y, theta)
    graph = {'nodes': [], 'edges': [], 'costs': {}}
    tree_data = free_coordinates[:, :2]
    kdtree = KDTree(tree_data)
    neighbors = kdtree.query_ball_tree(kdtree, radius)
    for node1_idx, node1 in enumerate(tree_data):
        for node2_idx in neighbors[node1_idx]:
            if node1_idx == node2_idx:
                continue
            if collision_path(map_arr, map_resolution, origin_x, origin_y, node1, tree_data[node2_idx]):
                edge = (node1_idx, node2_idx)
                graph['edges'].append(edge)
                graph['costs'][edge] = np.linalg.norm(node1 - tree_data[node2_idx])
        check = np.array(
            [free_coordinates[node1_idx, 1], free_coordinates[node1_idx, 0], free_coordinates[node1_idx, 2]])
        graph['nodes'].append(check)
        # graph['nodes'].append(free_coordinates[node1_idx, :])

    graph['nodes'] = np.array(graph['nodes'])
    return graph


def create_prm_traj(map_file):
    prm_traj = np.array([[0, 0, 0]])
    radius = 2
    mid_points = np.array([[0, 0, 0],
                           [9.5, 4.5, np.pi / 2],
                           [0, 8.5, np.pi],
                           [-13.5, 4.5, -np.pi / 2]])

    path_list = map_file.split('/')
    map_file = os.path.join(path_list[1])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)

    ####### your code goes here #######
    coordinates = sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y,
                                       n_points_to_sample=10000, dim=3)
    coordinates = np.concatenate((coordinates, mid_points), axis=0)
    free_coordinates = np.array([coord for coord in coordinates if not
    collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, coord[0], coord[1], 0)])

    graph = build_graph(free_coordinates, radius, map_arr, map_resolution, origin_x, origin_y)
    optimiser = A_star(graph)
    for i in range(len(mid_points)):
        tmp = np.array(optimiser.a_star(mid_points[i], mid_points[(i + 1) % len(mid_points)]))
        if tmp.any():
            prm_traj = np.concatenate((prm_traj, tmp), axis=0)
    ##################################

    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(), './prm_traj.npy'),
            prm_traj[:, :2])


def sample_control_inputs(number_of_samples=10):
    ####### your code goes here #######
    # system parameters
    low_steering_angle = -0.3
    high_steering_angle = 0.3

    low_speed = 0.2
    high_speed = 0.7

    return np.random.uniform((low_speed, low_steering_angle), (high_speed, high_steering_angle),
                             (number_of_samples, 2)).squeeze()
    ##################################

def temps(prm_traj):
    ys = []
    xs = []
    for point in prm_traj:
        temp = pose2map_coordinates(0.05, -51.224998, -51.224998, point[0], point[1])
        ys.append(temp[0])
        xs.append(temp[1])
    return ys, xs

def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt=0.5):
    ####### your code goes here #######
    w = 0.3302  # wheelbase
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = (v / w) * np.tan(delta)

    x_new = x + x_dot * dt
    y_new = y + y_dot * dt
    theta_new = theta + theta_dot * dt
    ##################################
    return np.array((x_new, y_new, theta_new))


def back_prop_path(x_start, x_new, tree_dict, rrt_ref_traj):
    if x_new == x_start:
        return np.array(rrt_ref_traj)
    rrt_ref_traj.append(x_new)
    return back_prop_path(x_start, tree_dict[x_new], tree_dict, rrt_ref_traj)


def create_kino_rrt_traj(map_file):
    kino_rrt_traj = np.array([[0, 0, 0]])
    mid_points = np.array([[0, 0, 0],
                           [9.5, 4.5, np.pi / 2],
                           [0, 8.5, np.pi],
                           [-13.5, 4.5, -np.pi / 2]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)
    # *** #
    for i in range(len(mid_points)):
        x_start = mid_points[i]
        x_goal = mid_points[(i + 1) % len(mid_points)]
        eps = 1.5
        tree_data = x_start[np.newaxis]
        kdtree = KDTree(tree_data)
        x_last = np.array([np.inf, np.inf, np.inf])
        tree_dict = {tuple(x_start): tuple(x_start)}
        count = 0
        x_rands = []
        x_news = []
        while np.linalg.norm(x_last - x_goal) > eps:
            x_rand = sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y,
                                          n_points_to_sample=1, dim=3, rand=True).squeeze()
            x_rands.append(x_rand)
            x_near_idx = np.argmin(np.linalg.norm(tree_data - x_rand, axis=1))
            x_near = tree_data[x_near_idx]
            dt = np.random.uniform(2, 4)
            v, delta = sample_control_inputs(1)
            x_new = forward_simulation_of_kineamtic_model(x_near[0], x_near[1], x_near[2], v, delta, dt)
            x_news.append(x_new)
            x_near_check = x_near.copy()
            x_near_check[0] = x_near[1]
            x_near_check[1] = x_near[0]
            x_new_check = x_new.copy()
            x_new_check[0] = x_new[1]
            x_new_check[1] = x_new[0]
            if collision_path(map_arr, map_resolution, origin_x, origin_y, x_near_check, x_new_check):
                tree_data = np.concatenate((tree_data, x_new[np.newaxis]), axis=0)
                tree_dict[tuple(x_new.round(3))] = tuple(x_near.round(3))
                x_last = x_new
                # if np.linalg.norm(x_last - x_goal) < eps:
                #     print('finished')
            count +=1
            if not count%1000:
                print()
        kino_rrt_traj = np.concatenate((kino_rrt_traj, back_prop_path(tuple(x_start.round(3)), tuple(x_last.round(3)), tree_dict, [])), axis=0)

    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(), './kino_rrt_traj.npy'),
            kino_rrt_traj)


if __name__ == "__main__":
    map_file = './levine.png'
    # map_file = r'mobile_robots\Lab\f1tenth_gym_ros\maps\levine.png'
    create_prm_traj(map_file)
    create_kino_rrt_traj(map_file)
