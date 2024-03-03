import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import yaml
import os
import pathlib
from a_star import A_star
from numpy import cos, sin, pi
import matplotlib.pyplot as plt


def load_map_and_metadata(map_file, only_borders=False):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    threshold = 219
    if only_borders:
        threshold = 0
    map_arr[map_arr <= threshold] = 1
    map_arr[map_arr > threshold] = 0
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
        
    return map_arr, map_hight, map_width, map_resolution, origin_x, origin_y


def pose2map_coordinates(map_resolution, origin_x, origin_y, x, y):
    x_map = int((x - origin_x) / map_resolution)
    y_map = int((y - origin_y) / map_resolution)
    return y_map, x_map

def map2pose_coordinates(map_resolution, origin_x, origin_y, x_map, y_map):
    x = x_map * map_resolution + origin_x
    y = y_map * map_resolution + origin_y
    return x, y


def collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, x, y):
    ####### your code goes here #######
    # TODO: transform configuration to workspace bounding box
    
    # TODO: overlay workspace bounding box on map (creating borders for collision search in the next step)
    
    # TODO: check for collisions by looking inside the bounding box on the map if there are values greater than 0    
    if map_arr[pose2map_coordinates(map_resolution, origin_x, origin_y, x, y)]:
        return True
    
    ##################################
    # raise NotImplementedError


def sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000, dim=2):
    ####### your code goes here #######
    rng = np.random.default_rng()
    top_x = np.where(map_arr == 0)[1].max()*map_resolution + origin_x
    top_y = np.where(map_arr == 0)[0].max()*map_resolution + origin_y
    bottom_x = np.where(map_arr == 0)[1].min()*map_resolution + origin_x
    bottom_y = np.where(map_arr == 0)[0].min()*map_resolution + origin_y
    points = rng.uniform(low=[bottom_x, bottom_y], high=[top_x, top_y], size=(n_points_to_sample, 2))
    pt = [tuple(point) for point in points]
    points = np.array(list(set(pt)))
    return points
    
    ##################################
    # raise NotImplementedError


def create_prm_traj(map_file):
    prm_traj = []
    mid_points = np.array([[0,0,0],
                           [9.5,4.5,pi/2],
                           [0,8.5,pi],
                           [-13.5,4.5,-pi/2]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)
    ####### your code goes here #######
    
    # TODO: create PRM graph
    prm_graph = {
        'nodes': [],
        'edges': [],
        'costs': {}
    }
    # TODO: sample configurations
    points = sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, 1000)
    # TODO: check for collisions
    for point in points:
        if not collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, *(point.tolist())):
            prm_graph['nodes'].append(tuple(point.tolist()))
    prm_graph['nodes'] = np.array(prm_graph['nodes'])
    # TODO: connect nodes using k-d tree
    kd_tree = KDTree(prm_graph['nodes'], copy_data=True)
    edgess = kd_tree.query_ball_point(kd_tree.data, 3)
    for i, edges in enumerate(edgess):
        for j in edges:
            if i != j:
                prm_graph['edges'].append((i,j))
                prm_graph['costs'][(i, j)] = np.linalg.norm(kd_tree.data[i] - kd_tree.data[j])
    # TODO: find the shortest path using A*
    A_star_obj = A_star(prm_graph)
    
    # TODO: create PRM trajectory (x,y) saving it to prm_traj list using a_star
    current_point = mid_points[0]
    for i in range(len(mid_points)):
        prm_traj.append(A_star_obj.a_star(current_point, mid_points[(i+1)%len(mid_points)]))
        current_point = prm_traj[-1][-1]
    
    print(prm_traj)
    
    ##################################
    
    prm_traj = np.concatenate(prm_traj, axis=0)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/prm_traj.npy'), prm_traj)


def sample_conrol_inputs(number_of_samples=10):
    ####### your code goes here #######
    
    ##################################
    raise NotImplementedError


def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt=0.5):
    ####### your code goes here #######
    
    
    ##################################
    raise NotImplementedError
    return x_new, y_new, theta_new


def create_kino_rrt_traj(map_file):
    kino_rrt_traj = []
    mid_points = np.array([[0,0],
                           [9.5,4.5],
                           [0,8.5],
                           [-13.5,4.5]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)
    ####### your code goes here #######
    
    # TODO: create RRT graph and find the path saving it to kino_rrt_traj list
    T = 0.7
    tree = {
        'nodes': [],
        'edges': [],
        'costs': {}
    }
    current_point = mid_points[0]
    for i in range(len(mid_points)):
        pass
        # TODO: create tree
            # TODO: sample configuration
            
            # TODO: check for collisions
            
            # TODO: find nearest neighbor
            
            # TODO: sample duration
            
            # TODO: sample control inputs
            
            # TODO: forward simulate
            
            # TODO: check for collisions on connected path
            
            # TODO: add to tree
            
        # TODO: return shortest path on it by using A*
        
    
    ##################################
    
    kino_rrt_traj = np.array(kino_rrt_traj, axis=0)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/kino_rrt_traj.npy'), kino_rrt_traj)


if __name__ == "__main__":
    map_file = 'maps/levine.png'
    # data = load_map_and_metadata(map_file)
    # print(data)
    # samples = sample_configuration(*data)
    # # points = (samples[:,:2] - np.array([data[4],data[5]]))
    # non_collision_points = []
    # for i, point in enumerate(samples):
    #     if not collision_check(*data, point[0], point[1], point[2]):
    #         non_collision_points.append(tuple(point))
    # non_collision_points = (np.array(non_collision_points)[:,:2] - np.array([data[4],data[5]]))//data[3]
    # print(non_collision_points)
    # points = (samples[:,:2] - np.array([data[4],data[5]]))//data[3]
    # plt.imshow(data[0])
    # # plt.scatter(points[:,0], points[:,1], c='b')
    # plt.scatter(non_collision_points[:,0], non_collision_points[:,1], c='r')
    # # plt.scatter(non_collision_points[:,0], non_collision_points[:,1])
    # plt.show()
    create_prm_traj(map_file)
    # create_kino_rrt_traj(map_file)
