import numpy as np
from collections import deque
from IPython import embed
from PIL import Image, ImageDraw
import math
import os
import pickle

def cartesian(arrays, out=None):
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

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def do_bounds_intersect(l1,h1,l2,h2,dims):

    intersect = 1

    for i in range(dims):
        if (l2[i] >= l1[i] and l2[i] <= h1[i]) or (h2[i] >= l1[i] and h2[i] <= h1[i]) :
            intersect = intersect*1
        else:
            intersect = intersect*0

    if intersect==1:
        return True
    else:
        return False

def is_point_in_bounds(pt,low,high,dims):


    for i in range(dims):
        if pt[i] < low[i] or pt[i] > high[i]:
            return False

    return True


def generate_obstacles(dim_size,dims,num_obs,max_width,start_coords=None,goal_coords=None):
    
    obstacles = list()
    idx = 0

    trial_factor = 5

    final_obs = 0

    for idx in range(trial_factor*num_obs):

        lower = [None] * dims
        higher = [None] * dims

        widths = np.random.random_integers(1,max_width+1,dims)

        # Generate lower and upper bounds subject to boundary constraints on width
        for i in range(dims):

            # Get a random width along dimension
            width = widths[i]

            # Sample lower bound between 1 and dim_size - width - 2
            l = np.random.random_integers(0,dim_size+1-width)
            h = l + width

            lower[i] = l
            higher[i] = h

        # Reject obstacles if they encircle start or goal
        if is_point_in_bounds(start_coords,lower,higher,dims) or is_point_in_bounds(goal_coords,lower,higher,dims):
            continue

        # Make sure new obstacle does not overlap previous one
        intersect = False
        for obs in obstacles:
            lo,ho = obs

            if do_bounds_intersect(lo,ho,lower,higher,dims) == True:
                intersect = True
                break

        if intersect == False:
            obstacles.append((lower,higher))
            final_obs = final_obs+1
            if final_obs == num_obs:
                break

    return obstacles


def bfs_till_depth(space_obj, start_coords, depth):
    '''
    Performs BFS from start_coords until depth 'depth'.
    Returns all nodes at that depth, and a parent lookup
    '''

    d = 0
    q1 = deque([start_coords])
    q2 = deque()
    curr_q = q1
    second_q = q2
    visited = {tuple(start_coords): True}
    parents = {tuple(start_coords): None}

    while True:
        # expand all nodes in curr_q, and add to second_q
        while len(curr_q) > 0:
            node = curr_q.pop()
            # add all neighbors of node if they have not been visited, and are free
            neighbors = space_obj.get_neighbors(node)
            for n in neighbors:
                # print n
                if tuple(n) not in visited:
                    visited[tuple(n)] = True
                    if not space_obj.check_collision(np.array(n)):
                        second_q.append(n)
                        parents[tuple(n)] = node

        # switch queues
        temp = curr_q
        curr_q = second_q
        second_q = temp

        d = d + 1
        if d == depth:
            return curr_q, parents

def get_random_node_at_depth(space_obj, start_coords, depth):
    '''
    Returns a random node at depth 'depth' from start_coords
    '''

    nodes, parents = bfs_till_depth(space_obj, start_coords, depth)
    if len(nodes) == 0:
        return None, None

    # randomly sample a node, and construct a path
    idx = np.random.randint(0, len(nodes))
    path = [nodes[idx]]
    curr_node = parents[tuple(nodes[idx])]
    while curr_node is not None:
        path.insert(0, curr_node)
        curr_node = parents[tuple(curr_node)]
    
    return nodes[idx], path

def load_manip_maps_for_difficulty(data_path, difficulty):
    return np.load(data_path + str(difficulty) + '.npy')

## -------------- MANIPULATOR ENV FUNCTIONS --------------------

## Functions for computing intersection
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1, a2, b1, b2) :
    # First line is a1 -> a2
    # Second line is b1 -> b2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot( dap, db)

    # If denom == 0, parallel
    if denom == 0:
        if (a1[0] > b1[0] and a1[0] < b2[0]) or (a2[0] > b1[0] and a2[0] < b2[0]):
            return True
        if (a1[1] > b1[1] and a1[1] < b2[1]) or (a2[1] > b1[1] and a2[1] < b2[1]):
            return True
        return False

    num = np.dot( dap, dp )
    int_pt =  (num / denom.astype(float))*db + b1

    # Segments intersect if int_pt is in-between both points
    mid_first = (a1+a2) / 2.0
    mid_second = (b1+b2) / 2.0

    in_first = np.linalg.norm(int_pt - mid_first) < np.linalg.norm(mid_first - a1)
    in_second = np.linalg.norm(int_pt - mid_second) < np.linalg.norm(mid_second - b1)

    return (in_first and in_second)
##


def get_manipulator_points(link_lengths, angles):
    # Assumes both arguments are numpy arrays
    # Computes forward kinematics
    # Returns 2 x (n+1) array with x-y positions of endpoints of manip links (including 0,0 for base)

    nlinks = len(link_lengths)
    assert nlinks == len(angles), "There must be as many links as angles!"

    angles_cumsum = np.cumsum(angles) # theta1 theta1 + theta2 theta1 + theta2 + theta3
    cosine_cumsum = np.cos(angles_cumsum) # c1 c12 c123 ...
    sine_cumsum = np.sin(angles_cumsum) # s1 s12 s123 ...

    x_positions = np.cumsum(np.multiply(link_lengths, cosine_cumsum)) # l1c1 l1c1 + l2c12  l1c1 + l2c12 + l3c123
    y_positions = np.cumsum(np.multiply(link_lengths, sine_cumsum))

    manip_pts = np.vstack((x_positions,y_positions))

    # Augment 0,0 at the beginning
    aug_manip_pts = np.concatenate( (np.array([[0.,0.]]).T , manip_pts), axis=1 )

    return aug_manip_pts


def manip_intersects_obstacle(obs,link_lengths,angles):

    # Checks if a single obstacle intersects manipulator
    # Used during env construction
    manip_points = get_manipulator_points(link_lengths,angles)
    _ , n_manip_pts = manip_points.shape

    _ , n_obs = obs.shape

    for i in range(n_obs):

        obs1 = obs[:,i]
        obs2 = obs[:,(i+1)%n_obs]

        for j in range(n_manip_pts):
            manip1 = manip_points[:,j]
            manip2 = manip_points[:,(j+1)%n_manip_pts]

            if seg_intersect(obs1,obs2,manip1,manip2) == True:
                return True

    return False

def is_manip_in_self_collision(link_lengths, angles):

    # Check if any of the angles after the zeroth are equal to +- pi
    # TODO : Is this correct check for self collision?

    any_pi = np.any(np.isclose(angles[1:], np.pi))
    any_minus_pi = np.any(np.isclose(angles[1:], -np.pi))

    # Also check if any of the manipulator points collide with each other
    manip_points = get_manipulator_points(link_lengths,angles)
    _ , n_manip_pts = manip_points.shape

    if any_pi or any_minus_pi:
        return True

    for j in range(1,n_manip_pts):
        manip0 = manip_points[:,j-1]
        manip1 = manip_points[:,j]
        manip2 = manip_points[:,(j+1)%n_manip_pts]

        if seg_intersect(manip0,manip1,manip1,manip2):
            return True

    return False



def is_manip_in_env_collision(link_lengths, angles, obstacles):

    # Assumes obstacles is list of 2 x n numpy array of vertices
    # DON'T call manip_intersects_obstacle here - for efficiency
    manip_points = get_manipulator_points(link_lengths,angles)
    _ , n_manip_pts = manip_points.shape

    for obs in obstacles:

        _ , n_obs = obs.shape

        for i in range(n_obs):

            obs1 = obs[:,i]
            obs2 = obs[:,(i+1)%n_obs]

            for j in range(n_manip_pts):
                manip1 = manip_points[:,j]
                manip2 = manip_points[:,(j+1)%n_manip_pts]

                if seg_intersect(obs1,obs2,manip1,manip2) == True:
                    return True

    return False


def generate_obstacles_continuous(num_obs, min_width, max_width, link_lengths, start_angles, goal_angles):

    '''
    Assumes 2D grid from [-1,-1] to [1,1] 
    Assumes robot is serial manipulator of zero thickness 
    and state is angle of each joint
    RETURNS list of 2 X n_i numpy arrays of upto num_obs obstacles
    that tries to avoid start and goal
    '''

    obstacles = list()
    numpy_obstacles = list()
    idx = 0

    trial_factor = 5
    dims = 2
    final_obs = 0

    for idx in range(trial_factor*num_obs):

        lower = np.zeros((dims,))
        higher = np.zeros((dims,))

        widths = min_width + (max_width - min_width)*np.random.rand(dims,)

        # Generate lower and upper bounds subject to boundary constraints on width
        for i in range(dims):

            # Get a random width along dimension
            width = widths[i]

            # Sample lower bound between 1 and dim_size - width - 2
            l = -1 + (2 - width)*np.random.rand()
            h = l + width

            lower[i] = l
            higher[i] = h

        # TODO : For now, just make sure manip does not intersect obstacle. Is this enough?
        # Numpy obstacle is 2 x n array of obstacle vertices
        x_pos = [lower[0],higher[0]]
        y_pos = [lower[1],higher[1]]
        
        # OBSTACLE GENERATED ANTI-CLOCKWISE
        temp_numpy_obs = np.array( [ [x_pos[0],y_pos[0]], [x_pos[1],y_pos[0]], [x_pos[1],y_pos[1]], [x_pos[0],y_pos[1]] ] )
        numpy_obs = temp_numpy_obs.T

        if manip_intersects_obstacle(numpy_obs,link_lengths,start_angles) or manip_intersects_obstacle(numpy_obs,link_lengths,goal_angles):
            continue

        #embed()
        # Make sure new obstacle does not overlap previous one
        intersect = False
        for obs in obstacles:
            lo,ho = obs

            if do_bounds_intersect(lo,ho,lower,higher,dims) == True:
                intersect = True
                break

        if intersect == False:
            # Append to two lists
            obstacles.append((lower,higher))            
            numpy_obstacles.append(numpy_obs)
            final_obs = final_obs + 1
            if final_obs == num_obs:
                break

    return numpy_obstacles

def get_r_c_from_x_y(x,y,size):
    # For size X size grid, convert x-y to r-c
    c = int((x+1)*size/2.0)
    r = int((1-y)*size/2.0)

    return r,c


def manipulator_simple_render(link_lengths, start_angles, goal_angles, angles, obstacles, size, im_name):
    '''
    grid_size : (square) image dimension length in pixels
    '''
    out_img = Image.new("RGB", (size,size))
    d = ImageDraw.Draw(out_img)

    pixels = [(255,255,255) for _ in range(size*size)]
    out_img.putdata(pixels)


    # Draw polygons for obstacles
    for obs in obstacles:

        _ , n_verts = obs.shape
        points = list()


        #embed()

        for j in range(n_verts):

            pr, pc = get_r_c_from_x_y(obs[0,j], obs[1,j], size)
            points.append((pc,pr))

        d.polygon(points, fill = (0,0,0), outline=(0,0,0))

    # Draw line for goal IF not equal to angle
    if np.any(np.isclose(angles,goal_angles)==False):
        manip_points = get_manipulator_points(link_lengths, goal_angles) # 2 X n
        _ , n_manip_pts = manip_points.shape

        for j in range(n_manip_pts-1):

            p1r, p1c = get_r_c_from_x_y(manip_points[0,j], manip_points[1,j], size)
            p2r, p2c = get_r_c_from_x_y(manip_points[0,(j+1)], manip_points[1,(j+1)], size)
            d.line([(p1c,p1r) , (p2c,p2r)], fill = (205,0,0), width = 5)

            # Draw circle for each endpoint
            d.ellipse([(p1c-2,p1r-2) , (p1c+2,p1r+2)],fill=(0,0,255))

    # Draw line for sstart IF not equal to angle
    if np.any(np.isclose(angles,start_angles)==False):
        manip_points = get_manipulator_points(link_lengths, start_angles) # 2 X n
        _ , n_manip_pts = manip_points.shape

        for j in range(n_manip_pts-1):

            p1r, p1c = get_r_c_from_x_y(manip_points[0,j], manip_points[1,j], size)
            p2r, p2c = get_r_c_from_x_y(manip_points[0,(j+1)], manip_points[1,(j+1)], size)
            d.line([(p1c,p1r) , (p2c,p2r)], fill = (205,0,205), width = 5)

            # Draw circle for each endpoint
            d.ellipse([(p1c-2,p1r-2) , (p1c+2,p1r+2)],fill=(0,0,255))


    # Draw angle no matter what
    manip_points = get_manipulator_points(link_lengths, angles) # 2 X n
    _ , n_manip_pts = manip_points.shape

    for j in range(n_manip_pts-1):

        p1r, p1c = get_r_c_from_x_y(manip_points[0,j], manip_points[1,j], size)
        p2r, p2c = get_r_c_from_x_y(manip_points[0,(j+1)], manip_points[1,(j+1)], size)
        d.line([(p1c,p1r) , (p2c,p2r)], fill = (0,205,0), width = 5)

        # Draw circle for each endpoint
        d.ellipse([(p1c-2,p1r-2) , (p1c+2,p1r+2)],fill=(0,0,255))

    


    if im_name == None:
        out_img.show()
    else:
        out_img.save(im_name)


def manipulator_render_video(link_lengths, start_coords, goal_coords, obstacles, size, coords_list, d_theta, video_name):
    '''
    Creates temp images of type image1, image2 etc
    Then creates video and GIF
    and then deletes images
    '''

    angles_list = [c*d_theta for c in coords_list]
    goal_angles = goal_coords * d_theta
    start_angles = start_coords * d_theta
    name_prefix = 'image00'

    im_name = name_prefix+str(1)+'.png'
    manipulator_simple_render(link_lengths,start_angles,goal_angles,start_angles,obstacles,size,im_name)

    idx = 1
    n_angles = len(angles_list)

    for i in range(1,n_angles):

        angles = angles_list[i]
        angles_show = np.copy(angles_list[i-1])

        
        # first angle change
        angles_show[0] = angles[0]
        idx = idx+1
        im_name = name_prefix+str(idx)+'.png'
        manipulator_simple_render(link_lengths,start_angles,goal_angles,angles_show,obstacles,size,im_name)
        

        # second angle change
        angles_show[1] = angles[1]
        idx = idx+1
        im_name = name_prefix+str(idx)+'.png'
        manipulator_simple_render(link_lengths,start_angles,goal_angles,angles_show,obstacles,size,im_name)
        

        # third angle change
        angles_show[2] = angles[2]
        idx = idx+1
        im_name = name_prefix+str(idx)+'.png'
        manipulator_simple_render(link_lengths,start_angles,goal_angles,angles_show,obstacles,size,im_name)
        

    # OS commands to generate GIF and delete image files
    os.system('ffmpeg -r 1 -f image2 -i image%03d.png '+video_name)
    os.system('rm image*')


def manipulator_render_video_from_pkl(picklefile,video_name):

    # Example call - utils.manipulator_render_video_from_pkl('success_1.pkl','test.gif')

    # FIXED PARAMS
    LINK_LENGTHS = np.array([0.3,0.3,0.3])
    D_THETA = np.pi/6
    SIZE = 1000

    # Load data from pickle file
    with open(picklefile,'rb') as f_in:
        data = pickle.load(f_in,encoding='latin1')

    # Assuming known format of pickle file
    manipulator_render_video(link_lengths = LINK_LENGTHS,
                            start_coords = data[0][0],
                            goal_coords = data[0][1], 
                            obstacles = data[0][4], 
                            size = SIZE, 
                            coords_list = data[0][5], 
                            d_theta = D_THETA, 
                            video_name = video_name)



def manipulator_render_path(link_lengths, obstacles, size, coords_list, d_theta):
    '''
    coords - LIST of np.arrays of coordinates along the path, including start and goal
    size - SCALAR size of image in pixels
    '''

    pathlength = len(coords_list)
    #angles_list = [coord_to_angle(c,d_theta) for c in coords_list]
    angles_list = [c*d_theta for c in coords_list]

    # Dark to light green
    colors = np.linspace(125,255,pathlength)


    # Set up image
    out_img = Image.new("RGB", (size,size))
    d = ImageDraw.Draw(out_img)

    pixels = [(255,255,255) for _ in range(size*size)]
    out_img.putdata(pixels)

    # Draw polygons for obstacles
    for obs in obstacles:

        _ , n_verts = obs.shape
        points = list()
        #embed()

        for j in range(n_verts):

            pr, pc = get_r_c_from_x_y(obs[0,j], obs[1,j], size)
            points.append((pc,pr))

        d.polygon(points, fill = (0,0,0), outline=(0,0,0))


    # Loop over and draw manipulators at various colors
    for i,angles in enumerate(angles_list):

        color = (0,int(colors[i]),0)

        # Render manipulator with color
        manip_points = get_manipulator_points(link_lengths, angles) # 2 X n
        _ , n_manip_pts = manip_points.shape

        for j in range(n_manip_pts-1):
            p1r, p1c = get_r_c_from_x_y(manip_points[0,j], manip_points[1,j], size)
            p2r, p2c = get_r_c_from_x_y(manip_points[0,(j+1)], manip_points[1,(j+1)], size)
            d.line([(p1c,p1r) , (p2c,p2r)], fill = color, width = 5)

            # Draw circle for each endpoint
            d.ellipse([(p1c-2,p1r-2) , (p1c+2,p1r+2)],fill=(0,0,0))

    out_img.show()


def angle_in_pi(angle):
    ''' Restricts 'angle' to lie between -pi and pi '''
    if angle > math.pi:
        angle -= 2*math.pi

    return angle

def coord_to_angle(coords, d_theta):
    ''' Converts coords to np array of angles in (-pi, pi) '''
    angles = coords * d_theta

    return map(angle_in_pi, angles)
