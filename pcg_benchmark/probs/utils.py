import numpy as np

"""
Get all the locations of a specific tile value in the 2D array

Parameters:
    map (int[][]): is a numpy 2D array that need to be searched
    tile_values (int[]): is an array of all the possible values that need to be discovered in the input map

Returns:
    [int,int][]: an array of (x,y) location in the input array that is equal to the values in tile_values
"""
def _get_certain_tiles(map, tile_values):
    tiles=[]
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if map[y][x] in tile_values:
                tiles.append((x,y))
    return tiles

"""
Run Dijkstra Algorithm and return the map and the location that are visited

Parameters:
    x(int): the x position of the starting point of the dijkstra algorithm
    y(int): the y position of the starting point of the dijkstra algorithm
    map(int[][]): the input 2D map that need to be tested for dijkstra
    passable_values(int[]): the values that are considered passable

Returns:
    int[][]: the dijkstra map where the value is the distance towards x, y location
    int[][]: a binary 2D array where 1 means visited by Dijkstra algorithm and 0 means not
"""
def _run_dikjstra(x, y, map, passable_values):
    dikjstra_map = np.full((map.shape[0], map.shape[1]),-1)
    visited_map = np.zeros((map.shape[0], map.shape[1]))
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx,cy,cd) = queue.pop(0)
        if map[cy][cx] not in passable_values or (dikjstra_map[cy][cx] >= 0 and dikjstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dikjstra_map[cy][cx] = cd
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny, cd + 1))
    return dikjstra_map, visited_map

"""
Get an array of positions that leads to the starting of Dijkstra

Parameters:
    dijkstra(int[][]): the dijkstra map that need to be tested
    sx(int): the x position to get path from
    sy(int): the y position to get path from

Returns:
    [int,int][]: an array of all the positions that lead from starting position to (sx,sy)
"""
def _get_path(dikjsta, sx, sy):
    path = []
    cx,cy = sx,sy
    while dikjsta[cy][cx] > 0:
        path.append((cx,cy))
        minx, miny = cx, cy
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(dikjsta[0]) or ny >= len(dikjsta) or dikjsta[ny][nx] < 0:
                continue
            if dikjsta[ny][nx] < dikjsta[miny][minx]:
                minx, miny = nx, ny
        if minx == cx and miny == cy:
            break
        cx, cy = minx, miny
    if len(path) > 0:
        path.append((cx, cy))
    path.reverse()
    return path

"""
Get the number of tiles that the flood fill algorithm fill

Parameters:
    x(int): the x position for the flood fill algorithm
    y(int): the y position for the flood fill algorithm
    color_map(int[][]): the color map where the test happen on map and is added to this variable
    map(int[][]): the maze that need to be flood filled (it doesn't change)
    color_index(int): the color that is used to color the region
    passable_Values(int[]): the tiles that are considered the same when are near each other

Returns:
    int: number of tiles for the flood fill
"""
def _flood_fill(x, y, color_map, map, color_index, passable_values):
    num_tiles = 0
    queue = [(x, y)]
    while len(queue) > 0:
        (cx, cy) = queue.pop(0)
        if color_map[cy][cx] != -1 or map[cy][cx] not in passable_values:
            continue
        num_tiles += 1
        color_map[cy][cx] = color_index
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny))
    return num_tiles

"""
Calculate the number of regions in the map that have the same values in the tile_values

Parameters:
    maze(int[][]):  the maze that need to be checked for regions
    tile_values(int[]): the values that need to be checked making regions

Returns:
    Number of seperate regions in the maze that have the same values using 1 size Von Neumann neighborhood
"""
def get_number_regions(maze, tile_values):
    empty_tiles = _get_certain_tiles(maze, tile_values)
    region_index=0
    color_map = np.full((maze.shape[0], maze.shape[1]), -1)
    for (x,y) in empty_tiles:
        num_tiles = _flood_fill(x, y, color_map, maze, region_index + 1, tile_values)
        if num_tiles > 0:
            region_index += 1
        else:
            continue
    return region_index

"""
Calculate the size of the regions in the maze specified by the locations that have the same values in the tile_values

Parameters:
    maze(int[][]): the maze that need to be tested for regions
    locations([int,int][]): an array of x,y locations that specify the starting point of the regions
    tile_values(int[]): the values that are considered the same in the regions
    
Returns:
    int[]: an array of the size of the regions that have the starting points in the locations
"""
def get_regions_size(maze, locations, tile_values):
    result = []
    region_index=0
    color_map = np.full((maze.shape[0], maze.shape[1]), -1)
    for (x,y) in locations:
        num_tiles = _flood_fill(x, y, color_map, maze, region_index + 1, tile_values)
        if num_tiles > 0:
            region_index += 1
            result.append(num_tiles)
        else:
            continue
    return result

"""
Get the longest shortest path in a maze. This is calculated by first calculating the Dijstra for all the tiles values.
Then Picking the highest value in the map and run Dijkstra again and get the maximum value

Parameters:
    maze(int[][]): the maze that need to be tested for the longest shortest path
    tile_values(int[]): the values that are passable in the maze

Returns:
    int: the longest shortest distance in a maze
"""
def get_longest_path(maze, tile_values):
    empty_tiles = _get_certain_tiles(maze, tile_values)
    final_visited_map = np.zeros((maze.shape[0], maze.shape[1]))
    final_value = 0
    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dikjstra_map, visited_map = _run_dikjstra(x, y, maze, tile_values)
        final_visited_map += visited_map
        (my,mx) = np.unravel_index(np.argmax(dikjstra_map, axis=None), dikjstra_map.shape)
        dikjstra_map, _ = _run_dikjstra(mx, my, maze, tile_values)
        max_value = np.max(dikjstra_map)
        if max_value > final_value:
            final_value = max_value
    return final_value

"""
Get the distance between two points in a maze

Parameters:
    maze(int[][]): the maze that need to be tested for distance
    start_tile([int,int]): the starting x,y position for the distance metric
    end_tile([int,int]): the ending x,y positon for the distance metric
    passable_tiles(int[]): the passable tiles in the maze

Returns:
    int: the distance between the starting tile and ending tile in the maze
"""
def get_distance_length(maze, start_tile, end_tile, passable_tiles):
    start_tiles = _get_certain_tiles(maze, [start_tile])
    end_tiles = _get_certain_tiles(maze, [end_tile])
    if len(start_tiles) == 0 or len(end_tiles) == 0:
        return -1
    (sx,sy) = start_tiles[0]
    (ex,ey) = end_tiles[0]
    dikjstra_map, _ = _run_dikjstra(sx, sy, maze, passable_tiles)
    return dikjstra_map[ey][ex]

"""
Get a path between two position as (x,y) locations

Parameters:
    maze(int[][]): the maze that need to check for path in it
    start_tile([int,int]): x,y for the starting tile for path finding
    end_tile([int,int]): x,y for the ending tile for path finding
    passable_tiles(int[]): the passable tiles in the maze

Returns:
    [int,int][]: an array of x,y corridnates that connected between start_tile and end_tile
"""
def get_path(maze, start_tile, end_tile, passable_tiles):
    maze = np.array(maze)
    start_tiles = _get_certain_tiles(maze, [start_tile])
    end_tiles = _get_certain_tiles(maze, [end_tile])
    if len(start_tiles) == 0 or len(end_tiles) == 0:
        return []
    (sx,sy) = start_tiles[0]
    (ex,ey) = end_tiles[0]
    dikjstra_map, _ = _run_dikjstra(sx, sy, maze, passable_tiles)
    return _get_path(dikjstra_map, ex, ey)

"""
Calculate horizontal symmetric tiles

Parameters:
    maze(int[][]): the maze that need to be tested for symmetry

Returns:
    int: get the number of tiles that are symmetric horizontally
"""
def get_horz_symmetry(maze):
    symmetry = 0
    for i in range(maze.shape[0]):
        for j in range(int(maze.shape[1]/2)):
            if maze[i][j] == maze[i][-j-1]:
                symmetry += 1
    return symmetry

"""
Get the input map modified using all the rotations and flipping

Parameters:
    map(int[][]): the input map that need to be transformed

Returns:
    int[][][]: all the possible transformed maps (rotate,flipping)
"""
def get_all_transforms(map):
    map = np.array(map)
    results = []
    for invH in [False, True]:
        for invV in [False, True]:
            for rot in [False, True]:
                newMap = np.zeros(map.shape)
                for y in range(len(newMap)):
                    for x in range(len(newMap[y])):
                        nx,ny = x,y
                        if invH:
                            nx = newMap.shape[1] - nx - 1
                        if invV:
                            ny = newMap.shape[0] - ny - 1
                        if rot and map.shape[0] == map.shape[1]:
                            temp = nx
                            nx = ny
                            ny = temp
                        newMap[ny][nx] = map[y][x]
                results.append(newMap)
    return results

"""
Get number of tiles in a maze that are equal to tile_values

Parameters:
    maze(int[][]): the 2d maze that need to be tested
    tile_values(int[]): the values that needs to be counted in the maze

Returns:
    int: return the number of tiles in the map of these values
"""
def get_num_tiles(maze, tile_values):
    return len(_get_certain_tiles(maze, tile_values))

"""
Get a histogram of horizontally connected region lengths

Parameters:
    maze(int[][]): the maze that need to be tested
    tile_values(int[]): the values that create groups

Returns:
    int[]: a histogram of length for the horizontal groups of the same value
"""
def get_horz_histogram(maze, tile_values):
    histogram = np.zeros(maze.shape[1])
    for i in range(maze.shape[0]):
        start_index = -1
        for j in range(maze.shape[1]):
            if maze[i][j] in tile_values:
                if start_index < 0:
                    start_index = j
            else:
                if start_index >= 0:
                    histogram[j - start_index] += 1
    return histogram

"""
Get a histogram of vertically connected region lengths

Parameters:
    maze(int[][]): the maze that need to be tested
    tile_values(int[]): the values that create groups

Returns:
    int[]: a histogram of length for the vertical groups of the same value
"""
def get_vert_histogram(maze, tile_values):
    histogram = np.zeros(maze.shape[0])
    for j in range(maze.shape[1]):
        start_index = -1
        for i in range(maze.shape[0]):
            if maze[i][j] in tile_values:
                if start_index < 0:
                    start_index = i
            else:
                if start_index >= 0:
                    histogram[i - start_index] += 1
    return histogram

"""
Get a bin number for a float value between 0 and 1

Parameters:
    value(float): a float value that needs to be discretized between 0 and bins
    bins(int): number of bins that are possible

Returns:
    int: a value between 0 and bins that reflect which discrete value works
"""
def discretize(value, bins):
    return int(bins * np.clip(value,0,1)-0.00000001)

"""
Reshape the reward value to be between 0 and 1 based on a trapizoid

Parameters:
    value(float): the value that need to be reshaped between 0 and 1
    min_value(float): the minimum value where the reward is 0 at it
    plat_low(float): the minimum value where the reward become 1
    plat_high(float): the maximum value where the rewards stays 1 at. Optional parameter where not provided 
    equal to plat_low
    max_value(float): the maximum value where the rewrad is back at 0. Optional parameter where not provided
    it will be equal to plat_high

Returns:
    float: a value between 0 and 1 based on where it falls in the trapizoid reward scheme
"""
def get_range_reward(value, min_value, plat_low, plat_high = None, max_value = None):
    if max_value == None:
        max_value = plat_high
    if plat_high == None:
        plat_high = plat_low
        max_value = plat_low
    if value >= plat_low and value <= plat_high:
        return 1.0
    if value <= min_value or value >= max_value:
        return 0.0
    if value < plat_low:
        return np.clip((value - min_value) / (plat_low - min_value + 0.00000001), 0.0, 1.0)
    if value > plat_high:
        return np.clip((max_value - value) / (max_value - plat_high + 0.00000001), 0.0, 1.0)

"""
Normalize a value between low and high

Parameters:
    value(float): the value need to be normalized
    low(float): the lowest value for the normalization
    high(float): the highest value for the normalization

Returns:
    float: the normalized value between 0 and 1
"""
def get_normalized_value(value, low, high):
    return (value - low) / (high - low + 0.00000000000001)


# import sys
# import random
# import numpy as np
# import torch
# import logging
# from logging.handlers import RotatingFileHandler
# from typing import Optional
# from pathlib import Path
# import colorlog
# import torch.nn.functional as F
# from gym_pcgrl.envs.helper import get_int_prob, get_string_map
# import matplotlib.pyplot as plt
# from gym_pcgrl.envs.helper import gen_random_map


# class ConstructionNetworkWrapper:
#     def __init__(self, network):
#         self.policy = self
#         self.network = network

#     def transform(self, obs, x, y, obs_size):
#         map = obs
#         size = obs_size[0]
#         pad = obs_size[0] // 2
#         padded = np.pad(map, pad, constant_values=1)
#         cropped = padded[y: y + size, x: x + size]
        
#         return cropped
        
#     def predict(self, state, x, y):
#         state = F.one_hot(torch.tensor(self.transform(state, x, y, (22,22,8))).long().unsqueeze(0), num_classes=8).float()
#         state = state #.cuda()
        
#         # Get network prediction
#         with torch.no_grad():
#             output = self.network(state.cuda())
            
#             # Calculate entropy from output probabilities
#             probs = F.softmax(output[0], dim=1)
#             entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            
#             action = torch.argmax(probs, dim=1)
            
#         return action.cuda(), entropy.item()  # Return action and entropy


# def show_state(env, step=0, changes=0, total_reward=0, name=""):
#     fig = plt.figure(10)
#     plt.clf()
#     plt.title("{} | Step: {} Changes: {} Total Reward: {}".format(name, step, changes, total_reward))
#     plt.axis('off')
#     plt.imshow(env.render(mode='rgb_array'))
#     plt.savefig(f"{step}.png")
#     # display.clear_output(wait=True)
#     # display.display(plt.gcf())


# def generate_pod_trajectory_with_next_state(env, difficulty, grid_size):
#     """
#     Generate a trajectory from an environment object using a trained neural network
#     that selects actions based on predicted state entropy.
    
#     Args:
#         env: The environment object to generate the trajectory from
#         difficulty (float): Difficulty factor that affects trajectory length
#         grid_size (int): Base grid size for calculating trajectory length
#         trained_network: A pre-trained construction network for entropy prediction
        
#     Returns:
#         tuple: (list of state, action, reward, next_state, done dictionaries, float average entropy)
#     """
#     def compute_returns(rewards, dones, gamma=0.95):
#         returns = []
#         discounted_reward = 0
#         reward = None
#         done = None


#         for reward, done in zip(reversed(rewards), reversed(dones)): 
#             if done:
#                 discounted_reward = 0

#             # import pdb; pdb.set_trace()
#             discounted_reward = reward + gamma * discounted_reward
#             if isinstance(discounted_reward, np.float64):
#                 discounted_reward = discounted_reward.item()
#             returns.insert(0, discounted_reward)

#         return returns

#     def transform(obs, x, y, obs_size):
#         map = obs
#         size = obs_size
#         pad = obs_size // 2
#         # import pdb; pdb.set_trace()
#         padded = np.pad(map.reshape(7,11), pad, constant_values=1)
#         cropped = padded[y: y + size, x: x + size]
        
#         return cropped
    
#     def hamm_dist(lvl1, lvl2):
#         value = np.abs(lvl1.flatten() - lvl2.flatten())
#         value[value != 0] = 1
#         return float(value.sum()) / float(len(lvl1.flatten()))
    
#     def get_closest_target(random_state):
#         goal_maps = []
#         for i in range(10,15):
#             with open(f"/home/jupyter-msiper/bootstrap-rl/src/goal_maps/zelda/{i}.txt", 'r') as f:
#                 lines = f.readlines()
            
#             # Remove empty lines and strip whitespace
#             lines = [line.strip() for line in lines if line.strip()]
            
#             # Create numpy array for the map
#             h = len(lines)
#             w = len(lines[0])
#             _map = np.zeros((h, w), dtype=np.uint8)
            
#             # Character to int mapping for Zelda
#             char_to_int = {
#                 '.': 0,  # empty
#                 'w': 1,  # wall
#                 'A': 2,  # goal
#                 '+': 3,  # key
#                 'g': 4,  # agent
#                 '1': 5,  # enemy 1
#                 '2': 6,  # enemy 2
#                 '3': 7   # enemy 3
#             }
            
#             # Convert characters to integers
#             for i, line in enumerate(lines):
#                 for j, char in enumerate(line):
#                     _map[i][j] = char_to_int[char]

#             goal_maps.append(_map)

#         closest_idx = 0
#         smallest_dist = np.inf
#         for curr_idx, goal_map in enumerate(goal_maps):
#             curr_dist = hamm_dist(random_state, goal_map)
#             if curr_dist < smallest_dist:
#                 smallest_dist = curr_dist
#                 closest_idx = curr_idx
#         return 10 + closest_idx
    
    
#     trajectory = []
#     trajectory_length = int(difficulty * grid_size[0] * grid_size[1])
    
#     # Reset the environment to get initial state
#     random_target_state = gen_random_map(None, 11, 7, {0: 0.58, 1:0.3, 2:0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})

#     lvl = get_closest_target(random_target_state)
#     state_pos_dict, info = env.reset(target_map=f"/home/jupyter-msiper/bootstrap-rl/src/goal_maps/zelda/{lvl}.txt")
#     # import pdb; pdb.set_trace()
#     y, x = state_pos_dict['pos']
#     y, x = y.item(), x.item()
#     state = state_pos_dict['map'] 
    
    
#     # Track entropies for averaging
#     episode_actions, episode_states, episode_rewards, episode_dones, episode_next_states = [], [], [], [], []
#     cum_reward = 0.0
    
#     for _ in range(trajectory_length):
#         # show_state(env)
#         # import pdb; pdb.set_trace()
#         repair_action = state[y][x]
        
#         # Select the action with the highest predicted entropy
#         noise_action = random_target_state[y][x]
#         next_state = F.one_hot(torch.tensor(transform(state, x, y, 22)).long().unsqueeze(0), num_classes=8).float()
#         episode_next_states.append(next_state)
#         next_state_pos_dict, reward, done, truncated, info = env.step(noise_action)
#         noise_state = F.one_hot(torch.tensor(transform(next_state_pos_dict['map'], x, y, 22)).long().unsqueeze(0), num_classes=8).float()
        
        
#         noise_stats = env._rep_stats
#         repaired_stats = env._prob.get_stats(get_string_map(state, env._prob.get_tile_types()))
#         repair_reward = env._prob.get_reward(repaired_stats, noise_stats)
#         # print(f"noise reward: {reward}, repair_reward: {repair_reward}, done: {done}")
        
#         episode_actions.append(F.one_hot(torch.tensor([repair_action]).long(), num_classes=8).float())
#         episode_states.append(noise_state)
#         cum_reward += repair_reward
#         episode_rewards.append(repair_reward)
#         # print(f"episode_rtgs: {episode_rtgs}")
#         episode_dones.append(done)

#         # step_dict['reward'] = reward
#         next_state = next_state_pos_dict['map']
#         y, x = next_state_pos_dict['pos']
#         x, y, = x.item(), y.item() 
        
#         # Update state for next iteration
#         state = next_state

#     # episode_rtgs = compute_returns(episode_rewards, episode_dones)
#     # print(f"episode_rtgs: {episode_rtgs}")
    
#     return episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones


# def generate_pod_trajectory(env, difficulty, grid_size):
#     """
#     Generate a trajectory from an environment object using a trained neural network
#     that selects actions based on predicted state entropy.
    
#     Args:
#         env: The environment object to generate the trajectory from
#         difficulty (float): Difficulty factor that affects trajectory length
#         grid_size (int): Base grid size for calculating trajectory length
#         trained_network: A pre-trained construction network for entropy prediction
        
#     Returns:
#         tuple: (list of state, action, reward, next_state, done dictionaries, float average entropy)
#     """
#     def compute_returns(rewards, dones, gamma=0.95):
#         returns = []
#         discounted_reward = 0
#         reward = None
#         done = None


#         for reward, done in zip(reversed(rewards), reversed(dones)): 
#             if done:
#                 discounted_reward = 0

#             # import pdb; pdb.set_trace()
#             discounted_reward = reward + gamma * discounted_reward
#             if isinstance(discounted_reward, np.float64):
#                 discounted_reward = discounted_reward.item()
#             returns.insert(0, discounted_reward)

#         return returns

#     def transform(obs, x, y, obs_size):
#         map = obs
#         size = obs_size
#         pad = obs_size // 2
#         # import pdb; pdb.set_trace()
#         padded = np.pad(map.reshape(7,11), pad, constant_values=1)
#         cropped = padded[y: y + size, x: x + size]
        
#         return cropped
    
#     def hamm_dist(lvl1, lvl2):
#         value = np.abs(lvl1.flatten() - lvl2.flatten())
#         value[value != 0] = 1
#         return float(value.sum()) / float(len(lvl1.flatten()))
    
#     def get_closest_target(random_state):
#         goal_maps = []
#         for i in range(10,30):
#             with open(f"./goal_maps/zelda/{i}.txt", 'r') as f:
#                 lines = f.readlines()
            
#             # Remove empty lines and strip whitespace
#             lines = [line.strip() for line in lines if line.strip()]
            
#             # Create numpy array for the map
#             h = len(lines)
#             w = len(lines[0])
#             _map = np.zeros((h, w), dtype=np.uint8)
            
#             # Character to int mapping for Zelda
#             char_to_int = {
#                 '.': 0,  # empty
#                 'w': 1,  # wall
#                 'A': 2,  # goal
#                 '+': 3,  # key
#                 'g': 4,  # agent
#                 '1': 5,  # enemy 1
#                 '2': 6,  # enemy 2
#                 '3': 7   # enemy 3
#             }
            
#             # Convert characters to integers
#             for i, line in enumerate(lines):
#                 for j, char in enumerate(line):
#                     _map[i][j] = char_to_int[char]

#             goal_maps.append(_map)

#         closest_idx = 0
#         smallest_dist = np.inf
#         for curr_idx, goal_map in enumerate(goal_maps):
#             curr_dist = hamm_dist(random_state, goal_map)
#             if curr_dist < smallest_dist:
#                 smallest_dist = curr_dist
#                 closest_idx = curr_idx

#         # print(f"closest target: {10 + closest_idx}")
#         return 10 + closest_idx
    
    
#     trajectory = []
#     trajectory_length = int(difficulty * grid_size[0] * grid_size[1])
    
#     # Reset the environment to get initial state
#     random_target_state = gen_random_map(None, 11, 7, {0: 0.58, 1:0.3, 2:0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})

#     lvl = get_closest_target(random_target_state)
#     state_pos_dict, info = env.reset(target_map=f"./goal_maps/zelda/{lvl}.txt")
#     y, x = state_pos_dict['pos']
#     y, x = y.item(), x.item()
#     state = state_pos_dict['map']
    
    
#     # Track entropies for averaging
#     episode_actions, episode_states, episode_rewards, episode_dones, episode_next_states = [], [], [], [], []
#     cum_reward = 0.0
    
#     for _ in range(trajectory_length):
#         # show_state(env)
#         # import pdb; pdb.set_trace()
#         repair_action = state[y][x]
        
#         # Select the action with the highest predicted entropy
#         noise_action = random_target_state[y][x]
#         next_state_pos_dict, reward, done, truncated, info = env.step(noise_action)
#         noise_state = F.one_hot(torch.tensor(transform(next_state_pos_dict['map'], x, y, 22)).long().unsqueeze(0), num_classes=8).float()
        
        
#         noise_stats = env._rep_stats
#         repaired_stats = env._prob.get_stats(get_string_map(state, env._prob.get_tile_types()))
#         repair_reward = env._prob.get_reward(repaired_stats, noise_stats)
#         # print(f"noise reward: {reward}, repair_reward: {repair_reward}, done: {done}")
        
#         episode_actions.append(F.one_hot(torch.tensor([repair_action]).long(), num_classes=8).float())
#         episode_states.append(noise_state)
#         cum_reward += repair_reward
#         episode_rewards.append(repair_reward)
#         # print(f"episode_rtgs: {episode_rtgs}")
#         episode_dones.append(done)

#         # step_dict['reward'] = reward
#         next_state = next_state_pos_dict['map']
#         y, x = next_state_pos_dict['pos']
#         x, y, = x.item(), y.item() 
        
#         # Update state for next iteration
#         state = next_state

#     episode_rtgs = compute_returns(episode_rewards, episode_dones)
#     # print(f"episode_rtgs: {episode_rtgs}")
    
#     return episode_states, episode_actions, episode_rtgs, episode_dones


# def generate_pod_trajectory_with_logprob_actions(env, difficulty, grid_size):
#     """
#     Generate a trajectory from an environment object using a trained neural network
#     that selects actions based on predicted state entropy.
    
#     Args:
#         env: The environment object to generate the trajectory from
#         difficulty (float): Difficulty factor that affects trajectory length
#         grid_size (int): Base grid size for calculating trajectory length
#         trained_network: A pre-trained construction network for entropy prediction
        
#     Returns:
#         tuple: (list of state, action, reward, next_state, done dictionaries, float average entropy)
#     """
#     def compute_returns(rewards, dones, gamma=0.95):
#         returns = []
#         discounted_reward = 0
#         reward = None
#         done = None


#         for reward, done in zip(reversed(rewards), reversed(dones)): 
#             if done:
#                 discounted_reward = 0

#             # import pdb; pdb.set_trace()
#             discounted_reward = reward + gamma * discounted_reward
#             if isinstance(discounted_reward, np.float64):
#                 discounted_reward = discounted_reward.item()
#             returns.insert(0, discounted_reward)

#         return returns

#     def transform(obs, x, y, obs_size):
#         map = obs
#         size = obs_size
#         pad = obs_size // 2
#         # import pdb; pdb.set_trace()
#         padded = np.pad(map.reshape(7,11), pad, constant_values=1)
#         cropped = padded[y: y + size, x: x + size]
        
#         return cropped
    
#     def hamm_dist(lvl1, lvl2):
#         value = np.abs(lvl1.flatten() - lvl2.flatten())
#         value[value != 0] = 1
#         return float(value.sum()) / float(len(lvl1.flatten()))
    
#     def get_closest_target(random_state):
#         goal_maps = []
#         for i in range(10,15):
#             with open(f"./goal_maps/zelda/{i}.txt", 'r') as f:
#                 lines = f.readlines()
            
#             # Remove empty lines and strip whitespace
#             lines = [line.strip() for line in lines if line.strip()]
            
#             # Create numpy array for the map
#             h = len(lines)
#             w = len(lines[0])
#             _map = np.zeros((h, w), dtype=np.uint8)
            
#             # Character to int mapping for Zelda
#             char_to_int = {
#                 '.': 0,  # empty
#                 'w': 1,  # wall
#                 'A': 2,  # goal
#                 '+': 3,  # key
#                 'g': 4,  # agent
#                 '1': 5,  # enemy 1
#                 '2': 6,  # enemy 2
#                 '3': 7   # enemy 3
#             }
            
#             # Convert characters to integers
#             for i, line in enumerate(lines):
#                 for j, char in enumerate(line):
#                     _map[i][j] = char_to_int[char]

#             goal_maps.append(_map)

#         closest_idx = 0
#         smallest_dist = np.inf
#         for curr_idx, goal_map in enumerate(goal_maps):
#             curr_dist = hamm_dist(random_state, goal_map)
#             if curr_dist < smallest_dist:
#                 smallest_dist = curr_dist
#                 closest_idx = curr_idx
#         return 10 + closest_idx
    
    
#     trajectory = []
#     trajectory_length = int(difficulty * grid_size[0] * grid_size[1])
    
#     # Reset the environment to get initial state
#     random_target_state = gen_random_map(None, 11, 7, {0: 0.58, 1:0.3, 2:0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02})

#     lvl = get_closest_target(random_target_state)
#     state_pos_dict, info = env.reset(target_map=f"./goal_maps/zelda/{lvl}.txt")
#     y, x = state_pos_dict['pos']
#     y, x = y.item(), x.item()
#     state = state_pos_dict['map'] 
    
    
#     # Track entropies for averaging
#     episode_actions, episode_states, episode_rewards, episode_dones, episode_logprob_actions = [], [], [], [], []
#     cum_reward = 0.0
    
#     for _ in range(trajectory_length):
#         # show_state(env)
#         # import pdb; pdb.set_trace()
#         repair_action = state[y][x]
        
#         # Select the action with the highest predicted entropy
#         noise_action = random_target_state[y][x]
#         next_state_pos_dict, reward, done, truncated, info = env.step(noise_action)
#         noise_state = F.one_hot(torch.tensor(transform(next_state_pos_dict['map'], x, y, 22)).long().unsqueeze(0), num_classes=8).float()
#         noise_stats = env._rep_stats
#         repaired_stats = env._prob.get_stats(get_string_map(state, env._prob.get_tile_types()))
#         repair_reward = env._prob.get_reward(repaired_stats, noise_stats)
#         # print(f"noise reward: {reward}, repair_reward: {repair_reward}, done: {done}")
        
#         episode_actions.append(F.one_hot(torch.tensor([repair_action]).long(), num_classes=8).float())
#         episode_logprob_actions.append(F.one_hot(torch.tensor([repair_action]).long(), num_classes=8).float()-1e-2)
#         episode_states.append(noise_state)
#         cum_reward += repair_reward
#         episode_rewards.append(repair_reward)
#         # print(f"episode_rtgs: {episode_rtgs}")
#         episode_dones.append(done)

#         # step_dict['reward'] = reward
#         next_state = next_state_pos_dict['map']
#         y, x = next_state_pos_dict['pos']
#         x, y, = x.item(), y.item() 
        
#         # Update state for next iteration
#         state = next_state

#     episode_rtgs = compute_returns(episode_rewards, episode_dones)
#     # print(f"episode_rtgs: {episode_rtgs}")
    
#     return episode_states, episode_actions, episode_rtgs, episode_dones, episode_logprob_actions


# def set_random_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def set_device() -> torch.device:
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         torch.cuda.empty_cache()
#     elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#         device = torch.device('mps')    # for Apple Macbook GPUs
#     else:
#         device = torch.device('cpu')

#     device = torch.device('cpu')

#     # Set default dtype to float32
#     torch.set_default_dtype(torch.float32)
#     return device


# def setup_logging(
#     verbose: bool = True,
#     log_dir: Optional[str] = None,
#     max_file_size: int = 10 * 1024 * 1024,  # 10MB
#     backup_count: int = 5
# ):
#     """
#     Sets up advanced logging configuration with colored console output and rotating file logs.
    
#     Args:
#         verbose: If True, sets logging level to INFO, otherwise WARNING
#         log_dir: Directory to store log files. If None, uses current directory
#         max_file_size: Maximum size of each log file in bytes
#         backup_count: Number of backup files to keep
#     """
#     # Create log directory if it doesn't exist
#     log_dir = Path(log_dir) if log_dir else Path.cwd() / 'logs'
#     log_dir.mkdir(parents=True, exist_ok=True)
    
#     # Set up logging format
#     log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     date_format = '%Y-%m-%d %H:%M:%S'
    
#     # Color format for console output
#     color_format = {
#         'DEBUG': 'cyan',
#         'INFO': 'green',
#         'WARNING': 'yellow',
#         'ERROR': 'red',
#         'CRITICAL': 'red,bg_white',
#     }
    
#     # Create root logger
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.INFO if verbose else logging.WARNING)

#     # Remove any existing handlers
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)
    
#     # Console Handler with colors
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(
#         colorlog.ColoredFormatter(
#             f'%(log_color)s{log_format}%(reset)s',
#             datefmt=date_format,
#             log_colors=color_format
#         )
#     )
#     root_logger.addHandler(console_handler)

#     # Rotating File Handler
#     file_handler = RotatingFileHandler(
#         filename=log_dir / 'training.log',
#         maxBytes=max_file_size,
#         backupCount=backup_count,
#         mode='a'
#     )
#     file_handler.setFormatter(
#         logging.Formatter(log_format, datefmt=date_format)
#     )
#     root_logger.addHandler(file_handler)
    
#     # Suppress verbose logging from other libraries
#     for lib in ['PIL', 'gym', 'wandb', 'urllib3', 'matplotlib']:
#         logging.getLogger(lib).setLevel(logging.WARNING)
    
#     logger = logging.getLogger(__name__)
#     logger.info('Logging system initialized')
#     logger.info(f'Log files will be saved in: {log_dir.absolute()}')

#     return logger