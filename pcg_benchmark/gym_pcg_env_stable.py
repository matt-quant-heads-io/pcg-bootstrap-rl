import copy 
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .pcg_env import PCGEnv, _recursiveDiversity
import random
from pathlib import Path


"""
PCG Environment class. This class is the base class where the user is interacting with.
Please don't construct this class by hand but instead use the make function from pcg_benchmark.
For example, the following code creates the environment class for the zelda-v0 problem.

import pcg_benchmark

env = pcg_benchmark.make('zelda-v0')
"""
class GymPCGEnv(PCGEnv, gym.Env):
    """
    Parameters:
        name(str): the string name that defines the current problem
        problem(Problem): a subclass of Problem that need to be solved
    """
    def __init__(self, name, problem):
        super().__init__(name, problem)
        # import pdb; pdb.set_trace()
        self.prob_config = yaml.safe_load(open(str(Path(__file__).parent.parent)+f"/configs/envs/{name}.yaml", "r")) 
        print(f"self.prob_config: {self.prob_config}")
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
        # Episode tracking
        self._current_content = None
        self._step_count = 0
        self._max_steps = 1000  # You may want to make this configurable
        self._done = False

    def _create_action_space(self):
        """Create the action space based on your problem's requirements"""
        range_dict = self._problem._content_space.range()
        return range_dict['max']
    
    # def _create_observation_space(self):
    #     """Create the observation space based on your content representation"""
    #     self._height = self._problem._height
    #     self._width = self._problem._width
    #     return (self._height, self._width, self.action_space)

    def _create_observation_space(self):
        self._height = self._problem._height
        self._width = self._problem._width
        self._channels = 1  # or set dynamically if multi-channel
        # import pdb; pdb.set_trace()

        obs_shape = (self._height, self._width)  # Channels First for PyTorch
        return spaces.Box(low=0, high=len(self.prob_config['tiles'])-1, shape=obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            self.seed(int(seed))
            
        self._current_content = self._get_initial_content()
        self._positions_queue = []

        for y in range(self._height):
            for x in range(self._width):
                self._positions_queue.append((y,x))

        self._step_count = 0
        self._done = False
        
        observation = self._get_observation()
        self.pos = observation['pos']
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # Gracefully handle stepping after episode end
        if self._done:
            # Option A: soft handling — return a terminal transition and immediately reset
            observation, info = self.reset()
            # terminal no-op reward; you can set to 0.0 or something else
            return observation, 0.0, True, False, info

            # Option B (strict): uncomment to enforce Gym behavior and catch bugs early
            # raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # --- normal step ---
        prev_content = copy.deepcopy(self._current_content)
        self._current_content = self._apply_action(action)
        self._step_count += 1

        observation = self._get_observation()
        self.pos = observation['pos']

        reward = self._calculate_reward(self._current_content, prev_content)

        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_steps
        self._done = terminated or truncated

        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_initial_content(self):
        """Get initial content for the episode"""
        # Sample from content space or use a default starting point
        arr = np.random.rand(self._height*self._width).astype(np.int32)
        
        return np.array(arr).reshape(self._height,self._width)
    
    def _apply_action(self, action):
        """Apply the given action to modify the current content"""
        import copy 
        
        updated_content = copy.deepcopy(self._current_content)
        updated_content[self._pos[0]][self._pos[1]] = float(action)
        
        return updated_content
    
    # def _get_observation(self):
    #     """Get current observation from the current content state"""
    #     self._pos = self._positions_queue.pop()
    #     return {"map": self._current_content, "pos": self._pos} 
    # def _get_observation(self):
    #     self._pos = self._positions_queue.pop()
    #     obs_map = np.array(self._current_content, dtype=np.float32).reshape(1, self._height, self._width)
    #     return {"map": obs_map, "pos": self._pos}

    def _get_observation(self):
        self._pos = self._positions_queue.pop()
        obs_map = np.array(self._current_content, dtype=np.int64).reshape(1, self._height, self._width)
        return {"map": obs_map, "pos": self._pos}
    
    def _calculate_reward(self, new_content, old_content):
        """Calculate reward based on current content"""
        # You can use your existing quality, diversity, controlability methods
        new_info = self._problem.info(new_content)
        old_info = self._problem.info(old_content)
        
        new_quality_score = self._problem.quality(new_info)
        old_quality_score = self._problem.quality(old_info)
        
        # Simple reward based on quality - you may want to make this more sophisticated
        return (new_quality_score - old_quality_score)*100.0
    
    def _is_terminated(self):
        """Check if episode should terminate early"""
        # For example, terminate if quality threshold is reached
        info = self._problem.info(np.array(self._current_content).reshape(self._height, self._width)) #self._problem.info(self.content_space.restructure(self._current_content))
        quality_score = self._problem.quality(info)
        return quality_score >= 1.0 or len(self._positions_queue) == 0  # Adjust threshold as needed
    
    def _get_info(self):
        """Get additional info dictionary"""
        if self._current_content is None:
            return {}
        
        
        info_dict = self._problem.info(self._current_content)
        quality_score = self._problem.quality(info_dict)
        
        return {
            'quality': quality_score,
            'step_count': self._step_count,
            'content_info': info_dict
        }

    def close(self):
        """Clean up resources"""
        pass

    """
    Content space property to check range or sample

    Returns:
        Space: the space of all the possible content
    """
    @property
    def content_space(self):
        return self._problem._content_space
    
    """
    Control parameter space property to check range or sample

    Returns:
        Space: the space of all the possible control parameters
    """
    @property
    def control_space(self):
        return self._problem._control_space
    
    """
    Adjust the seed of the random number generator used by the problem spaces

    Parameters:
        seed(int): the seed for the random number generator used by the problem
    """
    def seed(self, seed):
        self._problem._random = np.random.default_rng(seed)
        if(self.content_space == None):
            raise AttributeError("self._content_space is not initialized")
        self.content_space.seed(seed)
        if(self.control_space == None):
            raise AttributeError("self._control_space is not initialized")
        self.control_space.seed(seed)
    
    """
    Calculate some basic information about the contents. These information can be used to speed quality,
    diversity, and controlability calculations. You can send this to any of the other functions (quality, 
    diversity, controlability) and they will immedietly return the results as all the information are
    precomputed. 

    Parameters:
        contents(any|any[]): a single or an array of content that need to be analyzed

    Returns:
        any|any[]: a single or an array of information that can be cached to speed quality, diversity, 
        controlability calculations.
    """
    def info(self, contents):
        is_array = False
        is_content = self.content_space.isSampled(self.content_space.restructure(contents))
        if is_content:
            contents = [contents]
        else:
            is_array = hasattr(contents, "__len__") and not isinstance(contents, dict)
            if is_array:
                is_content = self.content_space.isSampled(self.content_space.restructure(contents))
            
        if not is_content:
            import pdb; pdb.set_trace()
            raise ValueError(f"wrong input for the function, the contents are not sampled from the content space.")

        info = []
        for c in contents:
            info.append(self._problem.info(c))
        if not is_array:
            return info[0]
        return info
        
    """
    Calculate the quality of the contents provided for the current problem

    Parameters:
        contents(any|any[]): a single or an array of contents or infos that need to be evaluated for quality

    Returns:
        float: percentage of passed content
        float[]: an array of the quality of each content between 0 and 1
        any[]: an array of the info of all the contents that can be cached to speed all the calculations
    """
    def quality(self, contents):
        is_array = False
        is_content = self.content_space.isSampled(self.content_space.restructure(contents))
        if is_content:
            contents = [contents]
        else:
            is_array = hasattr(contents, "__len__") and not isinstance(contents, dict)
            if is_array:
                is_content = self.content_space.isSampled(self.content_space.restructure(contents))
            else:
                contents = [contents]
        
        infos = []
        if is_content:
            infos = self._problem.info(np.array(self._current_content).reshape(self._height, self._width)) #self._problem.info(self.content_space.restructure(contents[0]))
        else:
            infos = contents

        quality = []
        for i in infos:
            quality.append(self._problem.quality(i))
        quality = np.array(quality)

        if not is_array:
            return float(quality[0] >= 1), quality[0], infos[0]
        return (quality >= 1).sum() / len(infos), quality, infos

    """
    Calculate the diversity of the contents provided for the current problem

    Parameters:
        contents(any|any[]): a single or an array of contents or infos that need to be evaluated for diversity

    Returns:
        float: percentage of passed content
        float[]: an array of the diversity values for each content between 0 and 1
        any[]: an array of the info of all the contents that can be cached to speed all the calculations
    """
    def diversity(self, contents):
        is_array = False
        is_content = self.content_space.isSampled(contents)
        if is_content:
            contents = [contents]
        else:
            is_array = hasattr(contents, "__len__") and not isinstance(contents, dict)
            if is_array:
                is_content = self.content_space.isSampled(contents[0])
            else:
                contents = [contents]

        infos = []
        if is_content:
            infos = self._problem.info(contents)
        else:
            infos = contents
        
        sim_mat = np.zeros((len(infos), len(infos)))
        for i1 in range(len(infos)):
            for i2 in range(len(infos)):
                sim_mat[i1][i2] = 1.0-self._problem.diversity(infos[i1], infos[i2])

        diversity = np.array(_recursiveDiversity(infos, sim_mat))

        if not is_array:
            return float(diversity[0] >= 1), diversity[0], infos[0]
        return (diversity >= 1).sum() / len(infos), diversity, infos

    """
    Calculate the controlability of the provided contents with respect to the provided controls 
    for the current problem

    Parameters:
        contents(any|any[]): a single or an array of contents or infos that need to be evaluated for controlability
        controls(any|any[]): a single or an array of controls that is used to evaluate the control of the contents

    Returns:
        float: percentage of passed content, 
        float[]: an array of the controlability values for each content between 0 and 1
        any[]: an array of the info of all the contents that can be cached to speed all the calculations
    """
    def controlability(self, contents, controls):
        is_array = False
        is_content = self.content_space.isSampled(contents)
        if is_content:
            contents = [contents]
            controls = [controls]
        else:
            is_array = hasattr(contents, "__len__") and not isinstance(contents, dict)
            if is_array:
                is_content = self.content_space.isSampled(contents[0])
            else:
                contents = [contents]
                controls = [controls]

        infos = []
        if is_content:
            infos = self._problem.info(contents)
        else:
            infos = contents
        if len(infos) != len(controls):
            raise ValueError(f"Length of contents ({len(infos)}) is not equal to length of controls ({len(controls)})")
        
        controlability = []
        for i, ct in zip(infos, controls):
            controlability.append(self._problem.controlability(i, ct))
        controlability = np.array(controlability)
        
        if not is_array:
            return float(controlability[0] >= 1), controlability[0], infos[0]
        return (controlability >= 1).sum() / len(infos), controlability, infos
    
    """
    Evaluate the input contets and controls for quality, diversity, and controlability for the current problem

    Parameters:
        contents(any|any[]): a single or an array of contents or infos that need to be evaluated
        controls(any|any[]): a single or an array of controls that is used to evaluate the control of the contents

    Returns:
        float: percentage of quality passed content
        float: percentage of diversity passed content
        float: percentage of controlability passed content 
        dict[str,float[]]: a dictionary of float arrays that contains the details for quality, diversity, and controlability 
        any[]: an array of the info of all the contents that can be cached to speed all the calculations
    """
    def evaluate(self, contents, controls=None):
        infos = self._problem.info(contents)
        q_score, quality, _ = self.quality(infos)
        d_score, diversity, _ = self.diversity(infos)
        ct_score, controlability = 0.0, 0.0
        if hasattr(infos, "__len__") and not isinstance(infos, dict):
            controlability = [0.0] * len(infos)
        if controls is not None:
            ct_score, controlability, _ = self.controlability(infos, controls)

        return q_score, d_score, ct_score, {
            "quality": quality, 
            "diversity": diversity, 
            "controlability": controlability
        }, infos
    
    """
    Generate an PIL.Image for each of the content

    Parameters:
        contents(any|any[]): a single or an array of the content that need to be rendered

    Returns:
        any|any[]: a rendered representaion for each of the input content which can be string, image, video, image sequence, etc.
    """
    def render(self, contents):
        single_input = False
        if self.content_space.isSampled(contents):
            contents = [contents]
            single_input = True

        result = []
        for c in contents:
            result.append(self._problem.render(c))
        
        if single_input:
            return result[0]
        return result

