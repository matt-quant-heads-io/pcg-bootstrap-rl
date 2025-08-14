import numpy as np

"""
An internal recurrsive function to calculate the number of unique content and the minimum 
diversity value for each input content. This function might be replaced in the future with better one 
such as Vendi Score.

Parameters:
    infos (dict[]): a list of dictionary containing the information about each content
    sim_matrix (float[][]): a matrix containing the similarity value between all the contents
    indces (int[]): an array of the indices of the contents that need to be evaluated (don't provide that)

Returns:
    float[]: an array of the minimum diversity value for each content
"""
def _recursiveDiversity(infos, sim_matrix, indices=None):
    if indices == None:
        indices = list(range(len(infos)))
    values = sim_matrix.sum(axis=1)
    max_value = np.max(values)
    if max_value <= 1:
        return [1.0] * len(infos)
    index = np.argmax(values)
    amount = (sim_matrix[index] > 0).sum()
    div_value = min(max(1-(max_value - 1) / amount, 0.0), 1.0)
    index_value = indices[index]
    new_infos = infos.copy()
    new_infos.pop(index)
    indices.pop(index)
    new_sim_matrix = sim_matrix.copy()
    new_sim_matrix=np.delete(new_sim_matrix, (index), axis=0)
    new_sim_matrix=np.delete(new_sim_matrix, (index), axis=1)
    tempArray = _recursiveDiversity(new_infos, new_sim_matrix, indices)
    tempArray.insert(index_value, div_value)
    return tempArray


"""
PCG Environment class. This class is the base class where the user is interacting with.
Please don't construct this class by hand but instead use the make function from pcg_benchmark.
For example, the following code creates the environment class for the zelda-v0 problem.

import pcg_benchmark

env = pcg_benchmark.make('zelda-v0')
"""
class PCGEnv:
    """
    Parameters:
        name(str): the string name that defines the current problem
        problem(Problem): a subclass of Problem that need to be solved
    """
    def __init__(self, name, problem):
        self._name = name
        self._problem = problem

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
        try:
            is_array = False
            import pdb; pdb.set_trace()
            is_content = self.content_space.isSampled(contents)
            if is_content:
                contents = [contents]
            else:
                is_array = hasattr(contents, "__len__") and not isinstance(contents, dict)
                if is_array:
                    is_content = self.content_space.isSampled(contents)
                
            if not is_content:
                raise ValueError(f"wrong input for the function, the contents are not sampled from the content space.")

            info = []
            for c in contents:
                info.append(self._problem.info(c))
            if not is_array:
                return info[0]
            return info
        except Exception as e:
            
            import pdb; pdb.set_trace()
        
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
            infos = self.info(contents)
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
            infos = self.info(contents)
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
            infos = self.info(contents)
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
        infos = self.info(contents)
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
