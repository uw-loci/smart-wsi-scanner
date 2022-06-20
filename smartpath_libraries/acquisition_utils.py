import numpy as np

def limit_stage(aim_pos, hard_limits, default=None):
    """
        Args: aim_pos (float): single axisl position to check. hard_limits (tuple): min and max stage position. default (float): return this value if out of range (optional).
    """
    assert len(hard_limits)==2, "Need 2 values for stage hard limits."
    if hard_limits[0] <= aim_pos <= hard_limits[1]:
        return aim_pos
    else:
        if default is not None and (hard_limits[0] <= default <= hard_limits[1]):
            print('Aimed position out of range. Reset to {}'.format(default))
        else:
            raise ValueError('Aimed position and default position out of range/not specified. Aborted.')
            

def distance(pos, support_points):
    """
        This function determines the closest support point to the input `pos`.
        Args: pos (tuple): position (x, y). support_points (list): list of support points, each element is a (x, y) tuple.
        Returns: (int) the index of the determined support point. (float) the distance to the returned support point.
    """
    pos = np.array(pos)
    support_points = np.array(support_points)
    distances = np.sqrt((pos - support_points)[:, 0]**2 + (pos - support_points)[:, 1]**2)
    idx = np.argmin(distances, axis=0)
    return idx, distances[idx]
            
            
