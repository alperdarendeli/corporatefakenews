import numpy as np


def _haversine(theta: np.ndarray) -> np.ndarray:
    '''
    Internal method for haversine function
    '''
    return np.square(np.sin(theta/2))


def get_distance(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    '''
    Calculates the distance in km between two points (lat, long) on Earth
    pt1 and pt2 have dimensions (N, 2) for N number of points
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    '''
    pt1, pt2 = np.radians(pt1), np.radians(pt2)
    lat1, long1, lat2, long2 = pt1[:,0], pt1[:,1], pt2[:,0], pt2[:,1]
    return 2 * 6371 * np.arcsin(np.sqrt(_haversine(lat2-lat1) + np.cos(lat1) * np.cos(lat2) * _haversine(long2-long1)))


if __name__ == '__main__':

    ## Unit test: Get distance
    sg = np.array([[1.352083, 103.819839]])
    ny = np.array([[40.712776, -74.005974]])
    mlb = np.array([[-37.813629, 144.963058]])
    sg_ny, sg_mlb = 15339.58, 6055.92
    print(f'SG-NY  : True={sg_ny}, Calculated={get_distance(sg, ny)[0]}')
    print(f'SG-MLB : True={sg_mlb}, Calculated={get_distance(sg, mlb)[0]}')