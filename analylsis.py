import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

TAG_POSITIONS = {
    "A0": [0.0,0.0,3.653],
    "A1": [3.35,0.0,3.653],
    "A2": [6.7,0.0,3.653],
    "A3": [0.0,2.89,2.653],
    "A4": [3.35,2.89,3.653],
    "A5": [6.7,2.89,3.653]
}

# returns list of keys of ranging anchors given a position from shortest to furthest distance
def findRangingAnchors(pt):
    distances = {}
    for key in TAG_POSITIONS:
        distances[np.linalg.norm(np.array(TAG_POSITIONS[key]) - np.array(pt))] = key

    distances_sorted = OrderedDict(sorted(distances.items()))
    keys_sorted = list(distances_sorted.values())
    vals_sorted = list(distances_sorted.keys())

    sum_euclid_dist = sum(vals_sorted[0:4])
    longest_euclid_dist = vals_sorted[3]
    return keys_sorted[0:4], sum_euclid_dist, longest_euclid_dist


def findDistances(x_act, y_act, z_act):
    
    # need to add data frame on to it
    dist_info1 = []
    dist_info2 = []
    dist_info3 = []
    for i in range(len(x_act)):
        pt = [x_act[i], y_act[i], z_act[i]]
        keys_ranging, sum_euclid_dist, longest_euclid_dist = findRangingAnchors(pt)
        dist_info1.append(keys_ranging)
        dist_info2.append(sum_euclid_dist)
        dist_info3.append(longest_euclid_dist)
        # print("Keys: " + str(keys_ranging))
        # print("Sum: " + str(sum_euclid_dist))
        # print("Longest: " + str(longest_euclid_dist))
    
    return dist_info1, dist_info2, dist_info3

def findError(x_act, y_act, z_act, x_obs, y_obs, z_obs):
    errs = []
    for i in range(len(x_act)):
        actual_pt = np.array([x_act[i], y_act[i], z_act[i]])
        measured_pt = np.array([x_obs[i], y_obs[i], z_obs[i]])  
        errs.append(np.linalg.norm(actual_pt-measured_pt))

    return errs


def graphLongestDist(longest_dist, err_info):
    plt.title("Error versus Euclidean Distance from Furthest Anchor")
    plt.xlabel('Distance (m)')
    plt.ylabel('Error (m)')
    plt.scatter(longest_dist, err_info)
    plt.show()

def graphSumDist(sum_dist, err_info):
    plt.title("Error versus Sum of Euclidean Distances from Ranging Anchors")
    plt.xlabel('Distance (m)')
    plt.ylabel('Error (m)')
    plt.scatter(sum_dist, err_info)
    plt.show()
    

def mainStationary():
    df = pd.read_csv('uwb_test_data_stationary.csv')  
    x_act = df['x_actual'].tolist()
    y_act = df['y_actual'].tolist()
    z_act = df['z_actual'].tolist()

    x_obs = df['x_obs'].tolist()
    y_obs = df['y_obs'].tolist()
    z_obs = df['z_obs'].tolist()

    ranging_keys, sum_dist, longest_dist = findDistances(x_act, y_act, z_act)
    err_info = findError(x_act, y_act, z_act, x_obs, y_obs, z_obs)
    
    # graphLongestDist(longest_dist, err_info)
    graphSumDist(sum_dist, err_info)



def mainTwoWayRanging():
    df = pd.read_csv('uwb_two_way_ranging.csv')
    actual = df['actual_distance']
    measured = df['md']
    
    errors = []
    for i in range(len(actual)):
        diff = abs(actual[i] - measured[i])
        errors.append(diff)

    plt.title("Error versus Distance for Two Way Ranging")
    plt.xlabel('Distance (m)')
    plt.ylabel('Error (m)')
    plt.scatter(actual, errors)
    plt.show()    


if __name__ == "__main__":
    # mainStationary()
    mainTwoWayRanging()
