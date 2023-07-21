# imports
import trajectory
import point
import region
import utils
import numpy as np
from rtree import index
import matplotlib.pyplot as plt
import time


def douglasPeucker(traj:trajectory,traj_number, epsilon, new_traj) -> trajectory:
    point_number = len(traj.points)

    distance_max = 0
    index = 0
    # calculate the distance between each point of trajectory and a line segment
    for i in range(1, point_number - 1):
        distance = utils.calculateDistance(traj.points[i], traj.points[0], traj.points[-1])

        # find a point with max distance
        if distance > distance_max:
            index = i
            distance_max = distance

    # compare with epsilon
    if distance_max >= epsilon:
        #if more than epsilon then split into two trajectories and repeat the method separately for them
        res_1 = douglasPeucker(utils.sliceTrajectory(traj, 0, index, traj_number), traj_number, epsilon, new_traj)
        res_1 = utils.sliceTrajectory(res_1, 0, -1, traj_number)
        res_2 = douglasPeucker(utils.sliceTrajectory(traj, index, point_number, traj_number), traj_number, epsilon, new_traj)
        new_traj = utils.uppendTrajectory(res_1, res_2, traj_number)

    else:
        # if not - add the edge points
        new_traj.addPoint(traj.points[0])
        new_traj.addPoint(traj.points[-1])

    return new_traj


def slidingWindow(traj:trajectory,epsilon, new_traj) -> trajectory:
    point_number = len(traj.points)
    anchor = 0

    # add the fist point at any case
    new_traj.addPoint(traj.points[0])

    while anchor < point_number:
        segment_end = anchor + 2

        while segment_end < point_number:

            # find the farthest point from the segment and calculate the distance
            segment_error_max = 0
            for i in range(anchor, segment_end - 1):
                segment_error = utils.calculateDistance(traj.points[i], traj.points[anchor], traj.points[segment_end])

                if segment_error > segment_error_max:
                    segment_error_max = segment_error

            # check if the error greater than epsilon
            if segment_error_max > epsilon:
                break

            # extend the segment
            segment_end += 1

        # change the anchor to the last point of the previous segment
        anchor = segment_end

        new_traj.addPoint(traj.points[segment_end - 1])

    new_traj.addPoint(traj.points[-1])

    return new_traj

def closestPairDistance(traj0:trajectory,traj1:trajectory) -> float:
    min_distance = float('inf')  # Initialize minimum distance as infinity
    point_pair = None  # Initialize the pair of points with the minimum distance

    # Iterate over each point in traj0
    for i, p0 in enumerate(traj0.points):
        # Iterate over each point in traj1
        for j, p1 in enumerate(traj1.points):
            # Calculate the Euclidean distance between p0 and p1 using the pointDistance function from utils.py
            distance = utils.pointDistance(p0, p1)

            # Update the minimum distance and point pair if the calculated distance is smaller
            if distance < min_distance:
                min_distance = distance
                point_pair = ((i, j), (p0, p1))
    return min_distance, point_pair


def dynamicTimeWarping(traj0:trajectory,traj1:trajectory) -> float:
    # Calculate the length of the two trajectories
    n = len(traj0.points)
    m = len(traj1.points)

    # Create a 2D matrix to store the DTW distances
    dtw_matrix = np.empty((n, m))

    # Calculate the DTW distances using dynamic programming
    for i in range(n):
        for j in range(m):
            # Calculate the pairwise Euclidean distances
            distance = utils.pointDistance(traj0.points[i], traj1.points[j]) 

            if i == 0 and j == 0:
                dtw_matrix[i][j] = distance
            elif i == 0:
                dtw_matrix[i][j] = distance + dtw_matrix[i][j-1]
            elif j == 0:
                dtw_matrix[i][j] = distance + dtw_matrix[i-1][j]
            else:
                dtw_matrix[i][j] = distance + min(
                    dtw_matrix[i-1][j],         # DTW(T0, R(T1))
                    dtw_matrix[i][j-1],         # DTW(R(T0), T1)
                    dtw_matrix[i-1][j-1]        # DTW(R(T0), R(T1))
                )

    # Retrieve the DTW distance by tracing the optimal path through the DTW matrix
    i = n - 1
    j = m - 1
    dtw_distance = dtw_matrix[i][j]

    # Store the indices of the optimal path
    optimal_path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_value = min(dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            if min_value == dtw_matrix[i-1][j-1]:
                i -= 1
                j -= 1
            elif min_value == dtw_matrix[i][j-1]:
                j -= 1
            else:
                i -= 1
        dtw_distance += dtw_matrix[i][j]
        # Add the current indices to the optimal path
        optimal_path.append((i, j))

    # Reverse the optimal path to start from the beginning
    optimal_path.reverse()

    return dtw_distance, optimal_path


def build_rtree_for_points(list_of_trajectories):
    p = index.Property()
    p.dimension = 2  # 2D points (x, y)
    p.near_minimum_overlap_factor = 4
    p.index_capacity = 5
    p.leaf_capacity = 5
    p.fill_factor = 0.9
    p.split_distribution_factor = 0.5
    idx = index.Index(properties=p)

    for traj in list_of_trajectories:
        for traj_point in traj.points:
            idx.insert(traj.number, (traj_point.X, traj_point.Y, traj_point.X, traj_point.Y))

    return idx

def get_trajectory_by_number(list_of_trajectories, trajectory_number):
    for traj in list_of_trajectories:
        if traj.number == trajectory_number:
            return traj
    return None

# def solveQueryWithRTree(r:region, tree:rtree, trajectories:list) -> list:
def solveQueryWithRTree(r:region,trajectories:list) -> list:
    results = set()

    # Build the R-tree for the points in trajectories
    rtree_index = build_rtree_for_points(trajectories)
    
    intersecting_rectangles = rtree_index.intersection((r.center.X - r.radius, r.center.Y - r.radius, r.center.X + r.radius, r.center.Y + r.radius))

    # Iterate through the intersecting rectangles to find points within the circle
    for rect_idx in intersecting_rectangles:
        # Get the trajectory index associated with the entry
        traj_index = rect_idx

        traj = get_trajectory_by_number(trajectories, traj_index)

        if traj:
            # If the trajectory is found, check if any of its points lie within the circle
            for traj_point in traj.points:
                if utils.pointDistance(r.center, traj_point) <= r.radius:
                    results.add(traj)

    return list(results)  

def solveQueryWithoutRTree(r: region, trajectories: list) -> list:
    # List to store the found trajectories 
    found_trajectories = []
 # Iterate through each trajectory 
    for traj in trajectories:
        # Iterate points of  each trajectory 
        for traj_point in traj.points:
            # include the trajectory information for each point
            traj_point.trajectory = traj

         # Check if the current point lies within the given region 'r
            if r.pointInRegion(traj_point):
                # Add found trajectory to the list 
                found_trajectories.append(traj)
                # Mark all points of the trajectory as visited
                for point in traj.points:
                    point.visited = True
                break  

    return found_trajectories

# Calculate the execution time of the query function.
def calculate_query_time(query_function, *args, **kwargs):
    start_time = time.time()
    query_result = query_function(*args, **kwargs)
    end_time = time.time()

    query_time = end_time - start_time
    return query_result, query_time