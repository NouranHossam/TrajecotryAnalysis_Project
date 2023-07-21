# imports
import point
import region
import utils
import trajectory
import functions_template as functions
import os
import timeit
import numpy as np
import matplotlib.pyplot as plt

# Changes the current working directory to the directory of the currently executing script.
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Import trajectories
listOfTrajectories = utils.importTrajectories("Trajectories")

# Visualize trajectories
utils.vizualizeAllTrajectories(listOfTrajectories[:3])

# Simplify at least one of the trajectories with Douglas Peucker and/or Sliding Window Algorithm
traj = listOfTrajectories[25]
epsilon_dp = 0.0001
epsilon_sw = 0.00005

simplified_traj_dp = trajectory.trajectory(traj.number)
simplified_traj_dp = functions.douglasPeucker(traj, traj.number, epsilon_dp, simplified_traj_dp)

simplified_traj_sw = trajectory.trajectory(traj.number)
simplified_traj_sw = functions.slidingWindow(traj, epsilon_sw, simplified_traj_sw)

# Visualize original trajectory and its two simplifications
utils.compareTrajectories(traj, simplified_traj_dp, simplified_traj_sw)

# Calculate the distance between at least two trajectories with Closest-Pair-Distance and/or Dynamic Time Warping

    # Choose two trajectories for testing (example: first and second trajectory in the list)
traj0 = listOfTrajectories[0]
traj1 = listOfTrajectories[1]

    # Calculate the closest pair distance usnig Euclidean Distance
distance1, point_pair = functions.closestPairDistance(traj0, traj1)
print("Closest pair distance:", distance1)

# Plot the trajectories and the closest pair
utils.plot_trajectories_with_CPD(traj0, traj1, point_pair)

    # Calculate the DTW usnig Euclidean Distance
distance2, optimal_path = functions.dynamicTimeWarping(traj0, traj1)
print("DTW distance: ", distance2)

# Plot the trajectories and optimal path
utils.plot_optimal_path(traj0, traj1, optimal_path)

# Build R-tree with all given 62 trajectories

# Query the trajectories using the built R-tree and the region. Which trajectories lie in the given region?
# This query should return the trajectories with ids 43, 45, 50, 71, 83
queryRegion = region.region(point.point(0.0012601754558545508,0.0027251228043638775,0.0),0.00003)


foundTrajectories_WithoutRTree = functions.solveQueryWithoutRTree(queryRegion,listOfTrajectories)
if foundTrajectories_WithoutRTree != None:
    if len(foundTrajectories_WithoutRTree)==0:
        print("No trajectories match the query.")
    for foundTrajectory in foundTrajectories_WithoutRTree:
        print(foundTrajectory.number)

query_result, time_taken = functions.calculate_query_time(functions.solveQueryWithoutRTree, queryRegion, listOfTrajectories)
print("Time taken by solveQueryWithoutRTree:", time_taken, "seconds")


foundTrajectories_WithRTree = functions.solveQueryWithRTree(queryRegion,listOfTrajectories)
if foundTrajectories_WithRTree != None:
    if len(foundTrajectories_WithRTree)==0:
        print("No trajectories match the query.")
    for foundTrajectory in foundTrajectories_WithRTree:
        print(foundTrajectory.number)

query_result, time_taken = functions.calculate_query_time(functions.solveQueryWithRTree, queryRegion, listOfTrajectories)
print("Time taken by solveQueryWithRTree:", time_taken, "seconds")

plot_extent = 0.00005
region_polygon = point.point(0.0012601754558545508,0.0027251228043638775, 0.0)
utils.vizualizeAllTrajectoriesQueryregion(foundTrajectories_WithoutRTree, region_polygon,0.00003,plot_extent )
