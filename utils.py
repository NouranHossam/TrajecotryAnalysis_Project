# imports
import point
import trajectory
import math
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.colors as mcolors


"""Import a single trajectory from a file with the file format 
xCoordinate yCoordinate day hour ... (other attributes will not be imported).
Each trajectory should hold an unique number (id)."""
def importTrajectory(filename:str,number:int) -> trajectory:
    # Import
    data = np.loadtxt(filename, delimiter=' ',dtype=str)

    # Create trajectory
    currTrajectory = trajectory.trajectory(number)

    # Convert data into points
    for entry in data:
        # Create point
        x = float(entry[0])
        y = float(entry[1])
        day = entry[2]
        hour = entry[3]
        timestamp = day + ':' + hour
        newPoint = point.point(x,y,timestamp)
        currTrajectory.addPoint(newPoint)

    # Return trajectory
    return currTrajectory

"""Import the given set of 62 with indexes between 1 and 96 trajectories"""
def importTrajectories(foldername:str) -> list:
    listOfTrajectories = []
    for i in range(1,97):
        filename = foldername + '/extractedTrace' + str(i) + '.txt'

        if glob(filename):
            currTrajectory = importTrajectory(filename,i)
            listOfTrajectories.append(currTrajectory)
    return listOfTrajectories

"""Method to calculate the perpendicular distance between one point
and a segment defined by two points"""
def calculateDistance(point:point,p1:point,p2:point):
    if (p2.X == p1.X) and (p2.Y == p1.Y):
        d = 0
    else:
        m = (p2.Y - p1.Y)/(p2.X - p1.X)
        a = m
        b = -1
        c = - (m*p1.X - p1.Y)
        d = abs((a * point.X + b * point.Y + c)) / (math.sqrt(a * a + b * b))
    #print("Perpendicular distance is " + str(d)),d
    return d

"""Calculate euclidean distance between two given points"""
def pointDistance(p0:point,p1:point) -> float:
    dist = math.sqrt((p0.X-p1.X)**2+(p0.Y-p1.Y)**2)
    return dist

'''Slice a trajectory with stert index and end index'''
def sliceTrajectory(traj: trajectory, start_index, end_index, traj_number):
    new_traj = trajectory.trajectory(traj_number)

    for i in range(start_index, end_index):
        new_traj.addPoint(traj.points[i])

    return new_traj

'''Compound two trajectories'''
def uppendTrajectory(traj: trajectory, add_traj: trajectory, traj_number):
    new_traj = trajectory.trajectory(traj_number)

    for i in range(len(traj.points)):
        new_traj.addPoint(traj.points[i])


    for i in range(len(add_traj.points)):
        new_traj.addPoint(add_traj.points[i])

    return new_traj

'''Visulaize all Trajectories as an animation, cosidering their actual time stamps'''
def vizualizeAllTrajectories(traj_list: list):
    # Create the figure and axis for the animation
    fig, ax = plt.subplots()

    # Find the minimum and maximum bounds of all trajectories
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for traj in traj_list:
        x_vals = [point.X for point in traj.points]
        y_vals = [point.Y for point in traj.points]

        min_x = min(min_x, min(x_vals))
        max_x = max(max_x, max(x_vals))
        min_y = min(min_y, min(y_vals))
        max_y = max(max_y, max(y_vals))

    ax.set_xlim(min_x, max_x)  # Set x-axis limits based on min and max x values
    ax.set_ylim(min_y, max_y)  # Set y-axis limits based on min and max y values

    lines = []  # List to store the line objects for each trajectory

    # Generate a list of distinct colors for each trajectory
    num_trajectories = len(traj_list)
    colors = list(mcolors.CSS4_COLORS.values())
    unique_colors = colors[:num_trajectories]

    # Function to update the animation frame
    def update(frame):
        all_finished = True  # Flag to check if all trajectories have finished displaying their points
        for idx, line in enumerate(lines):
            trajectory = traj_list[idx]  # Get the trajectory corresponding to the current line

            # If the current frame is within the range of trajectory points, it means the trajectory is still ongoing.
            if frame < len(trajectory.points):
                all_finished = False # At least one trajectory has not finished displaying points
                current_timestamp = trajectory.points[frame].timestamp  # Get the current timestamp from the trajectory's points.
                current_timestamp = datetime.strptime(current_timestamp, '%Y-%m-%d:%H:%M:%S') # Convert the timestamp string to a datetime object for comparison.

                # Filter points that have timestamps earlier than or equal to the current frame's timestamp
                filtered_points = [point for point in trajectory.points if datetime.strptime(point.timestamp, '%Y-%m-%d:%H:%M:%S') <= current_timestamp]

                # Extract the X and Y data from the filtered points for updating the line's data.
                x_data = [point.X for point in filtered_points]
                y_data = [point.Y for point in filtered_points]
                line.set_data(x_data, y_data) # Update the line's data with the new X and Y coordinates to reflect the trajectory's appearance at the current frame.
        if all_finished:
            # Stop the animation when all trajectories have finished displaying points
            ani.event_source.stop()
        return lines

    # Create line objects for each trajectory with unique colors
    for idx, trajectory in enumerate(traj_list):
        line, = ax.plot([], [], lw=2, color=unique_colors[idx])
        lines.append(line)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(max(traj_list, key=lambda t: len(t.points)).points), interval=10, blit=True)

    # Set the plot background color to light grey
    ax.set_facecolor('lightgrey')
    plt.title("All Trajectories Animation")
    plt.show()


'''Vizualize trajectories after preprocessing'''
def compareTrajectories(orig_traj: trajectory, dp_traj: trajectory, sw_traj: trajectory):

    # set a frame
    x_coords = []
    y_coords = []
    traj_names = {0:'original trajectory', 1:'Douglas-Peucker trajectory', 2:'sliding-window trajectory'}
    traj_points = dict.fromkeys([0, 1, 2])
    marker_sizes = [100, 60, 30]
    color = ['blue', 'red', 'green']

    # iterate over trajectories
    for i, traj in enumerate([orig_traj, dp_traj, sw_traj]):
        # make a list with coordinates
        for point in traj.points:
            x_coords.append(point.X)
            y_coords.append(point.Y)

        # add to a new dictionary
        traj_points[i] = [x_coords, y_coords]

        x_coords = []
        y_coords = []

    # vizualize each trajectory
    for t in traj_names.keys():
        plt.scatter(traj_points[t][0], traj_points[t][1], label=traj_names[t], s=marker_sizes[t], c=color[t])
        plt.plot(traj_points[t][0], traj_points[t][1], c=color[t])
        plt.legend(loc="best", fontsize=10)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory{} comparison".format(orig_traj.number))

    plt.show()
    return

# Plot the trajectories and the line connecting the closest pair
def plot_trajectories_with_CPD(traj0, traj1, point_pair):
    # Unpack the point pair and their indices
    (i, j), (p0, p1) = point_pair

    # Extract x and y coordinates from the points
    x0, y0 = p0.X, p0.Y
    x1, y1 = p1.X, p1.Y

    # Plot the trajectories
    plt.plot([p.X for p in traj0.points], [p.Y for p in traj0.points], label="Trajectory 0", color='blue')
    plt.plot([p.X for p in traj1.points], [p.Y for p in traj1.points], label="Trajectory 1", color='green')

    # Plot the closest pair points
    plt.scatter([x0, x1], [y0, y1], color='red', label='Closest Pair')

    # Plot the line connecting the closest pair
    plt.plot([x0, x1], [y0, y1], color='red', linestyle='dotted', label='distance line')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title("Closest Pair Distance example")

    # Show the plot
    plt.show()

'''plot the optimal path of DTW of 2 trajectories'''
def plot_optimal_path(traj0, traj1, optimal_path):
    # Unpack the trajectory points
    traj0_points = [(p.X, p.Y) for p in traj0.points]
    traj1_points = [(p.X, p.Y) for p in traj1.points]

    # Extract the indices from the optimal path
    path_indices = [idx for idx, _ in optimal_path]
    traj0_indices = [i for i, _ in optimal_path]
    traj1_indices = [j for _, j in optimal_path]

    # Plot the trajectories
    plt.plot(*zip(*traj0_points), label="Trajectory 0", color='blue')
    plt.plot(*zip(*traj1_points), label="Trajectory 1", color='green')

    # Plot all the points on the trajectories
    plt.scatter([p.X for p in traj0.points], [p.Y for p in traj0.points], color='blue', marker='o', label="Trajectory 0 Points")
    plt.scatter([p.X for p in traj1.points], [p.Y for p in traj1.points], color='green', marker='o', label="Trajectory 1 Points")

    # Plot the optimal path
    for i in range(len(path_indices) - 1):
        idx0, idx1 = traj0_indices[i], traj1_indices[i]
        next_idx0, next_idx1 = traj0_indices[i+1], traj1_indices[i+1]
        plt.plot([traj0_points[idx0][0], traj1_points[idx1][0]], [traj0_points[idx0][1], traj1_points[idx1][1]], 'r')

        # Add lines connecting the points along the optimal path
        if idx0 != next_idx0 or idx1 != next_idx1:
            plt.plot([traj0_points[next_idx0][0], traj1_points[next_idx1][0]], [traj0_points[next_idx0][1], traj1_points[next_idx1][1]], 'r')

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("DTW example")

    # Show the plot
    plt.show()



#Visualize trajectories intersected with the query region (point, raduis)
def vizualizeAllTrajectoriesQueryregion(traj_list: list, center_point: point, radius,plot_extent):
    # Create lists to hold combined X and Y coordinates of all trajectories
    x_coords = []
    y_coords = []

# Iterate over trajectories and combine their coordinates
    for traj in traj_list:
        x_coords.extend([point.X for point in traj.points])
        y_coords.extend([point.Y for point in traj.points])

        # Set trajectory parameters
        color = np.random.rand(3,).reshape(1,-1)
        plt.scatter([point.X for point in traj.points], [point.Y for point in traj.points], c=color)
        plt.plot([point.X for point in traj.points], [point.Y for point in traj.points], c=color)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

# Display title 
    title_str = "Trajectories obtained from the Query:\n"
    for traj in traj_list:
        title_str += f"Traj № {traj.number}, "
    plt.title(title_str)
        

# Add the query region polygon
    query_region = plt.Circle((center_point.X, center_point.Y), radius, color='red', fill=False)
    plt.gca().add_patch(query_region)

    # Show all resulted trajectories 
    plt.scatter(x_coords, y_coords, c='b', marker='.', label='Trajectories')
    plt.xlim(center_point.X - plot_extent, center_point.X + plot_extent)
    plt.ylim(center_point.Y - plot_extent, center_point.Y + plot_extent)
    plt.show()



