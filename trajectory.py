# imports
import point
class trajectory():
    # Initialization method of trajectory with an unique id
    def __init__(self,number) -> None:
        self.number = number
        self.points = []

    def __repr__(self) -> str:
        # Nice printing of trajectory
        resultString = 'Trajectory with number: ' + str(self.number) + ' and points '
        for p in self.points:
            resultString += str(p) + ' '
        return resultString

    # Adds a point to the list of points of the trajectory
    def addPoint(self,p:point) -> None:
        p.trajectory_number = self.number
        self.points.append(p)