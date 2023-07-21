# imports
import point
import utils

class region():
    # Initialization method of region
    def __init__(self,center:point.point,radius:float) -> None:
        self.center = center
        self.radius = radius
    
    # Checks if point lies in region
    def pointInRegion(self,queryPoint:point) -> bool:
        distance = utils.pointDistance(self.center, queryPoint)
        return distance <= self.radius
    