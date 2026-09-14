from map_to_circle import *
import matplotlib.pyplot as plt
import math

def get_vector_angle_from_origin(start, end):
    """
    Calculates the angle between the vector (from start to end) 
    and the positive X-axis (origin).
    
    Parameters:
    start (tuple/list): (x, y) coordinates of the starting point
    end (tuple/list): (x, y) coordinates of the ending point
    
    Returns:
    dict: Angle in 'radians' and 'degrees' (ranging from -180 to 180 or 0 to 360)
    """
    # 1. Get the direction components of the line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # 2. Calculate the angle in radians using atan2(y, x)
    angle_radians = math.atan2(dy, dx)
    
    # 3. Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    
    # Optional: If you want 0 to 360 degrees instead of -180 to 180:
    angle_360 = angle_degrees if angle_degrees >= 0 else angle_degrees + 360
    
    return {
        "radians": angle_radians,
        "degrees": angle_degrees,
        "degrees_360": angle_360
    }
def plot_vector(start, end, color='blue', label=None):
    """
    Plots a 2D vector from a start point to an end point.
    
    Parameters:
    start (tuple/list): (x, y) coordinates of the starting point
    end (tuple/list): (x, y) coordinates of the ending point
    color (str): Color of the vector arrow
    label (str): Optional label for the legend
    """
    # 1. Unpack coordinates
    start_x, start_y = start
    end_x, end_y = end
    
    # 2. Calculate vector components (dx, dy)
    U = end_x - start_x
    V = end_y - start_y
    
    # 3. Plot the vector
    # angles='xy', scale_units='xy', scale=1 ensure the vector scales 1:1 with the grid
    plt.quiver(start_x, start_y, U, V, angles='xy', scale_units='xy', scale=1, 
               color=color, label=label)
number = 12

point = (number,number)
plot_vector((0, 0), point, color='red', label='Vector from Origin to Point')
angles = get_vector_angle_from_origin((0,0), point)

print(f"Angle in Degrees: {angles['degrees']}°")
plt.show()
print(number_to_polar(number))