import math

def calculate_circle_area(radius):
    """
    Calculate the area of a circle.
    
    Args:
        radius (float): Radius of the circle
        
    Returns:
        float: Area of the circle
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    
    area = math.pi * (radius ** 2)
    return area

def main():
    # Example usage
    try:
        radius = 5.0
        area = calculate_circle_area(radius)
        print(f"Area of circle with radius {radius} is {area:.2f}")
        
        # Test with negative radius
        calculate_circle_area(-1)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()