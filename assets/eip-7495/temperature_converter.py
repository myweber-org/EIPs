def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def convert_temperature(value, unit):
    if unit.upper() == 'C':
        return celsius_to_fahrenheit(value), 'F'
    elif unit.upper() == 'F':
        return fahrenheit_to_celsius(value), 'C'
    else:
        raise ValueError("Unit must be 'C' or 'F'")

if __name__ == "__main__":
    try:
        temp = float(input("Enter temperature value: "))
        unit = input("Enter unit (C or F): ").strip()
        converted_temp, new_unit = convert_temperature(temp, unit)
        print(f"{temp}°{unit.upper()} = {converted_temp:.2f}°{new_unit}")
    except ValueError as e:
        print(f"Error: {e}")