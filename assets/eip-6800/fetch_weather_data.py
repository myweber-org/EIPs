import requests
import json
import os

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str, optional): OpenWeatherMap API key. If not provided,
                                 will try to get from environment variable.
    
    Returns:
        dict: Weather data if successful, None otherwise
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            print("Error: API key not provided and OPENWEATHER_API_KEY environment variable not set")
            return None
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'  # Use metric units (Celsius)
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        weather_data = response.json()
        
        if weather_data.get('cod') != 200:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")
            return None
        
        return weather_data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data from OpenWeatherMap API
    """
    if weather_data is None:
        print("No weather data to display")
        return
    
    city = weather_data['name']
    country = weather_data['sys']['country']
    temp = weather_data['main']['temp']
    feels_like = weather_data['main']['feels_like']
    humidity = weather_data['main']['humidity']
    description = weather_data['weather'][0]['description']
    wind_speed = weather_data['wind']['speed']
    
    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {description.capitalize()}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    """
    Example usage of the weather fetching functionality.
    """
    # Example: Get weather for London
    city = "London"
    
    # Try to get API key from environment variable
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if api_key is None:
        print("Please set your OpenWeatherMap API key as OPENWEATHER_API_KEY environment variable")
        print("Example: export OPENWEATHER_API_KEY='your_api_key_here'")
        return
    
    print(f"Fetching weather data for {city}...")
    weather = get_weather(city, api_key)
    
    if weather:
        display_weather(weather)
    else:
        print("Failed to retrieve weather data")

if __name__ == "__main__":
    main()import requests
import json
import sys

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Conditions: {weather_data['weather'].title()}")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
        
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)