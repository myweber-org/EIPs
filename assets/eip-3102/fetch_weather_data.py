
import requests
import os
from datetime import datetime

def get_weather_data(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str): OpenWeatherMap API key. If None, uses environment variable.
    
    Returns:
        dict: Weather data including temperature, humidity, description, etc.
    
    Raises:
        ValueError: If API key is not provided or city not found
        requests.exceptions.RequestException: If API request fails
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('cod') != 200:
            raise ValueError(f"City not found: {city_name}")
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 0),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']),
            'timestamp': datetime.fromtimestamp(data['dt'])
        }
        
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Failed to fetch weather data: {str(e)}")

def display_weather_info(weather_data):
    """
    Display weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data dictionary from get_weather_data()
    """
    if not weather_data:
        print("No weather data available")
        return
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print(f"Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise'].strftime('%H:%M')}")
    print(f"Sunset: {weather_data['sunset'].strftime('%H:%M')}")
    print(f"Last updated: {weather_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            print("Please set OPENWEATHER_API_KEY environment variable")
            exit(1)
        
        city = input("Enter city name: ").strip()
        if not city:
            city = "London"
        
        weather = get_weather_data(city, api_key)
        display_weather_info(weather)
        
    except ValueError as e:
        print(f"Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")