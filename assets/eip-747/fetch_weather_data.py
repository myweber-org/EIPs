import requests
import os

def get_current_weather(city_name, api_key=None):
    """
    Fetches current weather data for a given city from OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for.
        api_key (str, optional): OpenWeatherMap API key. If not provided,
                                 will try to get from environment variable.
    
    Returns:
        dict: Weather data if successful, None otherwise.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            print("Error: API key not provided and OPENWEATHER_API_KEY environment variable not set.")
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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    """
    Displays weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data from OpenWeatherMap API.
    """
    if weather_data is None:
        print("No weather data available.")
        return
    
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        wind_speed = weather_data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Conditions: {description}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    except KeyError as e:
        print(f"Error parsing weather data: Missing key {e}")

if __name__ == "__main__":
    # Example usage
    city = "London"
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    weather = get_current_weather(city, api_key)
    display_weather(weather)import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
        except json.JSONDecodeError:
            return "Error parsing weather data"

    def _parse_weather_data(self, data):
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind']['deg'],
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    def display_weather(self, weather_data):
        if isinstance(weather_data, dict):
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
            print(f"  Conditions: {weather_data['weather'].title()}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Pressure: {weather_data['pressure']} hPa")
            print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
            print(f"  Visibility: {weather_data['visibility']} meters")
            print(f"  Cloudiness: {weather_data['cloudiness']}%")
            print(f"  Sunrise: {weather_data['sunrise']}")
            print(f"  Sunset: {weather_data['sunset']}")
            print(f"  Last updated: {weather_data['timestamp']}")
        else:
            print(weather_data)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print("\n" + "="*50)
        weather = fetcher.get_weather(city)
        fetcher.display_weather(weather)

if __name__ == "__main__":
    main()