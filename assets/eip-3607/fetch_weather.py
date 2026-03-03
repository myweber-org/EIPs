import requests
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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to get weather data: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import time
from datetime import datetime, timedelta
import os

class WeatherFetcher:
    def __init__(self, api_key, cache_dir='weather_cache'):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_path(self, city):
        safe_name = city.lower().replace(' ', '_')
        return os.path.join(self.cache_dir, f"{safe_name}.json")
    
    def _is_cache_valid(self, cache_path):
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                cached_time = datetime.fromtimestamp(data.get('timestamp', 0))
                return datetime.now() - cached_time < timedelta(minutes=30)
        except (json.JSONDecodeError, KeyError):
            return False
    
    def fetch_weather(self, city, use_cache=True):
        cache_path = self._get_cache_path(city)
        
        if use_cache and self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    print(f"Using cached data for {city}")
                    return cached_data['weather_data']
            except (json.JSONDecodeError, KeyError, IOError):
                pass
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            cache_entry = {
                'timestamp': time.time(),
                'weather_data': weather_data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            
            return weather_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except json.JSONDecodeError:
            print("Error parsing weather data")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
        
        try:
            city = weather_data.get('name', 'Unknown')
            temp = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            description = weather_data['weather'][0]['description']
            
            print(f"Weather in {city}:")
            print(f"  Temperature: {temp}°C")
            print(f"  Humidity: {humidity}%")
            print(f"  Conditions: {description}")
            
        except KeyError as e:
            print(f"Invalid weather data format: missing key {e}")

def main():
    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        print("Please set WEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.fetch_weather(city)
        fetcher.display_weather(weather_data)
        time.sleep(1)

if __name__ == "__main__":
    main()