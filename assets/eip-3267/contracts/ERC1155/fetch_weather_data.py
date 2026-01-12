import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import os

class WeatherFetcher:
    def __init__(self, api_key: str, cache_dir: str = "./weather_cache"):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=1)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_key(self, city: str, country_code: str) -> str:
        identifier = f"{city.lower()}_{country_code.lower()}"
        return hashlib.md5(identifier.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _is_cache_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_mtime < self.cache_duration

    def _read_from_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _write_to_cache(self, cache_path: str, data: Dict[str, Any]) -> None:
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass

    def fetch_weather(self, city: str, country_code: str = "us") -> Dict[str, Any]:
        cache_key = self._get_cache_key(city, country_code)
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            cached_data = self._read_from_cache(cache_path)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data

        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'city': data.get('name'),
                'country': data.get('sys', {}).get('country'),
                'temperature': data.get('main', {}).get('temp'),
                'humidity': data.get('main', {}).get('humidity'),
                'description': data.get('weather', [{}])[0].get('description'),
                'wind_speed': data.get('wind', {}).get('speed'),
                'timestamp': datetime.now().isoformat(),
                'source': 'api'
            }
            
            self._write_to_cache(cache_path, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'city': city,
                'country': country_code,
                'timestamp': datetime.now().isoformat(),
                'source': 'error'
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                'error': f"Data parsing error: {str(e)}",
                'city': city,
                'country': country_code,
                'timestamp': datetime.now().isoformat(),
                'source': 'error'
            }

def display_weather(data: Dict[str, Any]) -> None:
    print("\n" + "="*50)
    if 'error' in data:
        print(f"Error fetching weather: {data['error']}")
        return
    
    print(f"Weather for {data['city']}, {data['country']}")
    print(f"Source: {data['source']}")
    print(f"Time: {data['timestamp']}")
    print("-"*50)
    print(f"Temperature: {data['temperature']}°C")
    print(f"Humidity: {data['humidity']}%")
    print(f"Conditions: {data['description']}")
    print(f"Wind Speed: {data['wind_speed']} m/s")
    print("="*50)

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "uk"),
        ("New York", "us"),
        ("Tokyo", "jp"),
        ("Paris", "fr")
    ]
    
    for city, country in cities:
        weather_data = fetcher.fetch_weather(city, country)
        display_weather(weather_data)
        time.sleep(1)

if __name__ == "__main__":
    main()