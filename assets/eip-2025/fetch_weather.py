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
        
    print(f"\nWeather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Weather: {weather_data['weather'].title()}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        print("Example: python fetch_weather.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch and display weather information from OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        """Initialize the fetcher with an API key."""
        self.api_key = api_key
        self.last_fetch_time = None
        self.cached_data = None
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Fetch weather data for a given city."""
        query = city
        if country_code:
            query += f",{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            self.last_fetch_time = datetime.now()
            self.cached_data = data
            
            return self._format_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except json.JSONDecodeError:
            return {'error': 'Invalid response from server'}
        except KeyError as e:
            return {'error': f'Missing expected data: {str(e)}'}
    
    def _format_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the raw weather data into a more readable structure."""
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        
        return {
            'location': data.get('name'),
            'country': data.get('sys', {}).get('country'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'description': weather.get('description'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'timestamp': datetime.fromtimestamp(data.get('dt', 0))
        }
    
    def display_weather(self, weather_data: Dict[str, Any]) -> str:
        """Create a formatted string representation of weather data."""
        if 'error' in weather_data:
            return f"Error: {weather_data['error']}"
        
        return f"""
Weather Report for {weather_data['location']}, {weather_data['country']}
--------------------------------------------------
Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)
Conditions: {weather_data['description'].title()}
Humidity: {weather_data['humidity']}%
Pressure: {weather_data['pressure']} hPa
Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°
Updated: {weather_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
"""

def main():
    """Example usage of the WeatherFetcher class."""
    # Note: In real usage, get API key from environment variable or config file
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Example cities to fetch weather for
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        print(f"\nFetching weather for {city}, {country}...")
        weather = fetcher.get_weather(city, country)
        print(fetcher.display_weather(weather))

if __name__ == "__main__":
    main()