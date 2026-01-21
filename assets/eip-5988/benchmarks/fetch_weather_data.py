
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch weather data from a public API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        """Initialize the weather fetcher with an API key."""
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_weather_by_city(self, city_name: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch weather data for a given city."""
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing weather data: {e}")
            return None
    
    def get_weather_by_coordinates(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetch weather data using latitude and longitude coordinates."""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing weather data: {e}")
            return None
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the raw weather API response."""
        return {
            'location': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', 'Unknown'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description', 'Unknown'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'visibility': raw_data.get('visibility'),
            'sunrise': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunset')),
            'timestamp': datetime.now().isoformat(),
            'data_source': 'OpenWeatherMap'
        }
    
    def _timestamp_to_datetime(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to ISO format datetime string."""
        if timestamp:
            return datetime.fromtimestamp(timestamp).isoformat()
        return None
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        """Display weather data in a readable format."""
        if not weather_data:
            print("No weather data available.")
            return
        
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['location']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather_description'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Report generated: {weather_data['timestamp']}")
        print("="*50)

def main():
    """Example usage of the WeatherFetcher class."""
    # Note: In production, use environment variables for API keys
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Example: Get weather by city
    print("Fetching weather for London, UK...")
    london_weather = fetcher.get_weather_by_city("London", "UK")
    fetcher.display_weather(london_weather)
    
    # Example: Get weather by coordinates (New York)
    print("\nFetching weather for New York coordinates...")
    ny_weather = fetcher.get_weather_by_coordinates(40.7128, -74.0060)
    fetcher.display_weather(ny_weather)
    
    # Example: Error handling
    print("\nAttempting to fetch weather for non-existent city...")
    invalid_weather = fetcher.get_weather_by_city("NonexistentCity123")
    fetcher.display_weather(invalid_weather)

if __name__ == "__main__":
    main()