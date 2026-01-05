
import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_current_weather(self, city_name, country_code=None):
        """Fetch current weather data for a given city."""
        try:
            query = city_name
            if country_code:
                query = f"{city_name},{country_code}"
                
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {city_name}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response for {city_name}: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        """Parse raw weather data into structured format."""
        try:
            parsed_data = {
                'city': raw_data.get('name'),
                'country': raw_data.get('sys', {}).get('country'),
                'temperature': raw_data.get('main', {}).get('temp'),
                'feels_like': raw_data.get('main', {}).get('feels_like'),
                'humidity': raw_data.get('main', {}).get('humidity'),
                'pressure': raw_data.get('main', {}).get('pressure'),
                'weather': raw_data.get('weather', [{}])[0].get('description'),
                'wind_speed': raw_data.get('wind', {}).get('speed'),
                'wind_direction': raw_data.get('wind', {}).get('deg'),
                'visibility': raw_data.get('visibility'),
                'cloudiness': raw_data.get('clouds', {}).get('all'),
                'sunrise': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunrise')),
                'sunset': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunset')),
                'timestamp': datetime.now().isoformat(),
                'data_source': 'OpenWeatherMap'
            }
            return parsed_data
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Failed to parse weather data: {e}")
            return None
    
    def _timestamp_to_datetime(self, timestamp):
        """Convert Unix timestamp to ISO format datetime string."""
        if timestamp:
            return datetime.fromtimestamp(timestamp).isoformat()
        return None
    
    def get_multiple_cities_weather(self, cities_list):
        """Fetch weather data for multiple cities."""
        results = {}
        for city_info in cities_list:
            if isinstance(city_info, dict):
                city = city_info.get('city')
                country = city_info.get('country')
            else:
                city = city_info
                country = None
            
            weather_data = self.get_current_weather(city, country)
            if weather_data:
                results[city] = weather_data
            else:
                results[city] = {'error': 'Failed to fetch data'}
        
        return results
    
    def save_weather_data(self, data, filename='weather_data.json'):
        """Save weather data to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Weather data saved to {filename}")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save weather data: {e}")
            return False

def main():
    """Example usage of the WeatherFetcher class."""
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities_to_check = [
        {'city': 'London', 'country': 'UK'},
        {'city': 'New York', 'country': 'US'},
        {'city': 'Tokyo', 'country': 'JP'},
        'Berlin',
        'Paris'
    ]
    
    logger.info("Fetching weather data for multiple cities...")
    weather_results = fetcher.get_multiple_cities_weather(cities_to_check)
    
    if weather_results:
        print("\nWeather Data Summary:")
        print("=" * 50)
        for city, data in weather_results.items():
            if 'error' not in data:
                print(f"\n{city}, {data.get('country', 'N/A')}:")
                print(f"  Temperature: {data.get('temperature')}°C")
                print(f"  Weather: {data.get('weather', 'N/A')}")
                print(f"  Humidity: {data.get('humidity')}%")
            else:
                print(f"\n{city}: {data['error']}")
        
        fetcher.save_weather_data(weather_results)
    else:
        print("Failed to fetch any weather data.")

if __name__ == "__main__":
    main()