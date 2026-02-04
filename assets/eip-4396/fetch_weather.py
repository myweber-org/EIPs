
import requests
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
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
            
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Weather data fetched for {city}")
            return weather_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            raise

def save_weather_data(data: Dict[str, Any], filename: str = "weather_data.json"):
    try:
        with open(filename, 'a') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        logger.info(f"Weather data saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save weather data: {e}")
        raise

def main():
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    all_weather_data = []
    
    for city, country in cities:
        try:
            weather_data = fetcher.get_weather(city, country)
            all_weather_data.append(weather_data)
            print(f"{city}, {country}: {weather_data['temperature']}°C, {weather_data['description']}")
        except Exception as e:
            print(f"Failed to get weather for {city}: {e}")
    
    if all_weather_data:
        save_weather_data(all_weather_data, "weather_history.json")

if __name__ == "__main__":
    main()