import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_current_weather(self, city_name, country_code=None):
        location = f"{city_name},{country_code}" if country_code else city_name
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 'N/A'),
                'visibility': data.get('visibility', 'N/A'),
                'cloudiness': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Weather data fetched for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching weather data: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Data parsing error: {e}")
            return None

    def save_to_json(self, data, filename=None):
        if not data:
            self.logger.warning("No data to save")
            return False
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"weather_data_{data['city']}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            self.logger.error(f"Error saving to file: {e}")
            return False

def main():
    API_KEY = "your_api_key_here"  # Replace with actual API key
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    for city in cities:
        weather_data = fetcher.get_current_weather(city)
        if weather_data:
            print(f"\nWeather in {weather_data['city']}, {weather_data['country']}:")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Conditions: {weather_data['weather']} - {weather_data['description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind: {weather_data['wind_speed']} m/s")
            
            fetcher.save_to_json(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()