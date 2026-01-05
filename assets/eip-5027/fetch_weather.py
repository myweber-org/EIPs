
import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name):
        try:
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {city_name}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse response for {city_name}: {e}")
            return None
    
    def _parse_weather_data(self, data):
        weather_info = {
            'city': data.get('name'),
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }
        return weather_info
    
    def save_to_file(self, weather_data, filename='weather_data.json'):
        if weather_data:
            try:
                with open(filename, 'a') as f:
                    json.dump(weather_data, f, indent=2)
                    f.write('\n')
                logging.info(f"Weather data saved to {filename}")
            except IOError as e:
                logging.error(f"Failed to save data to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        logging.info(f"Fetching weather for {city}")
        weather = fetcher.get_weather(city)
        
        if weather:
            print(f"Weather in {weather['city']}:")
            print(f"  Temperature: {weather['temperature']}°C")
            print(f"  Humidity: {weather['humidity']}%")
            print(f"  Conditions: {weather['description']}")
            print(f"  Wind Speed: {weather['wind_speed']} m/s")
            print(f"  Last Updated: {weather['timestamp']}")
            print("-" * 40)
            
            fetcher.save_to_file(weather)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()