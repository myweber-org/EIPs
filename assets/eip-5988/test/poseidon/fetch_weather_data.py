import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_current_weather(self, city_name):
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                logging.error(f"API Error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        parsed = {
            'city': raw_data['name'],
            'country': raw_data['sys']['country'],
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['main'],
            'description': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 'N/A'),
            'visibility': raw_data.get('visibility', 'N/A'),
            'cloudiness': raw_data['clouds']['all']
        }
        return parsed

    def save_to_file(self, data, filename="weather_data.json"):
        if data:
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                logging.info(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                logging.error(f"Failed to save file: {e}")
                return False
        return False

def main():
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    fetcher = WeatherFetcher(API_KEY)
    weather_data = fetcher.get_current_weather(CITY)
    
    if weather_data:
        print(json.dumps(weather_data, indent=2))
        fetcher.save_to_file(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()import requests

def get_weather(city, api_key):
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
        return {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity']
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = "London"
    weather_info = get_weather(city_name, API_KEY)
    if weather_info:
        print(f"Weather in {weather_info['city']}:")
        print(f"Temperature: {weather_info['temperature']}°C")
        print(f"Description: {weather_info['description']}")
        print(f"Humidity: {weather_info['humidity']}%")