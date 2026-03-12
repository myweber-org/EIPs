
import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def get_weather_by_city(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch weather data: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response: {e}")
            return None

    def _parse_weather_data(self, data):
        if not data or 'main' not in data:
            return None
            
        return {
            'city': data.get('name', 'Unknown'),
            'temperature': data['main'].get('temp'),
            'humidity': data['main'].get('humidity'),
            'pressure': data['main'].get('pressure'),
            'description': data['weather'][0].get('description') if data.get('weather') else 'N/A',
            'wind_speed': data['wind'].get('speed') if data.get('wind') else 0,
            'timestamp': datetime.fromtimestamp(data.get('dt', 0)).isoformat() if data.get('dt') else None
        }

    def save_to_file(self, weather_data, filename="weather_data.json"):
        if not weather_data:
            logging.warning("No weather data to save")
            return False
            
        try:
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            logging.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to save weather data: {e}")
            return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo"]
    
    for city in cities:
        logging.info(f"Fetching weather for {city}")
        weather = fetcher.get_weather_by_city(city)
        
        if weather:
            print(f"\nWeather in {weather['city']}:")
            print(f"  Temperature: {weather['temperature']}°C")
            print(f"  Humidity: {weather['humidity']}%")
            print(f"  Conditions: {weather['description']}")
            print(f"  Wind Speed: {weather['wind_speed']} m/s")
            
            filename = f"{city.lower().replace(' ', '_')}_weather.json"
            fetcher.save_to_file(weather, filename)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()
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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available.")
        return
    
    if weather_data.get('cod') != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
        return
    
    main = weather_data['main']
    weather = weather_data['weather'][0]
    
    print(f"Weather in {weather_data['name']}, {weather_data['sys']['country']}:")
    print(f"  Temperature: {main['temp']}°C")
    print(f"  Feels like: {main['feels_like']}°C")
    print(f"  Conditions: {weather['description'].capitalize()}")
    print(f"  Humidity: {main['humidity']}%")
    print(f"  Pressure: {main['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind']['speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()