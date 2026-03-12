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
            return self._parse_current_weather(data)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch weather data for {city_name}: {e}")
            return None

    def get_forecast(self, city_name, days=5):
        endpoint = f"{self.base_url}/forecast"
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8
        }
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._parse_forecast(data)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch forecast for {city_name}: {e}")
            return None

    def _parse_current_weather(self, data):
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }

    def _parse_forecast(self, data):
        forecast_list = []
        for item in data['list']:
            forecast_list.append({
                'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                'temperature': item['main']['temp'],
                'weather': item['weather'][0]['description'],
                'humidity': item['main']['humidity']
            })
        return {
            'city': data['city']['name'],
            'country': data['city']['country'],
            'forecast': forecast_list
        }

    def save_to_file(self, data, filename):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Weather data saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save data to {filename}: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    city = "London"
    
    current_weather = fetcher.get_current_weather(city)
    if current_weather:
        print("Current Weather:")
        print(json.dumps(current_weather, indent=2))
        fetcher.save_to_file(current_weather, f"{city}_current_weather.json")
    
    forecast = fetcher.get_forecast(city, days=3)
    if forecast:
        print("\nWeather Forecast:")
        print(json.dumps(forecast, indent=2))
        fetcher.save_to_file(forecast, f"{city}_forecast.json")

if __name__ == "__main__":
    main()import requests
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
    if data is None:
        print("No data to display.")
        return
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description']
    wind_speed = data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {weather_desc}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        print("Example: python fetch_weather_data.py your_api_key_here London")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime

def get_weather(api_key, city):
    """
    Fetch current weather data for a given city.
    """
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
            return {'error': data.get('message', 'Unknown error')}
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {str(e)}'}
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return {'error': f'Data parsing error: {str(e)}'}

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)