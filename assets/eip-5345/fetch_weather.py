
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
            
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available.")
        return
        
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print("="*40)

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        print("Example: python fetch_weather.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather_by_city(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            return None

        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind']['deg'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S')
        }
        return weather_info

    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available.")
            return

        print("\n" + "="*50)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Last updated: {weather_data['timestamp']}")
        print("="*50)

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)

    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather_by_city(city)
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"Could not fetch weather for {city}")

if __name__ == "__main__":
    main()