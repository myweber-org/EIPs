
import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather_by_city(self, city_name, units='metric'):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except json.JSONDecodeError:
            return {'error': 'Invalid response from server'}
        except KeyError:
            return {'error': 'Unexpected data format'}

    def _parse_weather_data(self, data):
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        sys = data.get('sys', {})

        return {
            'city': data.get('name'),
            'country': sys.get('country'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'weather': weather.get('description'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'sunrise': datetime.fromtimestamp(sys.get('sunrise')).strftime('%H:%M:%S') if sys.get('sunrise') else None,
            'sunset': datetime.fromtimestamp(sys.get('sunset')).strftime('%H:%M:%S') if sys.get('sunset') else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def display_weather(self, weather_data):
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return

        print(f"\nWeather in {weather_data['city']}, {weather_data['country']}")
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Conditions: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Last updated: {weather_data['timestamp']}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        weather = fetcher.get_weather_by_city(city)
        fetcher.display_weather(weather)
        print("-" * 50)

if __name__ == "__main__":
    main()