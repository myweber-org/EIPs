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
    main()