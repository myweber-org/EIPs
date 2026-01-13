
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
    if weather_data is None:
        print("No weather data to display.")
        return
    if weather_data.get('cod') != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
        return

    city = weather_data['name']
    country = weather_data['sys']['country']
    temp = weather_data['main']['temp']
    feels_like = weather_data['main']['feels_like']
    humidity = weather_data['main']['humidity']
    description = weather_data['weather'][0]['description']
    wind_speed = weather_data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {description}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        print("Example: python fetch_weather_data.py your_api_key_here London")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
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
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Data parsing error: {e}")
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
            
            forecasts = []
            for item in data['list'][:days]:
                forecasts.append({
                    'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d'),
                    'temp': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'weather': item['weather'][0]['main']
                })
            
            return forecasts
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Forecast request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Forecast data parsing error: {e}")
            return None

def save_weather_data(data, filename):
    if data:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Weather data saved to {filename}")
        return True
    return False

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    fetcher = WeatherFetcher(API_KEY)
    
    current = fetcher.get_current_weather(CITY)
    if current:
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Humidity: {current['humidity']}%")
        print(f"Conditions: {current['description']}")
        save_weather_data(current, f"{CITY.lower()}_current.json")
    
    forecast = fetcher.get_forecast(CITY, 3)
    if forecast:
        print(f"\n3-day forecast for {CITY}:")
        for day in forecast:
            print(f"{day['date']}: {day['temp']}°C, {day['weather']}")
        save_weather_data(forecast, f"{CITY.lower()}_forecast.json")