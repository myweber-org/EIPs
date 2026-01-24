
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        
        if data["cod"] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        main_data = data["main"]
        weather_data = data["weather"][0]
        wind_data = data["wind"]
        
        weather_info = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": main_data["temp"],
            "feels_like": main_data["feels_like"],
            "humidity": main_data["humidity"],
            "pressure": main_data["pressure"],
            "weather": weather_data["main"],
            "description": weather_data["description"],
            "wind_speed": wind_data["speed"],
            "wind_direction": wind_data.get("deg", "N/A"),
            "timestamp": datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        return None

def display_weather(weather_info):
    if not weather_info:
        return
        
    print("\n" + "="*50)
    print(f"Weather in {weather_info['city']}, {weather_info['country']}")
    print("="*50)
    print(f"Current Time: {weather_info['timestamp']}")
    print(f"Temperature: {weather_info['temperature']}°C")
    print(f"Feels Like: {weather_info['feels_like']}°C")
    print(f"Weather: {weather_info['weather']} ({weather_info['description']})")
    print(f"Humidity: {weather_info['humidity']}%")
    print(f"Pressure: {weather_info['pressure']} hPa")
    print(f"Wind Speed: {weather_info['wind_speed']} m/s")
    if weather_info['wind_direction'] != "N/A":
        print(f"Wind Direction: {weather_info['wind_direction']}°")
    print("="*50)

def save_to_file(weather_info, filename="weather_log.json"):
    if not weather_info:
        return
        
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        existing_data.append(weather_info)
        
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Weather data saved to {filename}")
        
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error saving to file: {e}")

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set")
        print("Please set your API key: export OPENWEATHER_API_KEY='your_api_key_here'")
        return
        
    city_name = input("Enter city name: ").strip()
    
    if not city_name:
        print("City name cannot be empty")
        return
        
    print(f"Fetching weather data for {city_name}...")
    weather_info = get_weather(city_name, api_key)
    
    if weather_info:
        display_weather(weather_info)
        
        save_choice = input("\nSave this data to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_to_file(weather_info)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()
import requests
import json
import os

def get_weather(city_name, api_key=None):
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['cod'] != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
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
            'wind_direction': data['wind'].get('deg', 'N/A')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {str(e)}")

def display_weather(weather_data):
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)")
    print(f"  Conditions: {weather_data['weather']} - {weather_data['description']}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty")
            exit(1)
            
        weather = get_weather(city)
        display_weather(weather)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        
    def get_current_weather(self, city_name, country_code=None):
        """Fetch current weather data for a given city."""
        try:
            query = city_name
            if country_code:
                query += f",{country_code}"
                
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse response: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        """Parse and structure raw weather data."""
        return {
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'city': raw_data['name'],
            'country': raw_data['sys']['country'],
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['main'],
            'description': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 0),
            'visibility': raw_data.get('visibility', 0),
            'cloudiness': raw_data['clouds']['all']
        }
    
    def get_forecast(self, city_name, days=5):
        """Fetch weather forecast for multiple days."""
        try:
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day
            }
            
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            forecast_data = response.json()
            return self._parse_forecast_data(forecast_data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Forecast request failed: {e}")
            return None
    
    def _parse_forecast_data(self, raw_data):
        """Parse and structure forecast data."""
        forecasts = []
        for item in raw_data['list']:
            forecast = {
                'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                'temperature': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'humidity': item['main']['humidity'],
                'weather': item['weather'][0]['main'],
                'description': item['weather'][0]['description'],
                'wind_speed': item['wind']['speed'],
                'precipitation': item.get('rain', {}).get('3h', 0)
            }
            forecasts.append(forecast)
        
        return {
            'city': raw_data['city']['name'],
            'country': raw_data['city']['country'],
            'forecasts': forecasts
        }

def main():
    # Example usage
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Get current weather
    current = fetcher.get_current_weather("London", "GB")
    if current:
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Conditions: {current['description']}")
        print(f"Humidity: {current['humidity']}%")
    
    # Get forecast
    forecast = fetcher.get_forecast("London", days=3)
    if forecast:
        print(f"\n3-day forecast for {forecast['city']}:")
        for i, day in enumerate(forecast['forecasts'][::8], 1):  # Get one reading per day
            print(f"Day {i}: {day['temperature']}°C, {day['description']}")

if __name__ == "__main__":
    main()