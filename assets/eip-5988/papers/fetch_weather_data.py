import requests
import json

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
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description}")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_data = get_weather(API_KEY, city_name)
    display_weather(weather_data)import requests
import os

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
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
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    api_key = "your_api_key_here"  # Replace with actual API key
    city = "London"
    weather = get_weather(city, api_key)
    
    if weather:
        print(f"Weather in {weather['city']}, {weather['country']}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Feels like: {weather['feels_like']}°C")
        print(f"Weather: {weather['weather']}")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Wind Speed: {weather['wind_speed']} m/s")import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_current_weather(self, city_name, country_code=None):
        """Fetch current weather data for a given city."""
        query = city_name
        if country_code:
            query += f",{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        """Parse and structure raw weather data."""
        if raw_data.get('cod') != 200:
            return None
            
        return {
            'location': {
                'city': raw_data['name'],
                'country': raw_data['sys']['country'],
                'coordinates': {
                    'lon': raw_data['coord']['lon'],
                    'lat': raw_data['coord']['lat']
                }
            },
            'weather': {
                'main': raw_data['weather'][0]['main'],
                'description': raw_data['weather'][0]['description'],
                'icon': raw_data['weather'][0]['icon']
            },
            'temperature': {
                'current': raw_data['main']['temp'],
                'feels_like': raw_data['main']['feels_like'],
                'min': raw_data['main']['temp_min'],
                'max': raw_data['main']['temp_max'],
                'humidity': raw_data['main']['humidity']
            },
            'wind': {
                'speed': raw_data['wind']['speed'],
                'direction': raw_data['wind'].get('deg', 0)
            },
            'visibility': raw_data.get('visibility', 0),
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'sun_times': {
                'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(raw_data['sys']['sunset']).isoformat()
            }
        }
    
    def format_weather_report(self, weather_data):
        """Format weather data into a readable report."""
        if not weather_data:
            return "Weather data unavailable."
        
        report = []
        report.append(f"Weather Report for {weather_data['location']['city']}, {weather_data['location']['country']}")
        report.append("=" * 50)
        report.append(f"Current: {weather_data['weather']['description'].title()}")
        report.append(f"Temperature: {weather_data['temperature']['current']}°C (Feels like: {weather_data['temperature']['feels_like']}°C)")
        report.append(f"Range: {weather_data['temperature']['min']}°C to {weather_data['temperature']['max']}°C")
        report.append(f"Humidity: {weather_data['temperature']['humidity']}%")
        report.append(f"Wind: {weather_data['wind']['speed']} m/s at {weather_data['wind']['direction']}°")
        report.append(f"Visibility: {weather_data['visibility']} meters")
        report.append(f"Sunrise: {weather_data['sun_times']['sunrise']}")
        report.append(f"Sunset: {weather_data['sun_times']['sunset']}")
        report.append(f"Last Updated: {weather_data['timestamp']}")
        
        return "\n".join(report)

def main():
    # Example usage
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Fetch weather for London
    weather_data = fetcher.get_current_weather("London", "UK")
    
    if weather_data:
        print(fetcher.format_weather_report(weather_data))
        
        # Save to JSON file
        with open('weather_data.json', 'w') as f:
            json.dump(weather_data, f, indent=2)
        logger.info("Weather data saved to weather_data.json")
    else:
        logger.error("Failed to retrieve weather data")

if __name__ == "__main__":
    main()import requests
import json
import os

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    return response.json()

def display_weather(data):
    if data.get("cod") != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    main = data["main"]
    weather = data["weather"][0]
    print(f"City: {data['name']}")
    print(f"Temperature: {main['temp']}°C")
    print(f"Weather: {weather['description']}")
    print(f"Humidity: {main['humidity']}%")
    print(f"Pressure: {main['pressure']} hPa")

if __name__ == "__main__":
    api_key = os.environ.get("OWM_API_KEY")
    if not api_key:
        print("Please set the OWM_API_KEY environment variable.")
        exit(1)

    city = input("Enter city name: ")
    weather_data = get_weather(city, api_key)
    display_weather(weather_data)