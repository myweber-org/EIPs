
import requests
import json
import sys
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            return
        
        main = weather_data.get('main', {})
        weather = weather_data.get('weather', [{}])[0]
        sys = weather_data.get('sys', {})
        
        print(f"Weather Report for {weather_data.get('name', 'Unknown')}")
        print(f"Country: {sys.get('country', 'N/A')}")
        print(f"Temperature: {main.get('temp', 'N/A')}°C")
        print(f"Feels like: {main.get('feels_like', 'N/A')}°C")
        print(f"Humidity: {main.get('humidity', 'N/A')}%")
        print(f"Pressure: {main.get('pressure', 'N/A')} hPa")
        print(f"Weather: {weather.get('description', 'N/A')}")
        print(f"Wind Speed: {weather_data.get('wind', {}).get('speed', 'N/A')} m/s")
        print(f"Visibility: {weather_data.get('visibility', 'N/A')} meters")
        print(f"Sunrise: {datetime.fromtimestamp(sys.get('sunrise', 0)).strftime('%H:%M:%S')}")
        print(f"Sunset: {datetime.fromtimestamp(sys.get('sunset', 0)).strftime('%H:%M:%S')}")

def main():
    api_key = "your_api_key_here"
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    fetcher = WeatherFetcher(api_key)
    
    weather_data = fetcher.get_weather(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        
        with open(f"weather_{city_name.replace(' ', '_')}.json", 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nData saved to weather_{city_name.replace(' ', '_')}.json")
    else:
        print("Failed to retrieve weather data.")

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
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description.capitalize()}")
    else:
        error_message = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Could not retrieve weather data. Error: {error_message}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests

def get_weather_data(api_key, city_name):
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
            return None
        
        weather_info = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data:
        print(f"Weather in {weather_data['city']}:")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Conditions: {weather_data['description']}")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    else:
        print("Unable to retrieve weather data.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather_data(API_KEY, CITY)
    display_weather(weather)
import requests
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch weather data from OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        """Initialize the fetcher with an API key."""
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch current weather data for a given city.
        
        Args:
            city: Name of the city
            country_code: Optional country code (e.g., 'US', 'GB')
        
        Returns:
            Dictionary containing weather data
        
        Raises:
            ValueError: If city is empty
            ConnectionError: If API request fails
        """
        if not city or not city.strip():
            raise ValueError("City name cannot be empty")
        
        query = city.strip()
        if country_code:
            query += f",{country_code.strip()}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather data: {str(e)}")
        except json.JSONDecodeError as e:
            raise ConnectionError(f"Invalid response from API: {str(e)}")
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw API response into a structured format."""
        main = raw_data.get('main', {})
        weather = raw_data.get('weather', [{}])[0]
        wind = raw_data.get('wind', {})
        
        return {
            'city': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', 'Unknown'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'weather_condition': weather.get('description', 'Unknown'),
            'weather_icon': weather.get('icon'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'visibility': raw_data.get('visibility'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'sunrise': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunset')),
            'timestamp': datetime.now().isoformat(),
            'raw_data': raw_data
        }
    
    def _timestamp_to_datetime(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to ISO format string."""
        if timestamp:
            return datetime.fromtimestamp(timestamp).isoformat()
        return None
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        """Display weather data in a readable format."""
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Condition: {weather_data['weather_condition'].title()}")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Wind: {weather_data['wind_speed']} m/s")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Sunrise: {weather_data['sunrise']}")
        print(f"  Sunset: {weather_data['sunset']}")

def main():
    """Example usage of the WeatherFetcher class."""
    # Note: In production, use environment variables for API keys
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    try:
        weather = fetcher.get_weather("London", "GB")
        fetcher.display_weather(weather)
        
        # Save to file
        with open('weather_data.json', 'w') as f:
            json.dump(weather, f, indent=2)
        print("\nWeather data saved to 'weather_data.json'")
        
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()