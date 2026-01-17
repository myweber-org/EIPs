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
        
    def get_current_weather(self, city_name, units="metric"):
        endpoint = f"{self.base_url}/weather"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"].get("deg", "N/A"),
                "visibility": data.get("visibility", "N/A"),
                "cloudiness": data["clouds"]["all"],
                "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M:%S"),
                "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M:%S"),
                "timestamp": datetime.now().isoformat()
            }
            
            logging.info(f"Weather data fetched successfully for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            return {"error": f"Network error: {str(e)}"}
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return {"error": f"Data parsing error: {str(e)}"}
        except KeyError as e:
            logging.error(f"Missing expected data field: {e}")
            return {"error": f"Invalid response structure: {str(e)}"}
    
    def save_to_file(self, data, filename="weather_data.json"):
        try:
            with open(filename, "a") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            logging.info(f"Weather data saved to {filename}")
        except IOError as e:
            logging.error(f"File save error: {e}")
    
    def get_weather_forecast(self, city_name, days=5, units="metric"):
        endpoint = f"{self.base_url}/forecast"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units,
            "cnt": days * 8
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            forecast_data = response.json()
            
            processed_forecast = []
            for item in forecast_data["list"]:
                forecast_entry = {
                    "datetime": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "weather": item["weather"][0]["description"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"]
                }
                processed_forecast.append(forecast_entry)
            
            return {
                "city": forecast_data["city"]["name"],
                "country": forecast_data["city"]["country"],
                "forecast": processed_forecast
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Forecast fetch error: {e}")
            return {"error": f"Forecast fetch failed: {str(e)}"}

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Sydney"]
    
    for city in cities:
        weather_data = fetcher.get_current_weather(city)
        if "error" not in weather_data:
            print(f"\nCurrent weather in {city}:")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            
            fetcher.save_to_file(weather_data)
            
            forecast = fetcher.get_weather_forecast(city, days=3)
            if "error" not in forecast:
                print(f"\n3-day forecast for {city}:")
                for day in forecast["forecast"][:3]:
                    print(f"  {day['datetime']}: {day['temperature']}°C, {day['weather']}")

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch weather data from OpenWeatherMap API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
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
            query = f"{query},{country_code}"
        
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
        """Parse raw API response into structured weather data."""
        main = raw_data.get('main', {})
        weather = raw_data.get('weather', [{}])[0]
        wind = raw_data.get('wind', {})
        
        return {
            'city': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'weather_condition': weather.get('description'),
            'weather_icon': weather.get('icon'),
            'wind_speed': wind.get('speed'),
            'wind_direction': wind.get('deg'),
            'timestamp': datetime.fromtimestamp(raw_data.get('dt', 0)),
            'sunrise': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunrise', 0)),
            'sunset': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunset', 0))
        }
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        """Display weather data in a readable format."""
        if not weather_data:
            print("No weather data available")
            return
        
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Condition: {weather_data['weather_condition']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Wind: {weather_data['wind_speed']} m/s")
        print(f"Updated: {weather_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Example usage of the WeatherFetcher class."""
    # Note: Replace with your actual API key
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    try:
        weather = fetcher.get_weather("London", "GB")
        fetcher.display_weather(weather)
        
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()