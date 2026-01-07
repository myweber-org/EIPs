
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
            
            return {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        except KeyError as e:
            logging.error(f"Unexpected API response format: {e}")
            return None
    
    def save_to_file(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, "a") as f:
                    json.dump(weather_data, f)
                    f.write("\n")
                logging.info(f"Weather data saved to {filename}")
            except IOError as e:
                logging.error(f"Failed to save data: {e}")

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_current_weather(city)
        
        if weather:
            print(f"Temperature in {weather['city']}: {weather['temperature']}°C")
            print(f"Conditions: {weather['description']}")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()