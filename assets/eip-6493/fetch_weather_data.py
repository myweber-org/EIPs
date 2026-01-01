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
    main()