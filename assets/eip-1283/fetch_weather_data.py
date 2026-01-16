
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
        exit(1)