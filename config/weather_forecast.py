import requests
import json
import sys

def get_weather(city, api_key):
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
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    print(f"\nWeather in {weather_data['city']}, {weather_data['country']}:")
    print(f"Temperature: {weather_data['temp']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 2:
        print("Usage: python weather_forecast.py <city_name>")
        print("Example: python weather_forecast.py London")
        sys.exit(1)
    
    city = ' '.join(sys.argv[1:])
    api_key = "YOUR_API_KEY_HERE"
    
    if api_key == "YOUR_API_KEY_HERE":
        print("Please replace 'YOUR_API_KEY_HERE' with your actual OpenWeatherMap API key")
        print("Get a free API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    weather_data = get_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()