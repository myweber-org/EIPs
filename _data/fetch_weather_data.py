import requests
import os
from datetime import datetime

def get_weather(city_name, api_key):
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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        main = data['main']
        weather = data['weather'][0]
        sys = data['sys']
        print(f"Weather in {data['name']}, {sys['country']}:")
        print(f"  Condition: {weather['description'].title()}")
        print(f"  Temperature: {main['temp']}°C")
        print(f"  Feels like: {main['feels_like']}°C")
        print(f"  Humidity: {main['humidity']}%")
        print(f"  Pressure: {main['pressure']} hPa")
        print(f"  Wind Speed: {data['wind']['speed']} m/s")
        sunrise = datetime.fromtimestamp(sys['sunrise']).strftime('%H:%M:%S')
        sunset = datetime.fromtimestamp(sys['sunset']).strftime('%H:%M:%S')
        print(f"  Sunrise: {sunrise}")
        print(f"  Sunset: {sunset}")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to retrieve weather. Error: {error_msg}")

if __name__ == "__main__":
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        print("Please set the OPENWEATHER_API_KEY environment variable.")
        exit(1)
    city = input("Enter city name: ").strip()
    if city:
        weather_data = get_weather(city, api_key)
        display_weather(weather_data)
    else:
        print("City name cannot be empty.")import requests
import json
import os
from datetime import datetime

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
    weather_desc = data["weather"][0]["description"]
    temp = main["temp"]
    feels_like = main["feels_like"]
    humidity = main["humidity"]
    wind_speed = data["wind"]["speed"]
    city = data["name"]
    country = data["sys"]["country"]
    timestamp = data["dt"]
    date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    print(f"Weather in {city}, {country} at {date_time}:")
    print(f"  Condition: {weather_desc.capitalize()}")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("Please set the OPENWEATHER_API_KEY environment variable.")
        return

    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty.")
        return

    weather_data = get_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()