
import requests
import sys
import os

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
        print(f"Error fetching weather data: {e}", file=sys.stderr)
        return None

def display_weather(weather_data):
    if weather_data is None:
        print("No weather data to display.")
        return
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        description = weather_data['weather'][0]['description']
        humidity = weather_data['main']['humidity']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Conditions: {description}")
        print(f"  Humidity: {humidity}%")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}", file=sys.stderr)

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        print("Please set the OPENWEATHER_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>", file=sys.stderr)
        sys.exit(1)

    city_name = ' '.join(sys.argv[1:])
    weather_data = get_weather(city_name, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()