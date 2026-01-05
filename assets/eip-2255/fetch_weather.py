
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
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        return f"Weather in {city}: {temp}°C, {weather_desc}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except KeyError:
        return "Error parsing weather data."

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)
    api_key = sys.argv[1]
    city = sys.argv[2]
    result = get_weather(api_key, city)
    print(result)