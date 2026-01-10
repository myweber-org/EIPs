import requests
import os

def get_current_weather(city_name, api_key=None):
    """
    Fetches current weather data for a given city.
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

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
        raise Exception(f"Network error occurred: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Failed to parse weather data: {e}")

if __name__ == "__main__":
    try:
        weather = get_current_weather("London")
        print(f"Weather in {weather['city']}, {weather['country']}:")
        print(f"  Temperature: {weather['temperature']}°C")
        print(f"  Feels like: {weather['feels_like']}°C")
        print(f"  Conditions: {weather['weather']}")
        print(f"  Humidity: {weather['humidity']}%")
        print(f"  Wind Speed: {weather['wind_speed']} m/s")
    except Exception as e:
        print(f"Error: {e}")