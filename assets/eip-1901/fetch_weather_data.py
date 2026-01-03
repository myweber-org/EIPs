import requests

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

def display_weather(weather_data):
    if weather_data:
        city = weather_data['name']
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        humidity = weather_data['main']['humidity']
        print(f"Weather in {city}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {description}")
        print(f"  Humidity: {humidity}%")
    else:
        print("No weather data to display.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city = input("Enter city name: ")
    weather = get_weather(city, API_KEY)
    display_weather(weather)