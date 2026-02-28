import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

def display_user_info(user_data):
    if not user_data:
        print("No user data to display.")
        return
    print(f"Username: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    data = get_github_user(username)
    display_user_info(data)