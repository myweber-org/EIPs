
import requests
import json
import sys

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return None

def save_to_json(data, filename):
    if data:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")
    else:
        print("No data to save", file=sys.stderr)

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>", file=sys.stderr)
        sys.exit(1)
    
    username = sys.argv[1]
    user_data = fetch_github_user(username)
    
    if user_data:
        filename = f"{username}_github_data.json"
        save_to_json(user_data, filename)
        print(f"Successfully fetched data for {username}")
    else:
        print(f"Failed to fetch data for {username}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

if __name__ == "__main__":
    user_data = get_github_user("octocat")
    if user_data:
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
    else:
        print("User not found")