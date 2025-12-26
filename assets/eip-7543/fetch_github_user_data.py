import requests
import sys

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(f"Username: {data.get('login')}")
        print(f"Name: {data.get('name', 'Not provided')}")
        print(f"Public Repos: {data.get('public_repos')}")
        print(f"Followers: {data.get('followers')}")
        print(f"Following: {data.get('following')}")
        print(f"Profile URL: {data.get('html_url')}")
    else:
        print(f"Error: User '{username}' not found or API request failed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_data.py <username>")
        sys.exit(1)
    fetch_github_user(sys.argv[1])