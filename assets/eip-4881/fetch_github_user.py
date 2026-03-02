import requests
import sys

def fetch_github_user(username):
    """Fetch basic information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user '{username}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    fetch_github_user(sys.argv[1])
import requests
import time
import json

def fetch_github_user(username, token=None):
    """
    Fetch GitHub user information with basic rate limit handling.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
        
        if remaining == 0:
            wait_time = reset_time - time.time()
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds.")
                time.sleep(wait_time + 1)
                return fetch_github_user(username, token)
        
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"error": "User not found"}
        elif e.response.status_code == 403:
            return {"error": "Rate limit exceeded. Try again later or use token."}
        else:
            return {"error": f"HTTP error: {e.response.status_code}"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def save_user_info(user_data, filename="user_info.json"):
    """
    Save user data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(user_data, f, indent=2)
    print(f"User info saved to {filename}")

if __name__ == "__main__":
    # Example usage
    username = "octocat"  # Default GitHub test user
    token = None  # Set your token here for higher rate limits
    
    print(f"Fetching info for GitHub user: {username}")
    user_info = fetch_github_user(username, token)
    
    if "error" not in user_info:
        print(f"Name: {user_info.get('name', 'N/A')}")
        print(f"Bio: {user_info.get('bio', 'N/A')}")
        print(f"Public Repos: {user_info.get('public_repos', 0)}")
        print(f"Followers: {user_info.get('followers', 0)}")
        save_user_info(user_info)
    else:
        print(f"Error: {user_info['error']}")