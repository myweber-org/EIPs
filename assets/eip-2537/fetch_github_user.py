
import requests
import time

def fetch_github_user(username):
    """
    Fetch public information for a GitHub user.
    Handles rate limiting and common HTTP errors.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-GitHub-Client"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"Error: User '{username}' not found on GitHub.")
            return None
        elif response.status_code == 403:
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            wait_time = max(0, reset_time - int(time.time()))
            print(f"Rate limit exceeded. Please wait {wait_time} seconds.")
            return None
        else:
            print(f"HTTP Error {response.status_code}: {response.reason}")
            return None
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

def display_user_info(user_data):
    """Display formatted user information."""
    if not user_data:
        return
    
    print(f"\nGitHub User: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        user_data = fetch_github_user(username)
        display_user_info(user_data)
    else:
        print("No username provided.")