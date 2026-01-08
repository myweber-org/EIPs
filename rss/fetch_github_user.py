import requests
import time

def fetch_github_user(username, token=None):
    """
    Fetch public profile information for a given GitHub username.
    Includes basic error handling and respects GitHub's rate limits.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check rate limit headers
        remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
        if remaining == 0:
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time()))
            wait_time = max(reset_time - time.time(), 0)
            print(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds.")
            time.sleep(wait_time + 1)
            return fetch_github_user(username, token)
        
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {"error": f"User '{username}' not found on GitHub"}
        elif response.status_code == 403:
            return {"error": "Rate limit exceeded. Try again later or use authentication."}
        else:
            return {"error": f"HTTP error occurred: {str(e)}"}
    
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def display_user_info(user_data):
    """Display formatted user information from GitHub API response."""
    if "error" in user_data:
        print(f"Error: {user_data['error']}")
        return
    
    print(f"GitHub Profile: {user_data.get('html_url', 'N/A')}")
    print(f"Username: {user_data.get('login', 'N/A')}")
    print(f"Name: {user_data.get('name', 'N/A')}")
    print(f"Bio: {user_data.get('bio', 'N/A')}")
    print(f"Public Repos: {user_data.get('public_repos', 0)}")
    print(f"Followers: {user_data.get('followers', 0)}")
    print(f"Following: {user_data.get('following', 0)}")
    print(f"Created: {user_data.get('created_at', 'N/A')}")

if __name__ == "__main__":
    # Example usage
    username = input("Enter GitHub username: ").strip()
    if username:
        result = fetch_github_user(username)
        display_user_info(result)