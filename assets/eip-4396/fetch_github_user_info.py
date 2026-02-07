
import requests
import sys

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def display_user_info(user_data):
    """Display selected user information in a readable format."""
    if not user_data:
        print("No user data to display.")
        return

    print(f"GitHub User: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repositories: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Profile URL: {user_data.get('html_url')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    print(f"Fetching information for GitHub user: {username}")
    user_info = fetch_github_user(username)

    if user_info:
        display_user_info(user_info)
    else:
        print(f"Failed to fetch information for user '{username}'.")