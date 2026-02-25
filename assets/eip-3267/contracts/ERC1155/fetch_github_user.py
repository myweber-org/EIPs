import requests

def fetch_github_user(username):
    """Fetch basic information for a GitHub user."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'login': data.get('login'),
            'name': data.get('name'),
            'public_repos': data.get('public_repos', 0),
            'followers': data.get('followers', 0),
            'following': data.get('following', 0)
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def display_user_info(user_data):
    """Display user information in a formatted way."""
    if user_data:
        print(f"Username: {user_data['login']}")
        print(f"Name: {user_data['name']}")
        print(f"Public Repositories: {user_data['public_repos']}")
        print(f"Followers: {user_data['followers']}")
        print(f"Following: {user_data['following']}")
    else:
        print("No user data to display.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    user_info = fetch_github_user(username)
    display_user_info(user_info)