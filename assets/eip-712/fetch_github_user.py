import requests

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'login': user_data.get('login'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'html_url': user_data.get('html_url')
        }
    else:
        return None

def display_user_info(user_info):
    """Display the fetched user information in a formatted way."""
    if user_info:
        print(f"GitHub User: {user_info['name']} (@{user_info['login']})")
        print(f"Profile URL: {user_info['html_url']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
    else:
        print("User not found or error fetching data.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    info = get_github_user_info(username)
    display_user_info(info)