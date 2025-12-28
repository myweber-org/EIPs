import requests
import sys

def get_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        
        info = {
            'name': user_data.get('name', 'N/A'),
            'company': user_data.get('company', 'N/A'),
            'blog': user_data.get('blog', 'N/A'),
            'location': user_data.get('location', 'N/A'),
            'public_repos': user_data.get('public_repos', 0),
            'followers': user_data.get('followers', 0),
            'following': user_data.get('following', 0)
        }
        return info
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None

def display_user_info(username, info):
    """
    Display the fetched user information in a formatted way.
    """
    if info is None:
        print(f"Failed to retrieve information for user '{username}'.")
        return
    
    print(f"\nGitHub User: {username}")
    print("=" * 40)
    print(f"Name: {info['name']}")
    print(f"Company: {info['company']}")
    print(f"Blog/Website: {info['blog']}")
    print(f"Location: {info['location']}")
    print(f"Public Repositories: {info['public_repos']}")
    print(f"Followers: {info['followers']}")
    print(f"Following: {info['following']}")
    print("=" * 40)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_info = get_github_user_info(username)
    display_user_info(username, user_info)