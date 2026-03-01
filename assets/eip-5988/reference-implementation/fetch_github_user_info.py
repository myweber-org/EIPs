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
        return {
            'name': user_data.get('name'),
            'company': user_data.get('company'),
            'blog': user_data.get('blog'),
            'location': user_data.get('location'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following')
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None

def display_user_info(info):
    """
    Display the fetched user information in a readable format.
    """
    if not info:
        print("No user information to display.")
        return
    print("GitHub User Information:")
    print(f"  Name: {info.get('name', 'Not provided')}")
    print(f"  Company: {info.get('company', 'Not provided')}")
    print(f"  Blog/Website: {info.get('blog', 'Not provided')}")
    print(f"  Location: {info.get('location', 'Not provided')}")
    print(f"  Public Repositories: {info.get('public_repos', 0)}")
    print(f"  Followers: {info.get('followers', 0)}")
    print(f"  Following: {info.get('following', 0)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    user_info = get_github_user_info(username)
    display_user_info(user_info)