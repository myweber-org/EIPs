import requests
import sys

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'bio': user_data.get('bio'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following')
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching data: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_info.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    user_info = get_github_user_info(username)

    if user_info:
        print(f"GitHub User: {username}")
        print(f"Name: {user_info['name']}")
        print(f"Bio: {user_info['bio']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
    else:
        print("Failed to retrieve user information.")

if __name__ == "__main__":
    main()