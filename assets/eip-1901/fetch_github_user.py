
import requests

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'location': user_data.get('location'),
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"User '{username}' not found."
        else:
            return f"HTTP Error occurred: {e}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    result = fetch_github_user(username)
    if isinstance(result, dict):
        print(f"\nGitHub Profile: {result['html_url']}")
        print(f"Name: {result['name']}")
        print(f"Public Repositories: {result['public_repos']}")
        print(f"Followers: {result['followers']}")
        print(f"Following: {result['following']}")
        print(f"Location: {result['location']}")
    else:
        print(result)