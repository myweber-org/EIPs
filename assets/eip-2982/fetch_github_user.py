
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

if __name__ == "__main__":
    user_data = get_github_user("octocat")
    if user_data:
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
    else:
        print("User not found")