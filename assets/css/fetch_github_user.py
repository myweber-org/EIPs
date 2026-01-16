
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
        print(f"Username: {user_data['login']}")
        print(f"Name: {user_data.get('name', 'N/A')}")
        print(f"Public Repos: {user_data['public_repos']}")
    else:
        print("User not found.")