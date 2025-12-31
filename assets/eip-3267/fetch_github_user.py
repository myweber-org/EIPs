import requests

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch user. Status code: {response.status_code}"}

if __name__ == "__main__":
    user_data = fetch_github_user("octocat")
    print(user_data)