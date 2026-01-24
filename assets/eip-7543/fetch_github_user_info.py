import requests

def fetch_github_user_info(username):
    """
    Fetches public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "public_repos": user_data.get("public_repos"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "html_url": user_data.get("html_url")
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        info = fetch_github_user_info(username)
        if info:
            print(f"Username: {info['login']}")
            print(f"Name: {info['name']}")
            print(f"Public Repos: {info['public_repos']}")
            print(f"Followers: {info['followers']}")
            print(f"Following: {info['following']}")
            print(f"Profile URL: {info['html_url']}")
        else:
            print("Failed to fetch user information.")