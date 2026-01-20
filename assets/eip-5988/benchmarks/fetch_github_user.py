import requests
import sys

def fetch_github_user(username):
    """
    Fetch public profile information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def display_user_info(user_data):
    """
    Display selected fields from the user data.
    """
    if not user_data:
        print("No user data to display.")
        return

    fields = ['login', 'name', 'company', 'blog', 'location', 'email', 'public_repos', 'followers', 'following']
    for field in fields:
        value = user_data.get(field)
        if value:
            print(f"{field.capitalize()}: {value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    user_data = fetch_github_user(username)

    if user_data:
        display_user_info(user_data)
    else:
        print(f"Failed to fetch data for user '{username}'.")