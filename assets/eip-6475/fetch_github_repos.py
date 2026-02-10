import requests
import sys

def fetch_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    try:
        response = requests.get(url)
        response.raise_for_status()
        repos = response.json()
        if not repos:
            print(f"No public repositories found for user '{username}'.")
            return
        print(f"Public repositories for '{username}':")
        for repo in repos:
            print(f"- {repo['name']}: {repo['description'] or 'No description'}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    username = sys.argv[1]
    fetch_github_repos(username)