import requests
import sys

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        if repos:
            print(f"Public repositories for {username}:")
            for repo in repos:
                print(f"- {repo['name']}: {repo['html_url']}")
        else:
            print(f"No public repositories found for {username}.")
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    fetch_github_repos(username)