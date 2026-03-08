
import requests
import sys

def fetch_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        if repos:
            print(f"Repositories for user '{username}':")
            for repo in repos:
                print(f"- {repo['name']}: {repo['description'] or 'No description'}")
        else:
            print(f"No repositories found for user '{username}'.")
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    fetch_github_repos(username)