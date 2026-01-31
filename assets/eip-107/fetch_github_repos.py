import requests
import sys

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_names = [repo['name'] for repo in repos]
        return repo_names
    else:
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = get_github_repos(username)
    
    if repos is None:
        print(f"Failed to fetch repositories for user '{username}'")
        sys.exit(1)
    
    print(f"Repositories for {username}:")
    for repo in repos:
        print(f"  - {repo}")

if __name__ == "__main__":
    main()