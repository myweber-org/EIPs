import requests
import sys

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
        return len(repos)
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    count = get_github_repos(username)
    print(f"Total repositories found: {count}")