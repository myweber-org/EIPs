
import requests
import sys

def get_user_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch repositories for user '{username}'")
        print(f"Status Code: {response.status_code}")
        return []
    
    repos = response.json()
    return repos

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    
    print(f"Found {len(repos)} repositories:")
    for repo in repos:
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        language = repo.get('language', 'Not specified')
        
        print(f"\nName: {name}")
        print(f"  Description: {description}")
        print(f"  Stars: {stars}, Forks: {forks}")
        print(f"  Language: {language}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    print(f"Fetching repositories for GitHub user: {username}")
    
    repos = get_user_repositories(username)
    display_repositories(repos)

if __name__ == "__main__":
    main()import requests
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    fetch_github_repos(username)