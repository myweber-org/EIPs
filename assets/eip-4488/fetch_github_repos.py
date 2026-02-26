import requests
import sys

def fetch_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {"page": page, "per_page": per_page}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        name = repo.get("name", "N/A")
        description = repo.get("description", "No description")
        stars = repo.get("stargazers_count", 0)
        print(f"{name} - {description} (Stars: {stars})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_repositories(username, page, per_page)
    if repos:
        display_repos(repos)

if __name__ == "__main__":
    main()