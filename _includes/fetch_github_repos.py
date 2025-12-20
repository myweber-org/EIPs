import requests
import sys

def fetch_user_repos(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    
    print(f"Found {len(repos)} repositories:")
    for repo in repos:
        print(f"- {repo['name']}")
        print(f"  Description: {repo.get('description', 'No description')}")
        print(f"  Language: {repo.get('language', 'Not specified')}")
        print(f"  Stars: {repo['stargazers_count']}")
        print(f"  URL: {repo['html_url']}")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_user_repos(username, page, per_page)
    if repos is not None:
        display_repos(repos)

if __name__ == "__main__":
    main()