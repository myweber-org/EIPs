import requests
import sys

def fetch_repositories(username, page=1, per_page=10):
    """Fetch repositories for a given GitHub username with pagination."""
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

def display_repositories(repos):
    """Display repository information."""
    if not repos:
        print("No repositories found or error occurred.")
        return

    for repo in repos:
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        language = repo.get('language', 'Not specified')
        print(f"Name: {name}")
        print(f"  Description: {description}")
        print(f"  Stars: {stars}, Forks: {forks}, Language: {language}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)

    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    repos = fetch_repositories(username, page, per_page)
    if repos is not None:
        display_repositories(repos)

if __name__ == "__main__":
    main()