import requests

def fetch_github_repos(username, page=1, per_page=10):
    url = f"https://api.github.com/users/{username}/repos"
    params = {'page': page, 'per_page': per_page}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description'] or 'No description'}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
        return repos
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    user = input("Enter GitHub username: ")
    fetch_github_repos(user)import requests
import sys

def fetch_repositories(username, per_page=30):
    """Fetch repositories for a given GitHub username with pagination."""
    repos = []
    page = 1
    headers = {'Accept': 'application/vnd.github.v3+json'}

    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {'page': page, 'per_page': per_page}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            repos.extend(data)
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}", file=sys.stderr)
            break

    return repos

def display_repositories(repos):
    """Display repository information."""
    if not repos:
        print("No repositories found.")
        return

    print(f"Found {len(repos)} repositories:")
    for repo in repos:
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        print(f"- {name}")
        print(f"  Description: {description}")
        print(f"  Stars: {stars}, Forks: {forks}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)

    username = sys.argv[1]
    repositories = fetch_repositories(username)
    display_repositories(repositories)import requests
import sys

def fetch_github_repos(username, token=None, per_page=30):
    """
    Fetch public repositories for a given GitHub username with pagination.
    Returns a list of repository names or an empty list on error.
    """
    repos = []
    page = 1
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    while True:
        url = f'https://api.github.com/users/{username}/repos'
        params = {'page': page, 'per_page': per_page}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            for repo in data:
                repos.append(repo['name'])
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}", file=sys.stderr)
            return []
        except ValueError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            return []
    return repos

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [access_token]")
        sys.exit(1)

    username = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    repositories = fetch_github_repos(username, token)

    if repositories:
        print(f"Public repositories for user '{username}':")
        for idx, repo in enumerate(repositories, 1):
            print(f"{idx}. {repo}")
        print(f"\nTotal repositories found: {len(repositories)}")
    else:
        print(f"No repositories found or an error occurred for user '{username}'.")