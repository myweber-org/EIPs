import requests
import sys

def fetch_github_repos(username, token=None, per_page=30):
    """
    Fetch all public repositories for a given GitHub username.
    Supports pagination and optional authentication for higher rate limits.
    """
    repos = []
    page = 1
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    while True:
        url = f'https://api.github.com/users/{username}/repos'
        params = {'page': page, 'per_page': per_page}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.json().get('message', 'Unknown error')}")
            sys.exit(1)

        data = response.json()
        if not data:
            break

        repos.extend(data)
        page += 1

    return repos

def display_repos(repos):
    """Display repository information in a simple format."""
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"  Description: {repo['description'] or 'No description'}")
        print(f"  URL: {repo['html_url']}")
        print(f"  Stars: {repo['stargazers_count']}")
        print(f"  Language: {repo['language'] or 'Not specified'}")
        print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [token]")
        sys.exit(1)

    username = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None
    repositories = fetch_github_repos(username, token)

    print(f"Found {len(repositories)} repositories for user '{username}':\n")
    display_repos(repositories)