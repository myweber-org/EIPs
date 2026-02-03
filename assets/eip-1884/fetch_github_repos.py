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
    display_repos(repositories)import requests
import sys

def fetch_user_repos(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {'page': page, 'per_page': per_page}
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return None

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"Description: {repo['description'] or 'No description'}")
        print(f"URL: {repo['html_url']}")
        print(f"Stars: {repo['stargazers_count']}")
        print(f"Forks: {repo['forks_count']}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_user_repos(username, page, per_page)
    if repos:
        display_repos(repos)
        print(f"\nTotal repositories displayed: {len(repos)}")
    else:
        print("Failed to fetch repositories.")

if __name__ == "__main__":
    main()import requests

def fetch_github_repos(username):
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
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    fetch_github_repos(username)