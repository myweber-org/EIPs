import requests

def fetch_github_repos(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No repositories found for user: {username}")
            return []
            
        print(f"Repositories for {username} (Page {page}):")
        for repo in repos:
            print(f"- {repo['name']}: {repo['description'] or 'No description'}")
            print(f"  Stars: {repo['stargazers_count']} | Forks: {repo['forks_count']}")
            print(f"  URL: {repo['html_url']}")
            print()
            
        return repos
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if response.status_code == 404:
            print(f"User '{username}' not found on GitHub.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    return []

def main():
    username = input("Enter GitHub username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return
        
    page = 1
    while True:
        repos = fetch_github_repos(username, page=page)
        if not repos:
            break
            
        if len(repos) < 30:
            print("No more repositories.")
            break
            
        cont = input("Fetch next page? (y/n): ").strip().lower()
        if cont != 'y':
            break
        page += 1

if __name__ == "__main__":
    main()