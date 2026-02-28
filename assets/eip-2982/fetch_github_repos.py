import requests
import sys

def fetch_repositories(username):
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
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    fetch_repositories(username)import requests
import argparse
import sys

def get_user_repositories(username, sort_by='created', direction='desc'):
    url = f"https://api.github.com/users/{username}/repos"
    params = {'sort': sort_by, 'direction': direction}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No repositories found for user: {username}")
            return []
            
        return repos
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

def display_repositories(repos, max_count=10):
    if not repos:
        return
    
    print(f"\nFound {len(repos)} repositories. Displaying first {min(max_count, len(repos))}:\n")
    print(f"{'Name':<30} {'Stars':<8} {'Language':<15} {'Created':<12}")
    print("-" * 70)
    
    for repo in repos[:max_count]:
        name = repo.get('name', 'N/A')[:28]
        stars = repo.get('stargazers_count', 0)
        language = repo.get('language', 'Not specified')[:13]
        created = repo.get('created_at', 'N/A')[:10]
        
        print(f"{name:<30} {stars:<8} {language:<15} {created:<12}")

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'], 
                       default='created', help='Sort repositories by field')
    parser.add_argument('--direction', choices=['asc', 'desc'], 
                       default='desc', help='Sort direction')
    parser.add_argument('--limit', type=int, default=10, 
                       help='Maximum number of repositories to display')
    
    args = parser.parse_args()
    
    repos = get_user_repositories(args.username, args.sort, args.direction)
    
    if repos:
        display_repositories(repos, args.limit)
        
        # Additional statistics
        total_stars = sum(repo.get('stargazers_count', 0) for repo in repos)
        languages = {}
        for repo in repos:
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\nStatistics for {args.username}:")
        print(f"  Total repositories: {len(repos)}")
        print(f"  Total stars: {total_stars}")
        if languages:
            print(f"  Languages used: {', '.join(languages.keys())}")
    
    return 0 if repos else 1

if __name__ == "__main__":
    sys.exit(main())