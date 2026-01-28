import requests
import sys

def get_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch contributors. Status code: {response.status_code}")
        return []
    
    contributors = response.json()
    return contributors

def display_contributors(contributors):
    if not contributors:
        print("No contributors found.")
        return
    
    print("Contributors:")
    for contributor in contributors:
        username = contributor.get('login', 'N/A')
        contributions = contributor.get('contributions', 0)
        print(f"  - {username}: {contributions} contributions")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    
    contributors = get_contributors(repo_owner, repo_name)
    display_contributors(contributors)