
import requests
import sys

def get_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code == 200:
        contributors = response.json()
        return contributors
    else:
        print(f"Error: Unable to fetch contributors. Status code: {response.status_code}")
        return None

def display_contributors(contributors):
    if contributors:
        print("Contributors:")
        for contributor in contributors:
            print(f"- {contributor['login']}: {contributor['contributions']} contributions")
    else:
        print("No contributors found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    
    contributors = get_contributors(repo_owner, repo_name)
    display_contributors(contributors)