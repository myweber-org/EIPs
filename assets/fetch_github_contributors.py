import requests
import sys

def get_top_contributors(repo_owner, repo_name, top_n=5):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        contributors = response.json()
        
        if not contributors:
            print(f"No contributors found for {repo_owner}/{repo_name}")
            return []
        
        sorted_contributors = sorted(contributors, key=lambda x: x['contributions'], reverse=True)
        top_contributors = sorted_contributors[:top_n]
        
        print(f"Top {top_n} contributors for {repo_owner}/{repo_name}:")
        for i, contributor in enumerate(top_contributors, 1):
            print(f"{i}. {contributor['login']}: {contributor['contributions']} contributions")
        
        return top_contributors
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    get_top_contributors(owner, repo)