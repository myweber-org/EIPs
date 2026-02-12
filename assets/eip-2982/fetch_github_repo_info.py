import requests
import json

def fetch_repo_info(username, repo_name):
    """
    Fetch basic information about a GitHub repository.
    """
    url = f"https://api.github.com/repos/{username}/{repo_name}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        repo_data = response.json()
        
        info = {
            "name": repo_data.get("name"),
            "full_name": repo_data.get("full_name"),
            "description": repo_data.get("description"),
            "html_url": repo_data.get("html_url"),
            "stargazers_count": repo_data.get("stargazers_count"),
            "forks_count": repo_data.get("forks_count"),
            "open_issues_count": repo_data.get("open_issues_count"),
            "language": repo_data.get("language"),
            "created_at": repo_data.get("created_at"),
            "updated_at": repo_data.get("updated_at")
        }
        return info
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository info: {e}")
        return None

def main():
    username = "torvalds"
    repo_name = "linux"
    
    repo_info = fetch_repo_info(username, repo_name)
    
    if repo_info:
        print("Repository Information:")
        print(json.dumps(repo_info, indent=2))

if __name__ == "__main__":
    main()