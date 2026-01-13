import requests

def get_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    if response.status_code == 200:
        contributors = response.json()
        for contributor in contributors:
            print(f"{contributor['login']}: {contributor['contributions']} commits")
    else:
        print(f"Failed to fetch contributors. Status code: {response.status_code}")

if __name__ == "__main__":
    get_contributors("torvalds", "linux")