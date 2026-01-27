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
    fetch_github_repos(user)