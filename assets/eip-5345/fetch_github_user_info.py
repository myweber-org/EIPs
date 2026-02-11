
import requests

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            'login': data.get('login'),
            'name': data.get('name'),
            'public_repos': data.get('public_repos'),
            'followers': data.get('followers'),
            'following': data.get('following'),
            'created_at': data.get('created_at')
        }
    else:
        return {"error": f"User '{username}' not found or API error"}

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    info = get_github_user_info(username)
    print(info)