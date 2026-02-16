
import requests

def get_github_user_info(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'bio': user_data.get('bio'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following')
        }
    else:
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    info = get_github_user_info(username)
    
    if info:
        print(f"Name: {info['name']}")
        print(f"Bio: {info['bio']}")
        print(f"Public Repositories: {info['public_repos']}")
        print(f"Followers: {info['followers']}")
        print(f"Following: {info['following']}")
    else:
        print("User not found or API error.")