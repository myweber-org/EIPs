
import requests
import json

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
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    else:
        return None

def save_user_info_to_file(username, user_info):
    if user_info:
        filename = f"{username}_github_info.json"
        with open(filename, 'w') as f:
            json.dump(user_info, f, indent=4)
        print(f"User info saved to {filename}")
        return True
    else:
        print(f"Failed to fetch info for user: {username}")
        return False

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    user_info = get_github_user_info(username)
    
    if user_info:
        print(f"\nGitHub User Info for {username}:")
        for key, value in user_info.items():
            print(f"{key}: {value}")
        
        save_choice = input("\nSave to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_user_info_to_file(username, user_info)