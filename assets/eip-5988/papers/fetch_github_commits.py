import requests
import sys
from datetime import datetime, timedelta

def fetch_recent_commits(owner, repo, days=7, max_commits=10):
    """
    Fetch recent commits from a GitHub repository within the specified days.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    params = {'since': since_date, 'per_page': max_commits}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        commits = response.json()
        
        if not commits:
            print(f"No commits found in the last {days} days.")
            return []
            
        print(f"Recent commits for {owner}/{repo} (last {days} days):")
        for commit in commits:
            sha = commit['sha'][:7]
            author = commit['commit']['author']['name']
            date_str = commit['commit']['author']['date']
            message = commit['commit']['message'].split('\n')[0]
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {sha} | {author} | {formatted_date} | {message}")
        
        return commits
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching commits: {e}", file=sys.stderr)
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_github_commits.py <owner> <repo> [days] [max_commits]")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    max_commits = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    fetch_recent_commits(owner, repo, days, max_commits)