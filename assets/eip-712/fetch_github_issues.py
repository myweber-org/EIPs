import requests
import sys

def fetch_issues(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    params = {"state": "open"}
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        issues = response.json()
        return issues
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        return []

def display_issues(issues):
    if not issues:
        print("No open issues found.")
        return

    print(f"Found {len(issues)} open issue(s):\n")
    for issue in issues:
        print(f"#{issue['number']}: {issue['title']}")
        print(f"    URL: {issue['html_url']}")
        print(f"    Created by: {issue['user']['login']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_issues.py <repo_owner> <repo_name>")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]

    issues = fetch_issues(owner, repo)
    display_issues(issues)