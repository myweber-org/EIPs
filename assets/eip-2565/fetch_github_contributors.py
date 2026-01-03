import requests
import csv
import sys

def fetch_contributors(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return []
    return response.json()

def save_to_csv(contributors, filename="contributors.csv"):
    if not contributors:
        print("No contributors to save.")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['login', 'id', 'contributions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for contributor in contributors:
            writer.writerow({
                'login': contributor['login'],
                'id': contributor['id'],
                'contributions': contributor['contributions']
            })
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <owner> <repo>")
        sys.exit(1)
    owner = sys.argv[1]
    repo = sys.argv[2]
    contributors = fetch_contributors(owner, repo)
    save_to_csv(contributors)