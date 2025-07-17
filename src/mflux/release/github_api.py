import requests


class GitHubAPI:
    @staticmethod
    def check_github_release_exists(
        github_token: str,
        github_repo: str,
        tag_name: str,
    ) -> bool:
        print("🔍 Checking GitHub release existence...")

        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }

        url = f"https://api.github.com/repos/{github_repo}/releases/tags/{tag_name}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            print(f"✅ GitHub release {tag_name} exists")
            return True

        if response.status_code == 404:
            print(f"   GitHub release {tag_name} does not exist")
            return False

        # Any other status code is unexpected and likely indicates auth/rate-limit issues.
        error_msg = f"GitHub API returned {response.status_code} while checking for release {tag_name}: {response.text}"
        raise requests.HTTPError(error_msg, response=response)

    @staticmethod
    def create_github_release(
        github_token: str,
        github_repo: str,
        tag_name: str,
        version: str,
        release_notes: str,
    ) -> dict:
        print("🐙 Creating GitHub release...")

        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MFLUX-Release-Script",
        }

        url = f"https://api.github.com/repos/{github_repo}/releases"

        data = {
            "tag_name": tag_name,
            "name": f"Release {version}",
            "body": release_notes,
            "draft": False,
            "prerelease": False,
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 201:
            print(f"✅ Successfully created GitHub release for {tag_name}")
            return response.json()
        else:
            raise Exception(f"Failed to create GitHub release: {response.status_code} - {response.text}")
