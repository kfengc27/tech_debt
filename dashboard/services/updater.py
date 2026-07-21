from pathlib import Path
import subprocess
from typing import Optional

from django.conf import settings

from dashboard.models import Repository, UpdateHistory


def get_local_repo_path(repository: Repository) -> Path:
    """
    Return the local path of a repository.

    Example:
        settings.REPOS_DIR / "django"
    """
    return Path(settings.REPOS_DIR) / repository.name


def run_git_command(
    repo_path: Path,
    command: list[str],
) -> subprocess.CompletedProcess:
    """
    Run a Git command inside a local repository.

    Raises:
        subprocess.CalledProcessError:
            If the Git command fails.
    """
    return subprocess.run(
        command,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )


def get_commit_hash(repo_path: Path) -> str:
    """
    Return the current HEAD commit hash.
    """
    result = run_git_command(
        repo_path,
        ["git", "rev-parse", "HEAD"],
    )

    return result.stdout.strip()


def is_git_repository(repo_path: Path) -> bool:
    """
    Check whether the given folder is a valid Git repository.
    """
    return (repo_path / ".git").exists()


def create_failed_history(
    repository: Repository,
    error_message: str,
    previous_commit: str = "",
) -> UpdateHistory:
    """
    Create a failed UpdateHistory record.
    """
    return UpdateHistory.objects.create(
        repository=repository,
        previous_commit=previous_commit,
        new_commit="",
        status="failed",
        error_message=error_message,
    )


def check_and_update_repository(repository: Repository) -> str:
    """
    Check and update one repository.

    Returns:
        "updated":
            New commits were pulled.

        "no_change":
            Repository is already up to date.

        "failed":
            The update process failed.
    """
    repo_path = get_local_repo_path(repository)

    if not repo_path.exists():
        create_failed_history(
            repository=repository,
            error_message=f"Repository folder does not exist: {repo_path}",
        )
        return "failed"

    if not repo_path.is_dir():
        create_failed_history(
            repository=repository,
            error_message=f"Repository path is not a directory: {repo_path}",
        )
        return "failed"

    if not is_git_repository(repo_path):
        create_failed_history(
            repository=repository,
            error_message=f"Not a valid Git repository: {repo_path}",
        )
        return "failed"

    previous_commit = ""

    try:
        previous_commit = get_commit_hash(repo_path)

        run_git_command(
            repo_path,
            ["git", "fetch", "origin"],
        )

        run_git_command(
            repo_path,
            ["git", "pull", "--ff-only"],
        )

        new_commit = get_commit_hash(repo_path)

        if previous_commit == new_commit:
            status = "no_change"
        else:
            status = "updated"

        UpdateHistory.objects.create(
            repository=repository,
            previous_commit=previous_commit,
            new_commit=new_commit,
            status=status,
            error_message="",
        )

        return status

    except subprocess.CalledProcessError as error:
        error_message = (
            error.stderr.strip()
            if error.stderr
            else error.stdout.strip()
            if error.stdout
            else str(error)
        )

        create_failed_history(
            repository=repository,
            previous_commit=previous_commit,
            error_message=error_message,
        )

        return "failed"

    except Exception as error:
        create_failed_history(
            repository=repository,
            previous_commit=previous_commit,
            error_message=str(error),
        )

        return "failed"


def update_all_repositories() -> dict[str, int]:
    """
    Check and update all repositories in the database.

    Returns:
        {
            "total": 100,
            "updated": 10,
            "no_change": 85,
            "failed": 5,
        }
    """
    result = {
        "total": 0,
        "updated": 0,
        "no_change": 0,
        "failed": 0,
    }

    repositories = Repository.objects.all()

    for repository in repositories:
        result["total"] += 1

        status = check_and_update_repository(repository)

        result[status] += 1

    return result