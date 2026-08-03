from datetime import timedelta
import re
from django.conf import settings
from django.contrib import messages
from django.db import IntegrityError
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import RepositoryForm
from .models import Repository, UpdateHistory
import subprocess
import threading

def calculate_percentage(value, total):
    if total == 0:
        return 0

    return round((value / total) * 100, 1)


def get_local_repository_names():
    repos_dir = settings.REPOS_DIR

    if not repos_dir.exists():
        return set()

    return {
        folder.name.lower()
        for folder in repos_dir.iterdir()
        if folder.is_dir()
    }

def dashboard_home(request):
    repository_form = RepositoryForm()

    if request.method == "POST":
        github_urls_text = request.POST.get(
            "github_urls",
            ""
        )

        # 支持：
        # 1. 每行一个 URL
        # 2. 空格分隔
        # 3. 逗号分隔
        github_urls = [
            url.strip()
            for url in re.split(
                r"[\s,]+",
                github_urls_text,
            )
            if url.strip()
        ]

        added_repositories = []
        duplicate_urls = []
        invalid_urls = []

        for github_url in github_urls:
            # 每一个 URL 单独使用现有的 RepositoryForm 验证
            form = RepositoryForm(
                {
                    "github_url": github_url
                }
            )

            if not form.is_valid():
                invalid_urls.append(github_url)
                continue

            try:
                repository = form.save()

                added_repositories.append(
                    repository.name
                )

            except IntegrityError:
                duplicate_urls.append(
                    github_url
                )

        # 成功添加
        if added_repositories:
            messages.success(
                request,
                (
                    f"{len(added_repositories)} "
                    "repositories were successfully "
                    "added to the project list."
                ),
            )

        # 重复
        if duplicate_urls:
            messages.warning(
                request,
                (
                    f"{len(duplicate_urls)} "
                    "repositories were skipped "
                    "because they already exist."
                ),
            )

        # 无效 URL
        if invalid_urls:
            messages.error(
                request,
                (
                    f"{len(invalid_urls)} "
                    "invalid repository URLs "
                    "could not be added."
                ),
            )

        if added_repositories:
            return redirect(
                "dashboard_home"
            )

    local_repository_names = (
        get_local_repository_names()
    )

    repositories = Repository.objects.all()

    # Synchronise database status with local folders.
    for repository in repositories:
        downloaded = (
            repository.name.lower()
            in local_repository_names
        )

        latest_update = (
            repository.update_history
            .filter(status="updated")
            .order_by("-checked_at")
            .first()
        )

        repository.latest_update = latest_update

        if repository.is_downloaded != downloaded:
            Repository.objects.filter(
                pk=repository.pk
            ).update(
                is_downloaded=downloaded
            )

            repository.is_downloaded = downloaded

    total_projects = repositories.count()

    downloaded_projects = repositories.filter(
        is_downloaded=True
    ).count()

    one_week_ago = (
        timezone.now()
        - timedelta(days=7)
    )

    updated_projects = (
        UpdateHistory.objects
        .filter(
            checked_at__gte=one_week_ago,
            status="updated",
        )
        .values("repository_id")
        .distinct()
        .count()
    )

    analysed_projects = repositories.filter(
        is_analysed=True
    ).count()

    context = {
        "repository_form": repository_form,
        "repositories": repositories,

        "total_projects": total_projects,
        "downloaded_projects": downloaded_projects,
        "updated_projects": updated_projects,
        "analysed_projects": analysed_projects,

        "total_percentage": (
            100 if total_projects > 0 else 0
        ),

        "downloaded_percentage": calculate_percentage(
            downloaded_projects,
            total_projects,
        ),

        "updated_percentage": calculate_percentage(
            updated_projects,
            total_projects,
        ),

        "analysed_percentage": calculate_percentage(
            analysed_projects,
            total_projects,
        ),
    }

    return render(
        request,
        "dashboard/home.html",
        context,
    )

@require_POST
def delete_repository(request, repository_id):
    repository = get_object_or_404(
        Repository,
        pk=repository_id,
    )

    repository_name = repository.name
    repository.delete()

    messages.success(
        request,
        f"{repository_name} was removed from the project list.",
    )

    return redirect("dashboard_home")



def project_detail(request, pk):
    repository = get_object_or_404(
        Repository,
        pk=pk
    )

    return render(
        request,
        "dashboard/project_detail.html",
        {
            "repository": repository,
        }
    )


def download_repositories_background(repository_ids):
    repo_dir = settings.REPOS_DIR

    repo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Background download started: {repo_dir}"
    )

    for repository_id in repository_ids:
        try:
            repository = Repository.objects.get(
                pk=repository_id
            )

            local_path = (
                repo_dir
                / repository.name
            )

            if local_path.exists():
                print(
                    f"Already exists: {repository.name}"
                )

                repository.is_downloaded = True
                repository.save(
                    update_fields=["is_downloaded"]
                )

                continue

            print(
                f"Downloading: {repository.name}"
            )

            subprocess.run(
                [
                    "git",
                    "clone",
                    repository.github_url,
                    str(local_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            repository.is_downloaded = True
            repository.save(
                update_fields=["is_downloaded"]
            )

            print(
                f"Completed: {repository.name}"
            )

        except Exception as error:
            print(
                f"Failed repository {repository_id}: "
                f"{error}"
            )

    print("Background download completed.")

@require_POST
def download_all_repositories(request):

    repositories = Repository.objects.filter(
        is_downloaded=False
    )

    repository_ids = list(
        repositories.values_list(
            "id",
            flat=True,
        )
    )

    thread = threading.Thread(
        target=download_repositories_background,
        args=(repository_ids,),
        daemon=True,
    )

    thread.start()

    messages.success(
        request,
        (
            f"Background download started for "
            f"{len(repository_ids)} repositories."
        ),
    )

    return redirect(
        "dashboard_home"
    )

def update_repositories_background(repository_ids):
    repo_dir = settings.REPOS_DIR

    for repository_id in repository_ids:
        repository = None
        previous_commit = ""

        try:
            repository = Repository.objects.get(
                pk=repository_id
            )

            local_path = repo_dir / repository.name

            if not local_path.exists():
                UpdateHistory.objects.create(
                    repository=repository,
                    status="failed",
                    error_message="Local repository folder not found.",
                )
                continue

            previous_commit = subprocess.run(
                [
                    "git",
                    "-C",
                    str(local_path),
                    "rev-parse",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            subprocess.run(
                [
                    "git",
                    "-C",
                    str(local_path),
                    "fetch",
                    "origin",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            remote_commit = subprocess.run(
                [
                    "git",
                    "-C",
                    str(local_path),
                    "rev-parse",
                    "@{u}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            if previous_commit == remote_commit:
                UpdateHistory.objects.create(
                    repository=repository,
                    previous_commit=previous_commit,
                    new_commit=remote_commit,
                    status="no_change",
                )
                continue

            subprocess.run(
                [
                    "git",
                    "-C",
                    str(local_path),
                    "pull",
                    "--ff-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            new_commit = subprocess.run(
                [
                    "git",
                    "-C",
                    str(local_path),
                    "rev-parse",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            UpdateHistory.objects.create(
                repository=repository,
                previous_commit=previous_commit,
                new_commit=new_commit,
                status="updated",
            )

        except Exception as error:
            if repository:
                UpdateHistory.objects.create(
                    repository=repository,
                    previous_commit=previous_commit,
                    status="failed",
                    error_message=str(error),
                )

@require_POST
def update_all_repositories(request):
    repository_ids = list(
        Repository.objects
        .filter(is_downloaded=True)
        .values_list("id", flat=True)
    )

    thread = threading.Thread(
        target=update_repositories_background,
        args=(repository_ids,),
        daemon=True,
    )

    thread.start()

    messages.success(
        request,
        (
            f"Update started for "
            f"{len(repository_ids)} repositories. "
            "It is running in the background."
        ),
    )

    return redirect("dashboard_home")


def get_commit_history(repo_path, limit=50):
    print(f"Fetching commit history for {repo_path} with limit {limit}")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "log",
            f"-{limit}",
            "--pretty=format:%H|%an|%ae|%ad|%s",
            "--date=iso",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    commits = []

    for line in result.stdout.splitlines():

        commit_hash, author, email, date, message = line.split("|", 4)

        commits.append({
            "hash": commit_hash,
            "short_hash": commit_hash[:7],
            "author": author,
            "email": email,
            "date": date,
            "message": message,
        })
    print(f"Fetched {len(commits)} commits for {repo_path}")
    return commits

def project_detail(request, pk):
    repository = get_object_or_404(
        Repository,
        pk=pk
    )

    repo_path = (
        settings.REPOS_DIR
        / repository.name
    )

    commits = get_commit_history(
        repo_path,
        limit=50
    )

    return render(
        request,
        "dashboard/project_detail.html",
        {
            "repository": repository,
            "commits": commits,
        }
    )