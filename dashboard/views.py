from datetime import timedelta

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
    if request.method == "POST":
        repository_form = RepositoryForm(request.POST)

        if repository_form.is_valid():
            try:
                repository = repository_form.save()

                messages.success(
                    request,
                    (
                        f"{repository.name} was successfully "
                        "added to the project list."
                    ),
                )

                return redirect("dashboard_home")

            except IntegrityError:
                repository_form.add_error(
                    "github_url",
                    (
                        "This GitHub repository is already "
                        "in the project list."
                    ),
                )
    else:
        repository_form = RepositoryForm()

    local_repository_names = get_local_repository_names()

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

    one_week_ago = timezone.now() - timedelta(days=7)

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