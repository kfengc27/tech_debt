from django.urls import path

from . import views


urlpatterns = [
    path(
        "",
        views.dashboard_home,
        name="dashboard_home",
    ),

    path(
        "repositories/<int:repository_id>/delete/",
        views.delete_repository,
        name="delete_repository",
    ),
    path(
    "projects/<int:pk>/",
    views.project_detail,
    name="project_detail",
),
    path(
        "repositories/download-all/",
        views.download_all_repositories,
        name="download_all_repositories",
    ),
    path(
        "repositories/update-all/",
        views.update_all_repositories,
        name="update_all_repositories",
    ),
]