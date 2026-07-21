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
]