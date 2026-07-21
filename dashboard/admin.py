from django.contrib import admin

from .models import Repository


@admin.register(Repository)
class RepositoryAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "github_url",
        "is_downloaded",
        "is_analysed",
        "last_updated",
        "created_at",
    )

    list_filter = (
        "is_downloaded",
        "is_analysed",
    )

    search_fields = (
        "name",
        "github_url",
    )

    ordering = (
        "name",
    )