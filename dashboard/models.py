from urllib.parse import urlparse

from django.core.exceptions import ValidationError
from django.db import models


class Repository(models.Model):
    github_url = models.URLField(
        max_length=500,
        unique=True,
    )

    name = models.CharField(
        max_length=255,
        blank=True,
    )

    is_downloaded = models.BooleanField(
        default=False,
    )

    is_analysed = models.BooleanField(
        default=False,
    )

    last_updated = models.DateTimeField(
        null=True,
        blank=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    class Meta:
        ordering = ["name", "github_url"]

    def __str__(self):
        return self.name or self.github_url

    def clean(self):
        self.github_url = self.normalise_github_url(
            self.github_url
        )

        parsed_url = urlparse(self.github_url)

        if parsed_url.netloc.lower() not in {
            "github.com",
            "www.github.com",
        }:
            raise ValidationError(
                {
                    "github_url": (
                        "Please enter a valid GitHub repository URL."
                    )
                }
            )

        path_parts = [
            part
            for part in parsed_url.path.split("/")
            if part
        ]

        if len(path_parts) < 2:
            raise ValidationError(
                {
                    "github_url": (
                        "The URL must include a GitHub owner "
                        "and repository name."
                    )
                }
            )

        self.name = path_parts[-1]

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    @staticmethod
    def normalise_github_url(url):
        url = url.strip()

        if not url:
            return url

        url = url.rstrip("/")

        if url.endswith(".git"):
            url = url[:-4]

        url = url.replace(
            "https://www.github.com/",
            "https://github.com/",
        )

        return url.lower()
    

class UpdateHistory(models.Model):
    STATUS_CHOICES = [
        ("updated", "Updated"),
        ("no_change", "No Change"),
        ("failed", "Failed"),
    ]

    repository = models.ForeignKey(
        Repository,
        on_delete=models.CASCADE,
        related_name="update_history",
    )

    checked_at = models.DateTimeField(auto_now_add=True)

    previous_commit = models.CharField(
        max_length=64,
        blank=True,
        default="",
    )

    new_commit = models.CharField(
        max_length=64,
        blank=True,
        default="",
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
    )

    error_message = models.TextField(
        blank=True,
        default="",
    )

    class Meta:
        ordering = ["-checked_at"]
        indexes = [
            models.Index(fields=["checked_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["repository", "checked_at"]),
        ]

    def __str__(self):
        return f"{self.repository.name} - {self.status} - {self.checked_at}"