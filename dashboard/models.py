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

    project_language = models.CharField(
        max_length=255,
        blank=True,
    )


    project_category = models.CharField(
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

    class Meta:
        db_table = "Repositories"



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
        db_table = "UpdateHistory"
        ordering = ["-checked_at"]
        indexes = [
            models.Index(fields=["checked_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["repository", "checked_at"]),
        ]

    def __str__(self):
        return f"{self.repository.name} - {self.status} - {self.checked_at}"




class AnalysisRun(models.Model):

    repository = models.ForeignKey(
        Repository,
        on_delete=models.CASCADE,
        related_name="analysis_runs"
    )

    # Basic metrics
    lines_of_code = models.IntegerField(
        null=True,
        blank=True
    )

    file_count = models.IntegerField(
        null=True,
        blank=True
    )

    repository_size_mb = models.FloatField(
        null=True,
        blank=True
    )

    # Technical debt metrics
    cyclomatic_complexity = models.FloatField(
        null=True,
        blank=True
    )

    maintainability_index = models.FloatField(
        null=True,
        blank=True
    )


    # Analysis information
    analysed_at = models.DateTimeField(
        auto_now_add=True
    )

    commit_hash = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    class Meta:
        db_table = "Analysis_Run"
        ordering = ["-analysed_at"]


class ProjectStatus(models.Model):

    ANALYSIS_STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    PROJECT_STATUS_CHOICES = [
        ("active", "Active"),
        ("watch", "Watch"),
        ("drop_candidate", "Drop Candidate"),
        ("dropped", "Dropped"),
        ("archived", "Archived"),
    ]

    repository = models.OneToOneField(
        Repository,
        on_delete=models.CASCADE,
        related_name="project_status",
    )

    # Analysis
    analysis_status = models.CharField(
        max_length=20,
        choices=ANALYSIS_STATUS_CHOICES,
        default="pending",
    )

    last_analyzed_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    # Project lifecycle
    project_status = models.CharField(
        max_length=30,
        choices=PROJECT_STATUS_CHOICES,
        default="active",
    )

    drop_reason = models.TextField(
        blank=True,
        default="",
    )

    # Optional human decision
    notes = models.TextField(
        blank=True,
        default="",
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "project_status"

    def __str__(self):
        return f"{self.repository.name} - {self.project_status}"




class ComplexitySnapshot(models.Model):
    repository = models.ForeignKey(
        Repository,
        on_delete=models.CASCADE,
        related_name="complexity_snapshots",
    )

    commit_hash = models.CharField(max_length=40)
    commit_date = models.DateField()
    month = models.CharField(max_length=7, blank=True)

    total_complexity = models.FloatField(default=0)
    average_complexity = models.FloatField(default=0)
    function_count = models.PositiveIntegerField(default=0)
    nloc = models.PositiveIntegerField(default=0)
    file_count = models.PositiveIntegerField(default=0)

    analyzed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["commit_date"]

        constraints = [
            models.UniqueConstraint(
                fields=["repository", "commit_hash"],
                name="unique_repository_complexity_commit",
            )
        ]

    class Meta:
        db_table = "ComplexitySnapshot"