from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Project paths
# ============================================================

# Expected location:
# project_root/
# ├── manage.py
# ├── tech_debt/
# │   └── settings.py
# └── dashboard/
#     └── analysis/
#         └── complexity_over_time.py

ANALYSIS_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = ANALYSIS_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Django configuration
# ============================================================

DJANGO_SETTINGS_MODULE = "tech_debt.settings"

DJANGO_AVAILABLE = False
Repository = None
ComplexitySnapshot = None


def setup_django() -> bool:
    """Initialise Django and load the required models."""
    global DJANGO_AVAILABLE
    global Repository
    global ComplexitySnapshot

    try:
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            DJANGO_SETTINGS_MODULE,
        )

        import django

        django.setup()

        from dashboard.models import (
            ComplexitySnapshot as ComplexitySnapshotModel,
        )
        from dashboard.models import Repository as RepositoryModel

        Repository = RepositoryModel
        ComplexitySnapshot = ComplexitySnapshotModel
        DJANGO_AVAILABLE = True

        print(
            "Django database integration enabled: "
            f"{DJANGO_SETTINGS_MODULE}"
        )
        return True

    except Exception as error:
        DJANGO_AVAILABLE = False

        print("Django setup failed.")
        print(f"{type(error).__name__}: {error}")
        traceback.print_exc()
        return False


# ============================================================
# Output paths
# ============================================================

OUTPUT_CSV = ANALYSIS_DIR / "complexity_results"
OUTPUT_PLOTS = ANALYSIS_DIR / "complexity_plots"
OUTPUT_LOGS = ANALYSIS_DIR / "complexity_logs"

OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUTPUT_LOGS.mkdir(parents=True, exist_ok=True)

# Keep False while populating or refreshing database snapshots.
SKIP_EXISTING = False


# ============================================================
# General utilities
# ============================================================

def run_cmd(
    command: list[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def log_message(message: str) -> None:
    print(message)

    with (OUTPUT_LOGS / "run.log").open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(message + "\n")


# ============================================================
# Git utilities
# ============================================================

def get_current_branch(repo_path: Path) -> str:
    result = run_cmd(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_path,
    )
    return result.stdout.strip()


def restore_branch(
    repo_path: Path,
    original_branch: str,
) -> None:
    if original_branch and original_branch != "HEAD":
        result = run_cmd(
            ["git", "checkout", "-q", original_branch],
            cwd=repo_path,
        )

        if result.returncode == 0:
            return

    for fallback_branch in ("main", "master"):
        result = run_cmd(
            ["git", "checkout", "-q", fallback_branch],
            cwd=repo_path,
        )

        if result.returncode == 0:
            return

    log_message(
        f"Warning: could not restore a branch for {repo_path.name}"
    )


def get_monthly_last_commits(
    repo_path: Path,
) -> list[dict[str, str]]:
    result = run_cmd(
        [
            "git",
            "log",
            "--reverse",
            "--format=%H|%ad",
            "--date=short",
        ],
        cwd=repo_path,
    )

    if result.returncode != 0:
        log_message(
            f"Git log failed for {repo_path.name}: "
            f"{result.stderr[:500]}"
        )
        return []

    monthly_last_commit: dict[str, dict[str, str]] = {}

    for line in result.stdout.splitlines():
        parts = line.split("|", 1)

        if len(parts) != 2:
            continue

        commit_hash, commit_date = parts
        month = commit_date[:7]

        monthly_last_commit[month] = {
            "commit": commit_hash,
            "date": commit_date,
            "month": month,
        }

    return list(monthly_last_commit.values())


# ============================================================
# Complexity calculation
# ============================================================

def calculate_complexity(
    repo_path: Path,
) -> dict[str, float | int] | None:
    """
    Run Lizard in CSV mode and return repository-level metrics.
    """
    result = run_cmd(
        [
            sys.executable,
            "-m",
            "lizard",
            ".",
            "--csv",
            "-x", "./.git/*",
            "-x", "./.tox/*",
            "-x", "./.venv/*",
            "-x", "./venv/*",
            "-x", "./env/*",
            "-x", "./build/*",
            "-x", "./dist/*",
            "-x", "./docs/*",
            "-x", "./test/*",
            "-x", "./tests/*",
            "-x", "./node_modules/*",
            "-x", "./vendor/*",
        ],
        cwd=repo_path,
    )

    if result.returncode != 0:
        log_message("Lizard stderr:")
        log_message(result.stderr[:1000])
        return None

    try:
        lizard_df = pd.read_csv(
            io.StringIO(result.stdout),
            header=None,
            names=[
                "nloc",
                "ccn",
                "token",
                "param",
                "length",
                "location",
                "file",
                "function",
                "long_name",
                "start",
                "end",
            ],
            on_bad_lines="skip",
        )

    except Exception as error:
        log_message(f"Lizard CSV parse failed: {error}")
        log_message(result.stdout[:1000])
        return None

    if lizard_df.empty:
        return {
            "total_complexity": 0,
            "avg_complexity": 0,
            "function_count": 0,
            "nloc": 0,
            "file_count": 0,
        }

    lizard_df["ccn"] = pd.to_numeric(
        lizard_df["ccn"],
        errors="coerce",
    )
    lizard_df["nloc"] = pd.to_numeric(
        lizard_df["nloc"],
        errors="coerce",
    )

    lizard_df = lizard_df.dropna(
        subset=["ccn", "nloc"],
    )

    if lizard_df.empty:
        return {
            "total_complexity": 0,
            "avg_complexity": 0,
            "function_count": 0,
            "nloc": 0,
            "file_count": 0,
        }

    return {
        "total_complexity": float(lizard_df["ccn"].sum()),
        "avg_complexity": float(lizard_df["ccn"].mean()),
        "function_count": int(len(lizard_df)),
        "nloc": int(lizard_df["nloc"].sum()),
        "file_count": int(lizard_df["file"].nunique()),
    }


# ============================================================
# Database persistence
# ============================================================

def get_database_repository(
    project_name: str,
) -> Any | None:
    if not DJANGO_AVAILABLE or Repository is None:
        return None

    repository = (
        Repository.objects
        .filter(name__iexact=project_name)
        .first()
    )

    if repository is None:
        log_message(
            "Repository does not exist in database: "
            f"{project_name}"
        )

    return repository


def save_complexity_snapshot(
    repository: Any,
    commit_hash: str,
    date: str,
    month: str,
    metrics: dict[str, float | int],
) -> bool:
    if (
        not DJANGO_AVAILABLE
        or ComplexitySnapshot is None
        or repository is None
    ):
        return False

    try:
        commit_date = datetime.strptime(
            date,
            "%Y-%m-%d",
        ).date()

        _, created = (
            ComplexitySnapshot.objects.update_or_create(
                repository=repository,
                commit_hash=commit_hash,
                defaults={
                    "commit_date": commit_date,
                    "month": month,
                    "total_complexity": metrics.get(
                        "total_complexity",
                        0,
                    ),
                    "average_complexity": metrics.get(
                        "avg_complexity",
                        0,
                    ),
                    "function_count": metrics.get(
                        "function_count",
                        0,
                    ),
                    "nloc": metrics.get("nloc", 0),
                    "file_count": metrics.get(
                        "file_count",
                        0,
                    ),
                },
            )
        )

        action = "Created" if created else "Updated"

        log_message(
            f"{action} DB snapshot: "
            f"{repository.name} {commit_hash[:7]}"
        )
        return True

    except Exception as error:
        log_message(
            f"Failed to save DB snapshot "
            f"{commit_hash[:7]}: {error}"
        )
        traceback.print_exc()
        return False


# ============================================================
# Plotting
# ============================================================

def plot_project(
    project_name: str,
    dataframe: pd.DataFrame,
) -> None:
    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["date"])
    dataframe = dataframe.sort_values("date")

    if dataframe.empty:
        return

    plt.figure(figsize=(12, 5))

    plt.plot(
        dataframe["date"],
        dataframe["total_complexity"],
        marker="o",
    )

    plt.title(
        f"Cyclomatic Complexity Over Time - {project_name}"
    )
    plt.xlabel("Commit Date")
    plt.ylabel("Total Cyclomatic Complexity")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = (
        OUTPUT_PLOTS
        / f"{safe_name(project_name)}_complexity.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()


# ============================================================
# Repository analysis
# ============================================================

def analyze_repo(repo_path: Path) -> None:
    project_name = repo_path.name
    safe_project = safe_name(project_name)

    csv_file = (
        OUTPUT_CSV
        / f"{safe_project}_complexity.csv"
    )

    if SKIP_EXISTING and csv_file.exists():
        log_message(
            f"Skip existing project: {project_name}"
        )
        return

    log_message(f"\nAnalyzing project: {project_name}")

    database_repository = get_database_repository(
        project_name
    )

    original_branch = get_current_branch(repo_path)
    rows: list[dict[str, Any]] = []

    try:
        commits = get_monthly_last_commits(repo_path)

        if not commits:
            log_message(
                f"No commits found: {project_name}"
            )
            return

        log_message(
            f"{project_name}: "
            f"{len(commits)} monthly commits selected"
        )

        for index, item in enumerate(
            commits,
            start=1,
        ):
            commit_hash = item["commit"]
            commit_date = item["date"]
            month = item["month"]

            log_message(
                f"[{project_name}] "
                f"{index}/{len(commits)} "
                f"{commit_hash[:7]} {commit_date}"
            )

            checkout = run_cmd(
                [
                    "git",
                    "checkout",
                    "-q",
                    commit_hash,
                ],
                cwd=repo_path,
            )

            if checkout.returncode != 0:
                log_message(
                    f"Checkout failed: "
                    f"{project_name} {commit_hash[:7]}"
                )
                log_message(checkout.stderr[:500])
                continue

            metrics = calculate_complexity(repo_path)

            if metrics is None:
                log_message(
                    f"Complexity failed: "
                    f"{project_name} {commit_hash[:7]}"
                )
                continue

            save_complexity_snapshot(
                repository=database_repository,
                commit_hash=commit_hash,
                date=commit_date,
                month=month,
                metrics=metrics,
            )

            rows.append(
                {
                    "project": project_name,
                    "commit": commit_hash,
                    "date": commit_date,
                    "month": month,
                    "total_complexity": (
                        metrics["total_complexity"]
                    ),
                    "avg_complexity": (
                        metrics["avg_complexity"]
                    ),
                    "function_count": (
                        metrics["function_count"]
                    ),
                    "nloc": metrics["nloc"],
                    "file_count": metrics["file_count"],
                }
            )

    finally:
        restore_branch(
            repo_path,
            original_branch,
        )

    if not rows:
        log_message(
            f"No valid results for {project_name}"
        )
        return

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_file, index=False)

    plot_project(
        project_name,
        dataframe,
    )

    log_message(f"Saved CSV: {csv_file}")
    log_message(
        f"Saved plot: "
        f"{OUTPUT_PLOTS / f'{safe_project}_complexity.png'}"
    )


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    if not setup_django():
        print(
            "Stopping because Django could not be initialised."
        )
        return

    from django.conf import settings

    repo_root = Path(settings.REPOS_DIR)

    start_time = datetime.now()

    log_message(f"Started at: {start_time}")
    log_message(f"Repository directory: {repo_root}")

    if not repo_root.exists():
        log_message(
            f"Repository directory does not exist: {repo_root}"
        )
        return

    repositories = [
        path
        for path in repo_root.iterdir()
        if path.is_dir()
        and (path / ".git").exists()
    ]

    log_message(
        f"Found {len(repositories)} git repositories"
    )

    for repo_path in repositories:
        try:
            analyze_repo(repo_path)

        except KeyboardInterrupt:
            log_message("Stopped by user.")
            break

        except Exception as error:
            log_message(
                f"Error in {repo_path.name}: {error}"
            )
            traceback.print_exc()

    end_time = datetime.now()

    log_message(f"Finished at: {end_time}")
    log_message(
        f"Total time: {end_time - start_time}"
    )


if __name__ == "__main__":
    main()