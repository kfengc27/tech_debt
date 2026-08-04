from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import io
from datetime import datetime

ROOT = Path("repos")

OUTPUT_CSV = Path("complexity_results")
OUTPUT_PLOTS = Path("complexity_plots")
OUTPUT_LOGS = Path("complexity_logs")

OUTPUT_CSV.mkdir(exist_ok=True)
OUTPUT_PLOTS.mkdir(exist_ok=True)
OUTPUT_LOGS.mkdir(exist_ok=True)

SKIP_EXISTING = True


def run_cmd(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def log_message(message):
    print(message)
    with open(OUTPUT_LOGS / "run.log", "a", encoding="utf-8") as f:
        f.write(message + "\n")


def get_current_branch(repo_path):
    result = run_cmd(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_path
    )
    return result.stdout.strip()


def restore_branch(repo_path, original_branch):
    if original_branch and original_branch != "HEAD":
        run_cmd(["git", "checkout", "-q", original_branch], cwd=repo_path)
    else:
        run_cmd(["git", "checkout", "-q", "main"], cwd=repo_path)


def get_monthly_last_commits(repo_path):
    result = run_cmd(
        ["git", "log", "--reverse", "--format=%H|%ad", "--date=short"],
        cwd=repo_path
    )

    monthly_last_commit = {}

    for line in result.stdout.splitlines():
        parts = line.split("|")

        if len(parts) != 2:
            continue

        commit_hash, date = parts
        month = date[:7]

        monthly_last_commit[month] = {
            "commit": commit_hash,
            "date": date,
            "month": month
        }

    return list(monthly_last_commit.values())


def calculate_complexity(repo_path):
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
            "-x", "./target/*",
        ],
        cwd=repo_path
    )

    if result.returncode != 0:
        log_message("Lizard stderr:")
        log_message(result.stderr[:1000])
        return None

    if not result.stdout.strip():
        return {
            "total_complexity": 0,
            "avg_complexity": 0,
            "function_count": 0,
            "nloc": 0,
            "file_count": 0
        }

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
                "end"
            ],
            on_bad_lines="skip"
        )
    except Exception as e:
        log_message(f"Lizard CSV parse failed: {e}")
        log_message(result.stdout[:1000])
        return None

    if lizard_df.empty:
        return {
            "total_complexity": 0,
            "avg_complexity": 0,
            "function_count": 0,
            "nloc": 0,
            "file_count": 0
        }

    lizard_df["ccn"] = pd.to_numeric(lizard_df["ccn"], errors="coerce")
    lizard_df["nloc"] = pd.to_numeric(lizard_df["nloc"], errors="coerce")

    lizard_df = lizard_df.dropna(subset=["ccn", "nloc"])

    if lizard_df.empty:
        return {
            "total_complexity": 0,
            "avg_complexity": 0,
            "function_count": 0,
            "nloc": 0,
            "file_count": 0
        }

    total_complexity = lizard_df["ccn"].sum()
    avg_complexity = lizard_df["ccn"].mean()
    function_count = len(lizard_df)
    total_nloc = lizard_df["nloc"].sum()
    file_count = lizard_df["file"].nunique()

    return {
        "total_complexity": total_complexity,
        "avg_complexity": avg_complexity,
        "function_count": function_count,
        "nloc": total_nloc,
        "file_count": file_count
    }


def plot_project(project_name, df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    if df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["total_complexity"], marker="o")

    plt.title(f"Cyclomatic Complexity Over Time - {project_name}")
    plt.xlabel("Commit Date")
    plt.ylabel("Total Cyclomatic Complexity")
    plt.xticks(rotation=45)

    plt.tight_layout()

    output_file = OUTPUT_PLOTS / f"{safe_name(project_name)}_complexity.png"
    plt.savefig(output_file, dpi=300)
    plt.close()


def analyze_repo(repo_path):
    project_name = repo_path.name
    safe_project = safe_name(project_name)

    csv_file = OUTPUT_CSV / f"{safe_project}_complexity.csv"

    if SKIP_EXISTING and csv_file.exists():
        log_message(f"Skip existing project: {project_name}")
        return

    log_message(f"\nAnalyzing project: {project_name}")

    original_branch = get_current_branch(repo_path)
    rows = []

    try:
        commits = get_monthly_last_commits(repo_path)

        if not commits:
            log_message(f"No commits found: {project_name}")
            return

        log_message(f"{project_name}: {len(commits)} monthly commits selected")

        for i, item in enumerate(commits, start=1):
            commit_hash = item["commit"]
            date = item["date"]
            month = item["month"]

            log_message(
                f"[{project_name}] {i}/{len(commits)} "
                f"{commit_hash[:7]} {date}"
            )

            checkout = run_cmd(
                ["git", "checkout", "-q", commit_hash],
                cwd=repo_path
            )

            if checkout.returncode != 0:
                log_message(f"Checkout failed: {project_name} {commit_hash[:7]}")
                log_message(checkout.stderr[:500])
                continue

            metrics = calculate_complexity(repo_path)

            if metrics is None:
                log_message(f"Complexity failed: {project_name} {commit_hash[:7]}")
                continue

            rows.append({
                "project": project_name,
                "commit": commit_hash,
                "date": date,
                "month": month,
                "total_complexity": metrics["total_complexity"],
                "avg_complexity": metrics["avg_complexity"],
                "function_count": metrics["function_count"],
                "nloc": metrics["nloc"],
                "file_count": metrics["file_count"]
            })

    finally:
        restore_branch(repo_path, original_branch)

    if not rows:
        log_message(f"No valid results for {project_name}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

    plot_project(project_name, df)

    log_message(f"Saved CSV: {csv_file}")
    log_message(f"Saved plot: {OUTPUT_PLOTS / f'{safe_project}_complexity.png'}")


def main():
    start_time = datetime.now()
    log_message(f"Started at: {start_time}")

    repos = [
        p for p in ROOT.iterdir()
        if p.is_dir() and (p / ".git").exists()
    ]

    log_message(f"Found {len(repos)} git repositories")

    for repo_path in repos:
        try:
            analyze_repo(repo_path)
        except KeyboardInterrupt:
            log_message("Stopped by user.")
            break
        except Exception as e:
            log_message(f"Error in {repo_path.name}: {e}")

    end_time = datetime.now()
    log_message(f"Finished at: {end_time}")
    log_message(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
