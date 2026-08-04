import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_FILE = "commit_history.csv"
OUTPUT_DIR = Path("commit_history_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

for project, data in df.groupby("project"):

    # 按时间排序
    data = data.sort_values("date").reset_index(drop=True)

    # 每个 commit 累加
    data["commit_number"] = range(1, len(data) + 1)

    plt.figure(figsize=(12,5))

    plt.plot(
        data["date"],
        data["commit_number"],
        linewidth=2
    )

    plt.title(f"Cumulative Commits - {project}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Commits")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = (
        project.replace("/", "_")
               .replace("\\", "_")
               .replace(":", "_")
               .replace(" ", "_")
    )

    plt.savefig(
        OUTPUT_DIR / f"{safe_name}_commit_history.png",
        dpi=300
    )

    plt.close()

print("Done!")
