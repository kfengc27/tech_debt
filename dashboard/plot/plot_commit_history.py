import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_FILE = "commit_history.csv"
OUTPUT_DIR = Path("commit_history_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df.groupby(["project", "month"])
      .size()
      .reset_index(name="commit_count")
)

for project, data in monthly.groupby("project"):
    data = data.sort_values("month")

    plt.figure(figsize=(12, 5))
    plt.plot(data["month"], data["commit_count"], marker="o")

    plt.title(f"Commit History - {project}")
    plt.xlabel("Month")
    plt.ylabel("Number of Commits")
    plt.xticks(rotation=45)

    plt.tight_layout()

    safe_name = (
        project.replace("/", "_")
               .replace("\\", "_")
               .replace(":", "_")
               .replace(" ", "_")
    )

    plt.savefig(OUTPUT_DIR / f"{safe_name}_commit_history.png", dpi=300)
    plt.close()

print(f"Done. Saved plots to: {OUTPUT_DIR.resolve()}")
