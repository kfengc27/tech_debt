"""
Scrape list of projects from the GitHub topics page and store them into AWS RDS MySQL
- 使用 ?page= 分页
- 每页解析完立刻写入 MySQL
- 用 github_scrape_progress 表记录 last_page，支持断点续爬
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import re
import json

import pymysql
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless=new")      # New headless mode for Chrome 109+
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
# -------------------- CONFIG --------------------
lang = 'java'  # options: 'php', 'python', 'java'
base_url = f"https://github.com/topics/{lang}?l={lang}"
max_pages = 1000  # 安全上限，防止死循环

# -------------------- DB CONNECTION --------------------
conn = pymysql.connect(
    host="suyi-db.cbiqcs2e4lyn.ap-southeast-2.rds.amazonaws.com",
    user="admin",
    password="Bird123!?!",
    database="tech_debt",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.Cursor
)

# -------------------- 1. 初始化表 & 读取进度 --------------------
with conn.cursor() as cursor:
    # 1.1 进度表：每种语言一行，记录 last_page
    create_progress_table_sql = """
    CREATE TABLE IF NOT EXISTS github_scrape_progress (
        language   VARCHAR(255) PRIMARY KEY,
        last_page  INT NOT NULL DEFAULT 0,
        updated_at DATETIME NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cursor.execute(create_progress_table_sql)

    # 1.2 主表：存 repo 信息
    create_projects_table_sql = """
    CREATE TABLE IF NOT EXISTS github_projects (
        id INT AUTO_INCREMENT PRIMARY KEY,
        language VARCHAR(255),
        framework VARCHAR(255),
        repo VARCHAR(255),
        stars INT,
        topics JSON,
        repo_name VARCHAR(255),
        repo_url VARCHAR(500),
        scraped_at DATETIME NULL,
        UNIQUE KEY uniq_repo (language, framework, repo)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cursor.execute(create_projects_table_sql)

    # 1.3 给旧表补 scraped_at（如果已存在会抛 1060，我们忽略）
    try:
        cursor.execute("ALTER TABLE github_projects ADD COLUMN scraped_at DATETIME NULL;")
    except pymysql.err.OperationalError as e:
        if e.args[0] != 1060:  # 1060 = Duplicate column name
            raise

    # 1.4 读取 / 初始化本语言的 last_page
    cursor.execute(
        "SELECT last_page FROM github_scrape_progress WHERE language=%s",
        (lang,)
    )
    row = cursor.fetchone()
    if row:
        last_page = row[0]
    else:
        last_page = 0
        cursor.execute(
            "INSERT INTO github_scrape_progress (language, last_page, updated_at) "
            "VALUES (%s, %s, NOW())",
            (lang, last_page)
        )

conn.commit()

start_page = last_page + 1
print(f"[INFO] Language={lang}, resuming from page {start_page}")

# -------------------- 2. Selenium 初始化 --------------------
driver = webdriver.Chrome(options=chrome_options)

# -------------------- 3. 按页爬取 & 每页入库 --------------------
try:
    for page in range(start_page, max_pages + 1):
        page_url = f"{base_url}&page={page}"
        print(f"[INFO] Scraping {page_url}")
        driver.get(page_url)
        time.sleep(5)  # 根据网络情况调整

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 和你原来的选择器保持一致：每个 repo 卡片的 title 区域
        project_elements = soup.find_all(
            class_="f3 color-fg-muted text-normal lh-condensed"
        )

        # 没项目了，直接停止
        if not project_elements:
            print(f"[INFO] No projects found on page {page}, stopping.")
            break

        # ⭐ 当前页的临时记录
        page_records = []

        for element in project_elements:
            # owner / repo 名称
            project_name = element.text.strip().split()
            if len(project_name) < 2:
                continue
            owner = project_name[0]
            repo = project_name[-1]

            # 所在卡片 article
            card = element.find_parent("article")
            if card:
                stars_ele = card.find(class_="Counter js-social-count")
            else:
                stars_ele = element.find_next(class_="Counter js-social-count")

            # stars
            if stars_ele and stars_ele.get("aria-label"):
                stars = int(re.sub(r'[^0-9]', '', stars_ele.get("aria-label")))
            else:
                stars_text = stars_ele.text if stars_ele else "0"
                stars = int(re.sub(r'[^0-9]', '', stars_text) or 0)

            # topics
            topics = []
            if card:
                for a in card.select("a.topic-tag"):
                    topics.append(a.get_text(strip=True))

            page_records.append((lang, owner, repo, stars, topics))

        if not page_records:
            print(f"[INFO] No valid projects on page {page}, skip DB save.")
            # 也可以选择 continue，不更新 last_page
            continue

        # -------------------- 3.1 当前页 → DataFrame --------------------
        df_page = pd.DataFrame(
            page_records,
            columns=['Language', 'Framework', 'Repo', 'Stars', 'topics']
        )

        # repo_name & repo_url
        repo_info = []
        for idx, row in df_page[['Framework', 'Repo']].iterrows():
            repo_name = f"{row['Framework']}-{row['Repo']}"
            repo_url = f"https://github.com/{row['Framework']}/{row['Repo']}.git"
            repo_info.append((repo_name, repo_url))

        repo_info = pd.DataFrame(repo_info, columns=['repo_name', 'repo_url'])

        df_page = pd.concat([df_page, repo_info], axis=1)

        # 如果你后续会用到这些列，可以在本地 DataFrame 保留（目前没入库）
        df_page[['folder_analy', 'no_fitted_segs',
                 'date_first_commit', 'date_last_commit']] = np.nan

        # topics 转 JSON 字符串
        df_page['topics'] = df_page['topics'].apply(json.dumps)

        # NaN -> None，方便存 MySQL NULL
        df_for_db = df_page.replace({np.nan: None})

        # -------------------- 3.2 当前页写入 MySQL --------------------
        insert_sql = """
        INSERT INTO github_projects (
            language,
            framework,
            repo,
            stars,
            topics,
            repo_name,
            repo_url,
            scraped_at
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, NOW()
        )
        ON DUPLICATE KEY UPDATE
            stars = VALUES(stars),
            topics = VALUES(topics),
            repo_name = VALUES(repo_name),
            repo_url = VALUES(repo_url);
        """

        data = df_for_db[
            [
                "Language",
                "Framework",
                "Repo",
                "Stars",
                "topics",
                "repo_name",
                "repo_url"
            ]
        ].values.tolist()

        with conn.cursor() as cursor:
            cursor.executemany(insert_sql, data)
            # ✅ 更新进度：本页数据入库成功之后再更新 last_page
            cursor.execute(
                "UPDATE github_scrape_progress "
                "SET last_page=%s, updated_at=NOW() "
                "WHERE language=%s",
                (page, lang)
            )
        conn.commit()

        print(f"[INFO] Page {page}: inserted/updated {len(data)} rows, progress saved.")

finally:
    driver.quit()
    print("[INFO] Selenium driver closed.")
    conn.close()
    print("[INFO] MySQL connection closed.")
