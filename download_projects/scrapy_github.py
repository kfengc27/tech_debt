"""
Scrape list of projects from the GitHub topics page and store them into AWS RDS MySQL
- 使用 ?page= 分页
- 每页解析完立刻写入 MySQL
- 用 github_scrape_progress 表记录 last_page，支持断点续爬
- 使用 logging 记录过程
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By  # 现在没用到，但保留也没问题
from selenium.webdriver.chrome.options import Options

import time
import re
import json
import logging
from logging.handlers import RotatingFileHandler

import pymysql

# -------------------- LOGGING SETUP --------------------
logger = logging.getLogger("github_topics_scraper")
logger.setLevel(logging.INFO)

# 避免重复添加 handler（防止在 notebook 里多次运行时日志重复）
if not logger.handlers:
    # 文件日志（自动滚动）
    file_handler = RotatingFileHandler(
        "scraper.log",
        maxBytes=2 * 1024 * 1024,  # 2MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


parser = argparse.ArgumentParser(description="GitHub Topic Scraper")

parser.add_argument(
    "language",
    type=str,
    help="Programming language to scrape, e.g. python, java, php"
)

parser.add_argument(
    "--max_pages",
    type=int,
    default=1000,
    help="Maximum pages to scrape (default=1000)"
)


args = parser.parse_args()

lang = args.language
max_pages = args.max_pages

# -------------------- CONFIG --------------------
# lang = 'java'  # options: 'php', 'python', 'java'
base_url = f"https://github.com/topics/{lang}?l={lang}"
# max_pages = 1000  # 安全上限，防止死循环

# -------------------- Selenium Headless 配置 --------------------
chrome_options = Options()
chrome_options.add_argument("--headless=new")      # New headless mode for Chrome 109+
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

# -------------------- DB CONNECTION --------------------
conn = pymysql.connect(
    host="suyi-db.cbiqcs2e4lyn.ap-southeast-2.rds.amazonaws.com",
    user="admin",
    password="Bird123!?!",  # 建议用环境变量或配置文件，不要写死在代码里
    database="tech_debt",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.Cursor
)

logger.info(f"DB connected. Target language={lang}, base_url={base_url}")

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
        logger.info(f"Found existing progress for {lang}: last_page={last_page}")
    else:
        last_page = 0
        cursor.execute(
            "INSERT INTO github_scrape_progress (language, last_page, updated_at) "
            "VALUES (%s, %s, NOW())",
            (lang, last_page)
        )
        logger.info(f"No existing progress for {lang}. Initialize last_page=0")

conn.commit()

start_page = last_page + 1
logger.info(f"Resuming from page {start_page}")

# -------------------- 2. Selenium 初始化 --------------------
driver = webdriver.Chrome(options=chrome_options)
logger.info("Headless Chrome driver started.")

# -------------------- 3. 按页爬取 & 每页入库 --------------------
try:
    for page in range(start_page, max_pages + 1):
        page_url = f"{base_url}&page={page}"
        logger.info(f"[PAGE] Scraping {page_url}")
        driver.get(page_url)
        time.sleep(5)  # 根据网络情况调整

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 每个 repo 卡片的 title 区域
        project_elements = soup.find_all(
            class_="f3 color-fg-muted text-normal lh-condensed"
        )

        # 没项目了，直接停止
        if not project_elements:
            logger.warning(f"[PAGE] No projects found on page {page}, stopping.")
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
            logger.warning(f"[PAGE] No valid projects on page {page}, skip DB save.")
            continue

        logger.info(f"[PAGE] Page {page} parsed: {len(page_records)} projects found.")

        # -------------------- 3.1 当前页 → DataFrame --------------------
        df_page = pd.DataFrame(
            page_records,
            columns=['Language', 'Framework', 'Repo', 'Stars', 'topics']
        )

        # repo_name & repo_url
        repo_info = []
        for idx, r in df_page[['Framework', 'Repo']].iterrows():
            repo_name = f"{r['Framework']}-{r['Repo']}"
            repo_url = f"https://github.com/{r['Framework']}/{r['Repo']}.git"
            repo_info.append((repo_name, repo_url))

        repo_info = pd.DataFrame(repo_info, columns=['repo_name', 'repo_url'])

        df_page = pd.concat([df_page, repo_info], axis=1)

        # 保留本地列（目前不入库）
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

        try:
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
            logger.info(
                f"[DB] Page {page}: inserted/updated {len(data)} rows, "
                f"progress saved (last_page={page})."
            )
        except Exception:
            logger.error(f"[DB] Error while inserting page {page} data.", exc_info=True)
            # 根据需要决定是否 break / continue，这里我选择继续下一个 page
            # break

except Exception:
    logger.error("[FATAL] Unhandled exception in main loop.", exc_info=True)
finally:
    driver.quit()
    logger.info("Selenium driver closed.")
    conn.close()
    logger.info("MySQL connection closed.")
    logger.info("Scraper finished.")
