"""Scrape list of projects from the GitHub topics page and store them into AWS RDS MySQL
"""

import pandas as pd
import numpy as np
from os.path import join
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import re
import json

import pymysql

# -------------------- DB CONNECTION --------------------
conn = pymysql.connect(
    host="suyi-db.cbiqcs2e4lyn.ap-southeast-2.rds.amazonaws.com",
    user="admin",
    password="Bird123!?!",
    database="tech_debt",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.Cursor
)

lang = 'java'  # options {'php', 'python', 'java'}

# Define the URL of the GitHub topics page
url = f"https://github.com/topics/{lang}?l={lang}"

# Root
folder_root = 'Data'  # root folder

# -------------------- SCRAPING --------------------
driver = webdriver.Chrome()
driver.get(url)

project_info = []

max_pages = 2000
current_page = 5

while current_page < max_pages:
    try:
        load_more_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Load more')]")
    except:
        print('%i pages loaded' % current_page)
        break

    if load_more_button:
        load_more_button.click()
        time.sleep(10)  # Wait for new content to load
        current_page += 1
    else:
        break

# Parse the HTML content of the page using BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# Find the elements containing project names (these are the h3s)
project_elements = soup.find_all(class_="f3 color-fg-muted text-normal lh-condensed")

for element in project_elements:
    # Extract the project name (owner / repo)
    project_name = element.text.strip().split()
    if len(project_name) < 2:
        continue
    owner = project_name[0]
    repo = project_name[-1]

    # Extract the stars (within the same card/article)
    card = element.find_parent("article")
    if card:
        stars_ele = card.find(class_="Counter js-social-count")
    else:
        stars_ele = element.find_next(class_="Counter js-social-count")

    if stars_ele and stars_ele.get("aria-label"):
        stars = int(re.sub(r'[^0-9]', '', stars_ele.get("aria-label")))
    else:
        # fallback: text
        stars_text = stars_ele.text if stars_ele else "0"
        stars = int(re.sub(r'[^0-9]', '', stars_text) or 0)

    # Extract topics for THIS repo only (within the card/article)
    topics = []
    if card:
        for a in card.select("a.topic-tag"):
            topics.append(a.get_text(strip=True))

    # Append the project information to the list
    # Language added as the first element
    project_info.append((lang, owner, repo, stars, topics))

# -------------------- DATAFRAME BUILDING --------------------
project_info = pd.DataFrame(
    project_info,
    columns=['Language', 'Framework', 'Repo', 'Stars', 'topics']
)

# repo_name and repo_url
repo_info = []
for idx, row in project_info[['Framework', 'Repo']].iterrows():
    repo_name = f"{row['Framework']}-{row['Repo']}"
    repo_url = f"https://github.com/{row['Framework']}/{row['Repo']}.git"
    repo_info.append((repo_name, repo_url))

repo_info = pd.DataFrame(repo_info, columns=['repo_name', 'repo_url'])

# Combine
project_info = pd.concat([project_info, repo_info], axis=1)

# Add new columns to project information
project_info[['folder_analy', 'no_fitted_segs',
              'date_first_commit', 'date_last_commit']] = np.nan

# Convert topics list -> JSON string
project_info['topics'] = project_info['topics'].apply(json.dumps)

# -------------------- SAVE INTO MYSQL TABLE --------------------
try:
    with conn.cursor() as cursor:
        # 1. Create table if not exists (now includes extra columns and unique key)
        create_table_sql = """
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
        cursor.execute(create_table_sql)


        # ⭐ 1.5 如果旧表里没有 scraped_at，就加上（第一次执行会生效，之后再执行会报“重复列”，我们忽略）
        try:
            alter_sql = "ALTER TABLE github_projects ADD COLUMN scraped_at DATETIME NULL;"
            cursor.execute(alter_sql)
        except pymysql.err.OperationalError as e:
            # 1060 = Duplicate column name 'scraped_at'，说明列已经存在，就忽略
            if e.args[0] != 1060:
                raise


        # 2. Prepare data: replace NaN with None so MySQL can store NULL
        df_for_db = project_info.replace({np.nan: None})

        # 3. Insert with ON DUPLICATE KEY UPDATE (no duplicates on rerun)
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

        cursor.executemany(insert_sql, data)
        conn.commit()
        print(f"Inserted/updated {cursor.rowcount} rows in tech_debt.github_projects")

finally:
    conn.close()
    print("MySQL connection closed.")
