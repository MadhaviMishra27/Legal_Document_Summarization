# Step 1
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://indiankanoon.org"
SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Final 15 Important Indian Legal Keywords
LEGAL_KEYWORDS = [
    "appeal", "petition", "bail",
    "affidavit",
    "jurisdiction",
    "writ",
    "litigation",
    "judgment",
    "abetment",
    "acquittal",
    "agreement",
    "consideration",
    "indemnity",
    "force majeure",
    "sale deed",
    "power of attorney",
    "will",
    "liability"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MAX_DOCS = 500    # ✅ total docs limit
MAX_PAGES = 30    # ✅ keep only docs ≤ 30 pages

def fetch_links(keyword, pages=5):
    """Fetch judgment links for a keyword from Indian Kanoon, with page count filter."""
    all_links = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/search/?formInput={keyword}&pagenum={page}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Each search result
        for result in soup.find_all("div", class_="result_title"):
            link = result.find("a", href=True)
            if not link or "/doc/" not in link["href"]:
                continue

            # Check for page info in metadata
            meta = result.find_next_sibling("div", class_="result_subtitle")
            pages = None
            if meta:
                txt = meta.get_text()
                if "pages" in txt.lower():
                    try:
                        pages = int([w for w in txt.split() if w.isdigit()][-1])
                    except Exception:
                        pass

            # ✅ Only keep links with ≤ MAX_PAGES
            if pages is None or pages <= MAX_PAGES:
                all_links.append(BASE_URL + link["href"])

        time.sleep(random.uniform(1, 3))  # polite pause
    return all_links

def download_judgments(keyword, limit=50, downloaded_so_far=0):
    """Download judgment HTML pages with ≤ MAX_PAGES."""
    links = fetch_links(keyword, pages=10)  # search deeper for each keyword
    print(f"🔎 {keyword}: Found {len(links)} valid links (≤ {MAX_PAGES} pages)")

    count = 0
    for idx, link in enumerate(tqdm(links[:limit], desc=f"📑 {keyword}")):
        if downloaded_so_far + count >= MAX_DOCS:
            print(f"⚠️ Reached max limit of {MAX_DOCS} docs. Stopping.")
            return count

        filename = os.path.join(SAVE_DIR, f"{keyword}_{idx+1}.html")
        if os.path.exists(filename):
            continue  # skip if already downloaded

        for attempt in range(3):  # retry up to 3 times
            try:
                resp = requests.get(link, headers=HEADERS, timeout=30)
                if resp.status_code == 429:  # rate limit
                    wait_time = (attempt + 1) * 30
                    print(f"⚠️ Rate limited! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()

                # ✅ Save raw HTML
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                count += 1
                break  # success

            except Exception as e:
                print(f"❌ Failed attempt {attempt+1} for {link}: {e}")
                time.sleep(5)

        time.sleep(random.uniform(1, 3))  # polite delay
    return count

if __name__ == "__main__":
    total_downloaded = 0
    for kw in LEGAL_KEYWORDS:
        docs = download_judgments(kw, limit=50, downloaded_so_far=total_downloaded)
        total_downloaded += docs
        print(f"✅ Total downloaded so far: {total_downloaded}")
        if total_downloaded >= MAX_DOCS:
            print(f"🎉 Finished! Collected {total_downloaded} docs.")
            break
