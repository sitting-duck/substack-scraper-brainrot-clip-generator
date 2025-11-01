import os, re, time, json
from urllib.parse import urljoin
import requests, feedparser
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
BASE = os.getenv("BASE_URL", "https://carrieradomski.substack.com").rstrip("/")
OUT_DIR = os.getenv("OUT_DIR", "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "posts.jsonl")

UA = "substack-rag/0.1 (contact: you@example.com)"

def get_entries():
    # prefer RSS
    feed_url = f"{BASE}/feed"
    d = feedparser.parse(feed_url)
    if d.entries:
        return [{"link": e.link, "title": e.get("title","")} for e in d.entries]

    # fallback: /archive scraping (basic)
    arch = f"{BASE}/archive"
    r = requests.get(arch, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"/p/|/post/", href):
            url = urljoin(BASE, href)
            title = a.get_text(strip=True) or url.split("/")[-1]
            links.append({"link": url, "title": title})
    # de-dupe, preserve order
    seen, uniq = set(), []
    for e in links:
        if e["link"] not in seen:
            seen.add(e["link"]); uniq.append(e)
    return uniq

def extract_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find("div", {"data-test-id":"post-body"}) or soup
    for tag in article.select("script, style, nav, aside, header, footer"):
        tag.decompose()
    # Normalize headings/links into readable Markdown-ish text
    text = md(str(article), heading_style="ATX")
    # light cleanup
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def fetch(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.text

def main():
    entries = get_entries()
    print(f"Discovered {len(entries)} posts from {BASE}")
    out = open(OUT_JSONL, "w", encoding="utf-8")
    try:
        for e in tqdm(entries):
            url = e["link"]
            try:
                html = fetch(url)
            except Exception as ex:
                print("Fetch failed:", url, ex); time.sleep(2); continue

            soup = BeautifulSoup(html, "html.parser")
            title = e.get("title") or (soup.find(["h1","h2"]).get_text(strip=True) if soup.find(["h1","h2"]) else url.split("/")[-1])
            text = extract_clean_text(html)
            doc_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", title)[:180]

            rec = {
                "id": doc_id,
                "url": url,
                "title": title,
                "text": text,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(1.5)  # be polite
    finally:
        out.close()
    print("Wrote:", OUT_JSONL)

if __name__ == "__main__":
    main()

