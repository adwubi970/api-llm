import csv
import json
import os
import re
import time
from datetime import datetime
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
SOURCES = [
    ("Face ID", "https://en.wikipedia.org/wiki/Face_ID"),
    ("Clearview AI", "https://en.wikipedia.org/wiki/Clearview_AI"),
    ("Dining Services", "https://dining.iastate.edu/"),
    ("IBM Watson", "https://en.wikipedia.org/wiki/IBM_Watson"),
    ("Grocery delivery", "https://www.instacart.com/help"),
]

SYSTEM = """
Return ONLY valid JSON (no markdown, no extra text).
Schema:
{
  "uses_ai": boolean,
  "risk_level": "low" | "medium" | "high",
  "top_risks": [string, string, string]
}
Rules:
- If the description clearly does NOT involve AI, set:
  uses_ai=false, risk_level="low", top_risks=[]
- top_risks must be 0 to 3 items max, chosen ONLY from:
  privacy, bias, safety, security, transparency, health
"""
HEADERS = {
    "User-Agent": "ai-risk-demo/1.0 (contact: aliceasiedu103@gmail.com)"
}

def extract_text_from_html(html: str, max_chars: int = 1200) -> str:
    paras = re.findall(r"<p\b[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = []
    for p in paras:
        p = re.sub(r"\[\d+\]", "", p)
        p = re.sub(r"<[^>]+>", "", p)
        p = p.replace("&amp;", "&").replace("&quot;", '"').replace("&nbsp;", " ")
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 80:  
            cleaned.append(p)

        if sum(len(x) for x in cleaned) > max_chars:
            break

    text = " ".join(cleaned)
    return text[:max_chars].strip()

def scrape_description(url: str) -> str:
    
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return extract_text_from_html(r.text, max_chars=1200)

def call_ollama(description: str, model: str = "llama3.2", retries: int = 2) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": description}
        ],
        "stream": False
    }

    last_error = None

    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)

            if resp.status_code >= 400:
                print("\n[Ollama error body]")
                print(resp.text)

            resp.raise_for_status()

            content = resp.json()["message"]["content"].strip()
            return json.loads(content)#here i'm converting the model's text into a python dictionary

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            last_error = e
            if attempt < retries:
                time.sleep(2)
            else:
                raise RuntimeError(f"Ollama call failed after {retries+1} attempts: {e}") from e

    raise RuntimeError(f"Ollama call failed: {last_error}")

def append_to_csv(csv_path: str, title: str, source_url: str, scraped_text: str, result: dict) -> None:
    row = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "title": title,
        "source_url": source_url,
        "scraped_text": scraped_text,
        "uses_ai": result.get("uses_ai", False),
        "risk_level": result.get("risk_level", ""),
        "top_risks": "|".join(result.get("top_risks", [])),
    }

    fieldnames = [
        "timestamp_utc", "title", "source_url", "scraped_text",
        "uses_ai", "risk_level", "top_risks"
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    out_csv = "ai_risk_dataset.csv"

    for title, url in SOURCES:
        print(f"\n--- Processing: {title} ---")
        try:
            scraped = scrape_description(url)

            model_input = f"""
SYSTEM NAME: {title}
SOURCE URL: {url}

EVIDENCE TEXT (scraped):
{scraped}

Task: Fill the JSON schema for basic AI risk classification.
"""

            result = call_ollama(model_input, model="llama3.2", retries=2)

            print(json.dumps(result, indent=2))
            append_to_csv(out_csv, title, url, scraped, result)

            time.sleep(1)

        except Exception as e:
            print(f"[FAILED] {title}: {e}")

    print(f"\nDone. The results have been saved to: {out_csv}")
