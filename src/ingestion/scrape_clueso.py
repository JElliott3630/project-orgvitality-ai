import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import requests
import base64
from PIL import Image
import pytesseract
import io
import os
import json
import re

URL = "https://strong-boar.clueso.site/share/4e22f5d6-1e01-488b-b880-8e78c207970d"
OUTDIR = "rag_content/technical_user_guides/clueso_images"
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON_PATH = "rag_content/technical_user_guides/clueso_linear_blocks.json"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(URL, wait_until="networkidle")

        # Scroll to bottom to trigger all lazy-loads
        previous_height = 0
        while True:
            current_height = await page.evaluate("""() => {
                window.scrollBy(0, window.innerHeight);
                return document.body.scrollHeight;
            }""")
            if current_height == previous_height:
                break
            previous_height = current_height
            await asyncio.sleep(0.5)

        # Select all <p>, <li>, <h1-4>, <img> in order
        elements = await page.eval_on_selector_all(
            "h1, h2, h3, h4, p, li, img",
            """
            els => els.map(e => {
                if (e.tagName.toLowerCase() === "img") {
                    return {type: "image", src: e.src, alt: e.alt || null}
                } else {
                    return {type: "text", tag: e.tagName, text: e.innerText.trim()}
                }
            })
            """
        )

        blocks = []
        img_counter = 0

        for el in elements:
            if el["type"] == "text":
                if el["text"]:
                    blocks.append({"type": "text", "tag": el["tag"], "text": el["text"]})
            elif el["type"] == "image":
                src = el["src"]
                alt = el.get("alt")
                ocr_text = ""
                img_filename = None

                if src.startswith("data:image"):
                    header, b64data = src.split(",", 1)
                    if "image/png" in header:
                        ext = "png"
                    elif "image/jpeg" in header or "image/jpg" in header:
                        ext = "jpg"
                    else:
                        ext = "img"
                    img_counter += 1
                    img_filename = f"img_{img_counter}.{ext}"
                    img_path = os.path.join(OUTDIR, img_filename)
                    try:
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(b64data))
                        # OCR directly
                        with Image.open(img_path) as im:
                            ocr_text = pytesseract.image_to_string(im).strip()
                    except Exception as e:
                        ocr_text = f"Error: {e}"
                elif src.startswith("http"):
                    img_url = src
                    img_filename = os.path.basename(urlparse(img_url).path)
                    img_path = os.path.join(OUTDIR, img_filename)
                    try:
                        img_data = requests.get(img_url).content
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        # OCR only if it's a png/jpg
                        if img_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                            with Image.open(img_path) as im:
                                ocr_text = pytesseract.image_to_string(im).strip()
                        else:
                            ocr_text = ""
                    except Exception as e:
                        ocr_text = f"Error: {e}"

                blocks.append({
                    "type": "image",
                    "img_path": img_path,
                    "alt": alt,
                    "ocr_text": ocr_text
                })

        # Save all linear blocks to JSON
        with open(OUT_JSON_PATH, "w") as f:
            json.dump(blocks, f, indent=2)

        print(f"Scraped and OCR'd {len([b for b in blocks if b['type']=='image'])} images and {len([b for b in blocks if b['type']=='text'])} text blocks.")
        print(f"See output: {OUT_JSON_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
