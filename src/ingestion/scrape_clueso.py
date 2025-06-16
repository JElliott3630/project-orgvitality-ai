import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urlparse
import requests
import base64
from PIL import Image
import pytesseract
import os
import json
import re

URL = "https://strong-boar.clueso.site/share/4e22f5d6-1e01-488b-b880-8e78c207970d"
OUTDIR = "data/images/clueso_images"
os.makedirs(OUTDIR, exist_ok=True)
OUT_JSON_PATH = "data/processed/clueso_grouped_blocks.json"

STEP_PATTERN = re.compile(r"^step\s*1\b", re.IGNORECASE)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(URL, wait_until="networkidle")

        # Scroll to bottom to lazy-load all content
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

        # Get relevant elements in visual order
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

        step_blocks = []
        current_block = []
        img_counter = 0

        for el in elements:
            is_step_1 = False
            block_item = None

            if el["type"] == "text":
                text = el["text"]
                if not text:
                    continue
                is_step_1 = bool(STEP_PATTERN.match(text.strip()))
                block_item = {"type": "text", "tag": el["tag"], "text": text}

            elif el["type"] == "image":
                src = el["src"]
                alt = el.get("alt")
                ocr_text = ""
                img_filename = None

                try:
                    if src.startswith("data:image"):
                        header, b64data = src.split(",", 1)
                        ext = "png" if "png" in header else "jpg"
                        img_counter += 1
                        img_filename = f"img_{img_counter}.{ext}"
                        img_path = os.path.join(OUTDIR, img_filename)
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(b64data))
                        with Image.open(img_path) as im:
                            ocr_text = pytesseract.image_to_string(im).strip()

                    elif src.startswith("http"):
                        img_filename = os.path.basename(urlparse(src).path)
                        img_path = os.path.join(OUTDIR, img_filename)
                        img_data = requests.get(src).content
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        if img_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                            with Image.open(img_path) as im:
                                ocr_text = pytesseract.image_to_string(im).strip()
                        else:
                            ocr_text = ""
                    else:
                        img_path = "unknown"
                        ocr_text = "Unsupported image src format"

                except Exception as e:
                    img_path = "error"
                    ocr_text = f"Error: {e}"

                block_item = {
                    "type": "image",
                    "img_path": img_path,
                    "alt": alt,
                    "ocr_text": ocr_text
                }

            if is_step_1 and current_block:
                step_blocks.append(current_block)
                current_block = []

            if block_item:
                current_block.append(block_item)

        if current_block:
            step_blocks.append(current_block)

        with open(OUT_JSON_PATH, "w") as f:
            json.dump(step_blocks, f, indent=2)

        print(f"✅ Scraped and grouped {len(step_blocks)} step blocks.")
        print(f"📄 Output saved to: {OUT_JSON_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
