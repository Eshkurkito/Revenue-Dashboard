from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BRAND = (0x2e, 0x48, 0x5f)  # #2e485f
WHITE = (255, 255, 255)
LIGHT = (243, 246, 249)
DARK  = (31, 41, 55)

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "assets" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

def rounded(draw, bbox, radius, outline, fill=None, width=2):
    try:
        draw.rounded_rectangle(bbox, radius=radius, outline=outline, fill=fill, width=width)
    except Exception:
        draw.rectangle(bbox, outline=outline, fill=fill, width=width)

def text(draw, xy, s, fill, size=28):
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, s, fill=fill, font=font)

def card_base(w=640, h=360):
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    pad = 10
    rounded(d, (pad, pad, w - pad, h - pad), radius=16, outline=(229,231,235), fill=LIGHT, width=2)
    d.rectangle((pad, pad, w - pad, pad + 8), fill=BRAND)
    return img, d

def icon_bars(d, w, h):
    cx, cy = w // 2, h // 2 + 20
    bar_w, gap = 40, 28
    heights = [90, 150, 120]
    start_x = cx - (bar_w * 3 + gap * 2) // 2
    for i, ht in enumerate(heights):
        x0 = start_x + i * (bar_w + gap)
        y0 = cy - ht // 2
        d.rectangle((x0, y0, x0 + bar_w, cy + ht // 2), fill=BRAND, outline=BRAND)

def icon_magnifier(d, w, h):
    cx, cy, r = w // 2 - 20, h // 2 + 10, 70
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=BRAND, width=10)
    d.line((cx + r - 5, cy + r - 5, cx + r + 70, cy + r + 70), fill=BRAND, width=12)

def icon_sliders(d, w, h):
    left, right = 140, w - 140
    ys = [h // 2 - 60, h // 2, h // 2 + 60]
    knobs = [left + 180, left + 80, left + 240]
    for y, k in zip(ys, knobs):
        d.line((left, y, right, y), fill=BRAND, width=8)
        d.ellipse((k - 20, y - 20, k + 20, y + 20), fill=WHITE, outline=BRAND, width=6)

def build_card(filename: str, title: str, icon_fn):
    img, d = card_base()
    w, h = img.size
    icon_fn(d, w, h)
    text(d, (24, 24), title, fill=DARK, size=28)
    text(d, (24, 60), "Florit Flats", fill=BRAND, size=22)
    img.save(IMG_DIR / filename, format="PNG")

def build_logo():
    w, h = 640, 160
    img = Image.new("RGB", (w, h), BRAND)
    d = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("arial.ttf", 48)
        font_sub = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_big = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    d.text((24, 36), "Florit Flats", fill=WHITE, font=font_big)
    d.text((26, 96), "Revenue Dashboard", fill=(230, 240, 245), font=font_sub)
    img.save(IMG_DIR / "florit-flats-logo.png", format="PNG")

def main():
    build_logo()
    build_card("consulta.png", "Consulta normal", icon_bars)
    build_card("pro.png", "Cuadro de mando PRO", icon_magnifier)
    build_card("whatif.png", "What‑if", icon_sliders)
    print(f"Imágenes creadas en: {IMG_DIR}")

if __name__ == "__main__":
    main()