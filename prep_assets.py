from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import json

root = Path('/workspace/lily-pad-jump')
scene = Image.open(root / 'public/scene.png').convert('RGBA')
w, h = scene.size
arr = np.asarray(scene)
R, G, B = [arr[:, :, i].astype(np.int16) for i in range(3)]
pink = (R > 185) & (G < 175) & (B > 95) & (R > G + 35)
gold = (R > 175) & (G > 135) & (B < 115) & (R > B + 45)
# include dark outline near pink
char = pink | gold
m = Image.fromarray((char.astype(np.uint8) * 255), 'L').filter(ImageFilter.MaxFilter(5))
char = np.asarray(m) > 0
# expand once more lightly for outlines
m = Image.fromarray((char.astype(np.uint8) * 255), 'L').filter(ImageFilter.MaxFilter(3))
char = np.asarray(m) > 0
alpha_img = Image.fromarray((char.astype(np.uint8) * 255), 'L').filter(ImageFilter.GaussianBlur(1.5))
alpha = np.asarray(alpha_img).astype(np.float32) / 255.0

ys, xs = np.where(char)
pad = 4
x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
x1, y1 = min(w - 1, int(xs.max()) + pad), min(h - 1, int(ys.max()) + pad)
print('bbox', x0, y0, x1, y1, 'feet_y_pct', (y1 / h) * 100)

crop = scene.crop((x0, y0, x1 + 1, y1 + 1))
mask = Image.fromarray((char[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255), 'L')
sprite = Image.new('RGBA', crop.size, (0, 0, 0, 0))
sprite.paste(crop, (0, 0), mask)
sprite.save(root / 'public/character.png')

# Better playfield: telea-like fill using side samples per row (no ghost)
pa = arr.astype(np.float32).copy()
for y in range(y0, y1 + 1):
    # find left/right non-char samples on this row
    row_mask = char[y]
    xs_row = np.where(~row_mask)[0]
    for x in range(x0, x1 + 1):
        a = float(alpha[y, x])
        if a < 0.02:
            continue
        # nearest outside pixels left/right
        left = x - 1
        while left >= 0 and char[y, left]:
            left -= 1
        right = x + 1
        while right < w and char[y, right]:
            right += 1
        cols = []
        if left >= 0:
            cols.append(arr[y, left, :3])
        if right < w:
            cols.append(arr[y, right, :3])
        # also below for feet area
        if y + 6 < h and not char[min(h-1, y + 6), x]:
            cols.append(arr[min(h-1, y + 6), x, :3])
        if not cols:
            continue
        col = np.mean(cols, axis=0)
        pa[y, x, :3] = pa[y, x, :3] * (1 - a) + col * a

play = Image.fromarray(pa.astype(np.uint8), 'RGBA')
# blur only patched zone
blur = play.crop((x0, y0, x1 + 1, y1 + 1)).filter(ImageFilter.GaussianBlur(1.5))
# mix slight blur
orig_patch = play.crop((x0, y0, x1 + 1, y1 + 1))
mixed = Image.blend(orig_patch, blur, 0.55)
play.paste(mixed, (x0, y0))
play.save(root / 'public/scene-play.png')

feet_pct = (y1 / h) * 100
center_x = ((x0 + x1) / 2) / w * 100
meta = {
    'spriteOrigin': {'x': x0, 'y': y0, 'w': x1 - x0 + 1, 'h': y1 - y0 + 1},
    'stage': {'w': w, 'h': h},
    'startPct': {'x': center_x, 'y': feet_pct},  # feet anchor
    'feetAnchor': True,
}
(root / 'public/layout.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
