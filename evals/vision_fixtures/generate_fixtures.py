"""Generate deterministic synthetic TFT-like observation fixtures (not real game captures)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent
FONT = ImageFont.truetype("arial.ttf", 30)
SMALL = ImageFont.truetype("arial.ttf", 23)


CASES = {
    "full_board.png": [
        "STAGE 4-2    LEVEL 7    HP 35    GOLD 40",
        "BOARD: Ahri [1 star] | Sett [2 star] | Garen [2 star]",
        "BENCH: Ahri x2 | Riven x1",
        "ITEMS: Tear | Sword    COMPLETED: Blue Buff",
        "AUGMENTS: Jeweled Lotus | Cluttered Mind",
    ],
    "hud_only.png": ["STAGE 3-5    LEVEL 6    HP 72    GOLD 31", "HUD-ONLY CROP"],
    "board_no_bench.png": [
        "STAGE 4-1    LEVEL 7    HP 48    GOLD 22",
        "BOARD: Aphelios [2 star] | Brambleback [1 star] | Lux [1 star]",
        "BENCH REGION NOT IN IMAGE",
    ],
    "blurred.png": [
        "STAGE 5-?    LEVEL ?    HP ?8    GOLD ?0",
        "BOARD: Ahrl [? star] | S?tt [2 star]",
        "LOW RESOLUTION / BLURRED",
    ],
    "items.png": [
        "ITEM COMPONENTS: Sword | Glove | Tear",
        "COMPLETED ITEMS: Infinity Edge | Warmog Armor",
        "BOARD: Ahri [2 star] {Blue Buff}",
    ],
    "augments.png": [
        "STAGE 3-2    AUGMENTS PANEL",
        "AUGMENTS: Jeweled Lotus | Cluttered Mind | Tiny Titans",
    ],
    "many_units.png": [
        "LEVEL 9    HP 54    GOLD 18",
        "BOARD: Ahri [2] | Sett [2] | Garen [2] | Lux [1] | Riven [1]",
        "BOARD+: Ezreal [2] | Shen [1] | Kayle [1] | Draven [1]",
        "BENCH: Maokai x1 | Alune x2",
    ],
    "partial_board.png": [
        "LEVEL 7    GOLD 36",
        "BOARD CROP: Ahri [1 star] | Sett [2 star]",
        "HUD HP AND STAGE NOT IN IMAGE / BENCH NOT IN IMAGE",
    ],
}


def main() -> None:
    for name, lines in CASES.items():
        image = Image.new("RGB", (1400, 760), "#152238")
        draw = ImageDraw.Draw(image)
        draw.rectangle((35, 35, 1365, 725), outline="#d5b15c", width=4)
        draw.text((70, 70), "CONTROLLED TFT OBSERVATION FIXTURE", fill="#f2d17c", font=FONT)
        for index, line in enumerate(lines):
            draw.text((70, 160 + index * 85), line, fill="white", font=SMALL)
        if name == "blurred.png":
            image = image.resize((420, 228)).filter(ImageFilter.GaussianBlur(2.2))
        image.save(ROOT / name)


if __name__ == "__main__":
    main()
