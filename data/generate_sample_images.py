"""
Generates the 12 synthetic sample images used by the demo.

NOTE: This sandbox has no internet access to an image host / stock photo
API, so real photographs can't be downloaded here. Instead this script
renders simple, clearly-labeled schematic room scenes with PIL (flat
colored shapes representing furniture/appliances + a caption). They are
good enough to exercise the full pipeline (upload -> vision -> retrieval
-> answer) end to end.

To use real photos instead: just drop 10-15 real indoor photos into
data/sample_images/ (any .png/.jpg) - nothing else in the project needs
to change, since the vision module takes an arbitrary image path.
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 480, 360

SCENES = [
    {
        "file": "01_kitchen_unattended_stove.png",
        "bg": (250, 235, 215),
        "caption": "Kitchen: pot boiling on\nunattended lit stove",
        "shapes": [("rect", (60, 200, 420, 320), (180, 180, 180)),   # counter
                   ("rect", (150, 150, 260, 200), (60, 60, 60)),      # stove
                   ("ellipse", (170, 130, 240, 170), (255, 100, 40)), # flame/pot
                   ],
    },
    {
        "file": "02_stairs_cluttered.png",
        "bg": (235, 235, 245),
        "caption": "Staircase: toys and shoes\nscattered on the steps",
        "shapes": [("polygon", [(60, 320), (420, 320), (420, 260), (60, 60)], (200, 190, 170)),
                   ("rect", (120, 300, 150, 320), (220, 60, 60)),
                   ("rect", (220, 260, 250, 280), (60, 120, 220)),
                   ("ellipse", (300, 230, 330, 250), (250, 210, 40)),
                   ],
    },
    {
        "file": "03_bathroom_wet_floor.png",
        "bg": (225, 245, 250),
        "caption": "Bathroom: wet floor tiles\nnext to the tub",
        "shapes": [("rect", (40, 60, 220, 220), (255, 255, 255)),
                   ("rect", (60, 230, 420, 330), (170, 220, 235)),
                   ("ellipse", (250, 250, 320, 290), (120, 190, 220)),
                   ],
    },
    {
        "file": "04_living_room_loose_cords.png",
        "bg": (240, 230, 210),
        "caption": "Living room: extension cord\nrunning across the walkway",
        "shapes": [("rect", (300, 40, 420, 160), (90, 60, 40)),
                   ("line", [(320, 160), (150, 300), (60, 320)], (30, 30, 30), 6),
                   ("rect", (30, 300, 90, 340), (60, 60, 60)),
                   ],
    },
    {
        "file": "05_bedroom_space_heater_curtains.png",
        "bg": (250, 240, 235),
        "caption": "Bedroom: space heater placed\nright next to curtains",
        "shapes": [("rect", (330, 20, 420, 260), (200, 130, 90)),
                   ("rect", (250, 200, 320, 260), (60, 60, 60)),
                   ("ellipse", (270, 215, 300, 245), (255, 120, 40)),
                   ],
    },
    {
        "file": "06_garage_cluttered_chemicals.png",
        "bg": (220, 220, 220),
        "caption": "Garage: chemical containers\nand tools stacked loosely",
        "shapes": [("rect", (40, 250, 140, 320), (230, 60, 60)),
                   ("rect", (150, 260, 230, 320), (60, 160, 60)),
                   ("line", [(260, 320), (260, 120), (340, 90)], (90, 90, 90), 5),
                   ],
    },
    {
        "file": "07_childroom_small_objects.png",
        "bg": (250, 245, 220),
        "caption": "Child's room: small toy pieces\nand a button battery on the floor",
        "shapes": [("ellipse", (100, 300, 115, 315), (255, 210, 40)),
                   ("ellipse", (140, 305, 150, 315), (200, 60, 60)),
                   ("rect", (250, 60, 380, 160), (150, 200, 240)),
                   ],
    },
    {
        "file": "08_diningroom_candle_curtains.png",
        "bg": (245, 235, 225),
        "caption": "Dining room: lit candle burning\nunattended near curtains",
        "shapes": [("rect", (300, 20, 400, 240), (210, 150, 100)),
                   ("rect", (150, 260, 320, 300), (120, 80, 50)),
                   ("ellipse", (225, 220, 235, 240), (255, 160, 40)),
                   ],
    },
    {
        "file": "09_hallway_exposed_wiring.png",
        "bg": (230, 230, 235),
        "caption": "Hallway: damaged outlet with\nexposed wiring near the floor",
        "shapes": [("rect", (30, 30, 450, 330), (235, 230, 220)),
                   ("rect", (200, 260, 230, 290), (255, 255, 255)),
                   ("line", [(215, 290), (240, 320)], (200, 40, 40), 4),
                   ],
    },
    {
        "file": "10_backyard_pool_no_fence.png",
        "bg": (200, 230, 245),
        "caption": "Backyard: pool area with\nno fence or gate",
        "shapes": [("ellipse", (100, 180, 400, 320), (60, 150, 210)),
                   ("rect", (0, 320, 480, 360), (140, 180, 90)),
                   ],
    },
    {
        "file": "11_office_overloaded_outlet.png",
        "bg": (240, 240, 230),
        "caption": "Home office: power strip\noverloaded with many plugs",
        "shapes": [("rect", (60, 60, 260, 160), (120, 90, 60)),
                   ("rect", (280, 260, 400, 280), (40, 40, 40)),
                   ("line", [(285, 270), (200, 300)], (200, 40, 40), 3),
                   ("line", [(295, 270), (220, 310)], (200, 40, 40), 3),
                   ("line", [(305, 270), (240, 300)], (200, 40, 40), 3),
                   ],
    },
    {
        "file": "12_bedroom_normal_safe.png",
        "bg": (245, 245, 240),
        "caption": "Bedroom: tidy room, cords\nsecured, no visible hazards",
        "shapes": [("rect", (60, 180, 260, 300), (150, 120, 100)),
                   ("rect", (300, 60, 420, 180), (200, 190, 170)),
                   ],
    },
]


def _get_font(size=18):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def draw_scene(scene: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), scene["bg"])
    draw = ImageDraw.Draw(img)

    for shape in scene["shapes"]:
        kind = shape[0]
        if kind == "rect":
            draw.rectangle(shape[1], fill=shape[2])
        elif kind == "ellipse":
            draw.ellipse(shape[1], fill=shape[2])
        elif kind == "polygon":
            draw.polygon(shape[1], fill=shape[2])
        elif kind == "line":
            draw.line(shape[1], fill=shape[2], width=shape[3])

    # caption banner
    draw.rectangle((0, H - 60, W, H), fill=(20, 20, 20))
    font = _get_font(16)
    draw.multiline_text((12, H - 54), scene["caption"], fill=(255, 255, 255), font=font, spacing=4)

    return img


def main():
    for scene in SCENES:
        img = draw_scene(scene)
        out_path = os.path.join(OUT_DIR, scene["file"])
        img.save(out_path)
        print(f"wrote {out_path}")
    print(f"\nGenerated {len(SCENES)} sample images in {OUT_DIR}")


if __name__ == "__main__":
    main()
