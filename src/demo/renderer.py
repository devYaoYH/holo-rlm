"""Stable fixture renderer used as the exact model visual input in harness runs."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def render_fixture(config: dict, state: dict) -> Image.Image:
    width = int(config["viewport"]["width"])
    height = int(config["viewport"]["height"])
    image = Image.new("RGB", (width, height), "#eef3f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bold = ImageFont.load_default()
    if state["details_open"]:
        draw.rectangle((0, 0, width, 78), fill="#102b4e")
        draw.text((48, 30), "StayLocal   LOCAL FIXTURE", fill="white", font=bold)
        draw.rounded_rectangle((160, 155, 1120, 550), radius=14, fill="white", outline="#d7e0ea")
        draw.text((215, 215), "LOCAL DETAILS PAGE", fill="#0e7490", font=bold)
        draw.text((215, 260), "Harbor Lantern Hotel", fill="#172033", font=bold)
        draw.text((215, 310), "SUCCESS: local details open", fill="#116466", font=bold)
        draw.text((215, 350), "No booking, payment, account, or external navigation is available.", fill="#607086", font=font)
        return image

    scroll = int(state["scroll_y"])
    intro_y = 112 - scroll
    draw.text((160, intro_y), "DETERMINISTIC SEARCH", fill="#0e7490", font=bold)
    draw.text((160, intro_y + 30), "Places to stay in Harbor City", fill="#172033", font=bold)
    draw.text((160, intro_y + 62), f"Seed {config['seed']} | fixed prices | no real reservations", fill="#5f6f84", font=font)
    for hotel in config["hotels"]:
        top = 250 + hotel["index"] * 262 - scroll
        bottom = top + 244
        if bottom < 78 or top > height:
            continue
        draw.rounded_rectangle((160, top, 1120, bottom), radius=12, fill="white", outline="#d7e0ea")
        draw.rectangle((160, top, 430, bottom), fill=hotel["colors"][0])
        draw.text((275, top + 105), hotel["initials"], fill="white", font=bold)
        draw.text((465, top + 35), hotel["name"], fill="#172033", font=bold)
        draw.text((465, top + 70), hotel["description"], fill="#607086", font=font)
        draw.text((948, top + 35), f"{hotel['rating']} / 10", fill="#116466", font=bold)
        draw.text((1028, top + 105), f"${hotel['price']}", fill="#172033", font=bold)
        draw.rounded_rectangle((1030, top + 160, 1135, top + 225), radius=7, fill="#0d6e73")
        draw.text((1048, top + 185), "View details", fill="white", font=bold)
    # The browser fixture's sticky header has z-index 5, so it must occlude
    # scrolled cards rather than being painted underneath them.
    draw.rectangle((0, 0, width, 78), fill="#102b4e")
    draw.text((48, 26), "StayLocal   LOCAL FIXTURE", fill="white", font=bold)
    draw.text((940, 26), "Harbor City | 2 guests | 3 nights", fill="#d7e9f8", font=font)
    draw.rounded_rectangle((1052, 690, 1264, 782), radius=8, fill="#ffffff", outline="#a7b8c9")
    draw.text((1065, 703), "Test state", fill="#172033", font=bold)
    draw.text((1065, 725), f"seed={config['seed']}", fill="#172033", font=font)
    draw.text((1065, 745), f"scrollY={scroll}", fill="#172033", font=font)
    draw.text((1065, 765), "details=false", fill="#172033", font=font)
    return image


def target_click(config: dict, state: dict) -> tuple[int, int]:
    target = next(hotel for hotel in config["hotels"] if hotel["id"] == config["target_id"])
    top = 250 + target["index"] * 262 - state["scroll_y"]
    return 1080, top + 190


def target_scroll(config: dict) -> int:
    target = next(hotel for hotel in config["hotels"] if hotel["id"] == config["target_id"])
    return max(0, 250 + target["index"] * 262 - 160)
