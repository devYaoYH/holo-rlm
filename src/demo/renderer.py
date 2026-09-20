"""Stable fixture renderer used as the exact model visual input in harness runs."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def render_fixture(config: dict, state: dict) -> Image.Image:
    width = int(config["viewport"]["width"])
    height = int(config["viewport"]["height"])
    layout = config["layout"]
    image = Image.new("RGB", (width, height), "#eef3f8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bold = ImageFont.load_default()
    if state["details_open"]:
        selected_id = state.get("selected_hotel_id") or config["target_id"]
        selected = next(hotel for hotel in config["hotels"] if hotel["id"] == selected_id)
        header_height = layout["header_height"]
        draw.rectangle((0, 0, width, header_height), fill="#102b4e")
        draw.text((48, 30), "StayLocal   LOCAL FIXTURE", fill="white", font=bold)
        panel_left = max(40, layout["content_left"])
        panel_right = min(width - 40, panel_left + layout["card_width"])
        draw.rounded_rectangle((panel_left, 155, panel_right, min(height - 80, 550)), radius=14, fill="white", outline="#d7e0ea")
        draw.text((panel_left + 55, 215), "LOCAL DETAILS PAGE", fill="#0e7490", font=bold)
        draw.text((panel_left + 55, 260), selected["name"], fill="#172033", font=bold)
        draw.text((panel_left + 55, 310), "SUCCESS: local details open", fill="#116466", font=bold)
        draw.text((panel_left + 55, 350), "No booking, payment, account, or external navigation is available.", fill="#607086", font=font)
        return image

    scroll = int(state["scroll_y"])
    left = layout["content_left"]
    intro_y = max(layout["header_height"] + 34, layout["first_card_y"] - 138) - scroll
    draw.text((left, intro_y), "DETERMINISTIC SEARCH", fill="#0e7490", font=bold)
    draw.text((left, intro_y + 30), "Places to stay in Harbor City", fill="#172033", font=bold)
    item_label = config.get("item_id", f"seed-{config.get('seed', 0)}")
    draw.text((left, intro_y + 62), f"Item {item_label} | synthetic prices | no real reservations", fill="#5f6f84", font=font)
    stride = layout["card_height"] + layout["card_gap"]
    for hotel in config["hotels"]:
        top = layout["first_card_y"] + hotel["index"] * stride - scroll
        bottom = top + layout["card_height"]
        if bottom < layout["header_height"] or top > height:
            continue
        right = left + layout["card_width"]
        image_right = left + layout["image_width"]
        draw.rounded_rectangle((left, top, right, bottom), radius=12, fill="white", outline="#d7e0ea")
        draw.rectangle((left, top, image_right, bottom), fill=hotel["colors"][0])
        draw.text((left + layout["image_width"] // 2 - 12, top + layout["card_height"] // 2 - 8), hotel["initials"], fill="white", font=bold)
        draw.text((left + layout["name_x_offset"], top + layout["name_y_offset"]), hotel["name"], fill="#172033", font=bold)
        draw.text((left + layout["name_x_offset"], top + layout["description_y_offset"]), hotel["description"], fill="#607086", font=font)
        draw.text((left + layout["rating_x_offset"], top + layout["rating_y_offset"]), f"{hotel['rating']} / 10", fill="#116466", font=bold)
        draw.text((left + layout["price_x_offset"], top + layout["price_y_offset"]), f"{hotel.get('currency', '$')}{hotel['price']}", fill="#172033", font=bold)
        button_left = left + layout["button_x_offset"]
        button_top = top + layout["button_y_offset"]
        draw.rounded_rectangle(
            (
                button_left,
                button_top,
                button_left + layout["button_width"],
                button_top + layout["button_height"],
            ),
            radius=7,
            fill="#0d6e73",
        )
        draw.text((button_left + 10, button_top + max(10, layout["button_height"] // 2 - 6)), "View details", fill="white", font=bold)
    # The browser fixture's sticky header has z-index 5, so it must occlude
    # scrolled cards rather than being painted underneath them.
    draw.rectangle((0, 0, width, layout["header_height"]), fill="#102b4e")
    draw.text((32, max(18, layout["header_height"] // 2 - 7)), "StayLocal   LOCAL FIXTURE", fill="white", font=bold)
    draw.text((max(360, width - 330), max(18, layout["header_height"] // 2 - 7)), "Harbor City | 2 guests | 3 nights", fill="#d7e9f8", font=font)
    panel_width = min(212, width // 5)
    panel_left = width - panel_width - 16
    panel_top = height - 92
    draw.rounded_rectangle((panel_left, panel_top, width - 16, height - 16), radius=8, fill="#ffffff", outline="#a7b8c9")
    draw.text((panel_left + 12, panel_top + 11), "Test state", fill="#172033", font=bold)
    draw.text((panel_left + 12, panel_top + 31), item_label, fill="#172033", font=font)
    draw.text((panel_left + 12, panel_top + 49), f"scrollY={scroll}", fill="#172033", font=font)
    draw.text((panel_left + 12, panel_top + 65), "details=false", fill="#172033", font=font)
    return image


def target_click(config: dict, state: dict) -> tuple[int, int]:
    return hotel_click(config, state, config["target_id"])


def hotel_click(config: dict, state: dict, hotel_id: str) -> tuple[int, int]:
    hotel = next(hotel for hotel in config["hotels"] if hotel["id"] == hotel_id)
    layout = config["layout"]
    top = (
        layout["first_card_y"]
        + hotel["index"] * (layout["card_height"] + layout["card_gap"])
        - state["scroll_y"]
    )
    return (
        layout["content_left"] + layout["button_x_offset"] + layout["button_width"] // 2,
        top + layout["button_y_offset"] + layout["button_height"] // 2,
    )


def target_scroll(config: dict) -> int:
    return hotel_scroll(config, config["target_id"])


def hotel_scroll(config: dict, hotel_id: str) -> int:
    hotel = next(hotel for hotel in config["hotels"] if hotel["id"] == hotel_id)
    layout = config["layout"]
    return min(
        config.get("max_scroll", 2400),
        max(
            0,
            layout["first_card_y"]
            + hotel["index"] * (layout["card_height"] + layout["card_gap"])
            - layout["header_height"]
            - 82,
        ),
    )
