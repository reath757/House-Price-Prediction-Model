import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.property.co.zw/houses-for-sale"
SITEMAP_URL = "https://www.property.co.zw/content/sitemap/advert-en.xml"
SITE_ROOT = "https://www.property.co.zw"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "houses.csv"
IMAGES_DIR = ROOT / "images"
USER_AGENT = "ZimbabweHousePriceResearch/1.0 (academic research)"

CSV_FIELDS = [
    "listing_id",
    "property_url",
    "title",
    "price_text",
    "price_amount",
    "currency",
    "location_text",
    "suburb",
    "city",
    "bedrooms",
    "bathrooms",
    "floor_area_m2",
    "land_area_m2",
    "property_type",
    "description",
    "amenities_json",
    "photo_ids_json",
    "photo_urls_json",
    "photo_paths_json",
    "collected_at_utc",
]


class PoliteClient:
    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds
        self.last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get(self, url):
        wait = self.delay_seconds - (time.monotonic() - self.last_request_at)
        if self.last_request_at and wait > 0:
            time.sleep(wait)

        self.last_request_at = time.monotonic()
        response = self.session.get(url, timeout=30)
        if response.status_code == 429:
            raise RuntimeError(
                "The site returned HTTP 429. Stopping; wait before trying again."
            )
        response.raise_for_status()
        return response


def load_robots(client):
    response = client.get(urljoin(SITE_ROOT, "/robots.txt"))
    parser = RobotFileParser()
    parser.set_url(urljoin(SITE_ROOT, "/robots.txt"))
    parser.parse(response.text.splitlines())
    return parser


def require_allowed(robots, url):
    if not robots.can_fetch(USER_AGENT, url):
        raise PermissionError(f"robots.txt disallows this URL: {url}")


def get_sitemap_listing_urls(xml_content):
    root = ElementTree.fromstring(xml_content)
    namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {
        element.text.strip()
        for element in root.findall(".//sitemap:loc", namespace)
        if element.text
        and (
            "/for-sale/houses-" in element.text
            or "/for-sale/townhouses-" in element.text
        )
    }
    return sorted(urls)


def get_listing_id(card, property_url):
    card_id = card.get("id")
    if card_id:
        return card_id

    route_id = re.search(r"(\d+)(?:/)?$", urlparse(property_url).path)
    if route_id:
        return route_id.group(1)

    return hashlib.sha256(property_url.encode("utf-8")).hexdigest()[:16]


def parse_price(text):
    if not text:
        return None, None

    match = re.search(
        r"\b(USD|US\$|ZAR|ZWL)\s*\$?\s*([\d,]+(?:\.\d+)?)"
        r"|(?<!\w)\$\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None, None

    currency = match.group(1)
    amount_text = match.group(2) or match.group(3)
    if currency:
        currency = "USD" if currency.upper() == "US$" else currency.upper()

    return currency, float(amount_text.replace(",", ""))


def parse_cards(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = []

    for card in soup.select(".ResultCardItem"):
        heading = card.select_one("h2")
        link = heading.select_one('a[href^="/for-sale/"]') if heading else None
        if not link:
            continue

        property_url = urljoin(SITE_ROOT, link["href"])
        listing_id = get_listing_id(card, property_url)
        heading_container = heading.parent
        detail_divs = heading_container.find_all("div", recursive=False)

        location_text = detail_divs[0].get_text(" ", strip=True) if detail_divs else ""
        property_type_text = (
            detail_divs[1].get_text(" ", strip=True) if len(detail_divs) > 1 else ""
        )
        type_match = re.match(
            r"\s*\d+\s+Bedroom\s+(.+?)\s*$", property_type_text, re.IGNORECASE
        )

        price_text = ""
        for price_link in card.select('a[href^="/for-sale/"]'):
            candidate = price_link.get_text(" ", strip=True)
            if parse_price(candidate)[1] is not None:
                price_text = candidate
                break

        currency, price_amount = parse_price(price_text)
        cards.append(
            {
                "listing_id": listing_id,
                "property_url": property_url,
                "title": link.get_text(" ", strip=True),
                "price_text": price_text,
                "price_amount": price_amount,
                "currency": currency,
                "location_text": location_text,
                "property_type": type_match.group(1) if type_match else property_type_text,
            }
        )

    return cards


def first_integer(element):
    if not element:
        return None
    match = re.search(r"\d+", element.get_text(" ", strip=True))
    return int(match.group()) if match else None


def parse_measurement(text):
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*m(?:²|2)", text, re.IGNORECASE)
    return float(match.group(1).replace(",", "")) if match else None


def get_photo_records(soup):
    photos = []
    seen_urls = set()

    for anchor in soup.select('a[href*="/uploadedfiles/"]'):
        photo_url = urljoin(SITE_ROOT, anchor.get("href", ""))
        path = urlparse(photo_url).path
        if not path.lower().endswith(".webp") or "_thumbnail" in path.lower():
            continue
        if photo_url in seen_urls:
            continue

        seen_urls.add(photo_url)
        photos.append(
            {
                "photo_id": Path(path).stem,
                "photo_url": photo_url,
            }
        )

    return photos


def parse_detail(html, card):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    if not title:
        raise ValueError(f"No listing title found at {card['property_url']}")

    price_element = soup.select_one("#priceVal")
    price_text = (
        price_element.get_text(" ", strip=True)
        if price_element
        else card["price_text"]
    )
    currency, price_amount = parse_price(price_text)
    if currency is None:
        currency = card["currency"]
    if price_amount is None:
        price_amount = card["price_amount"]

    address = soup.select_one(".address")
    location_text = address.get_text(" ", strip=True) if address else card["location_text"]
    location_parts = [part.strip() for part in location_text.split(",") if part.strip()]
    suburb = location_parts[0] if location_parts else ""
    city = location_parts[-1] if len(location_parts) >= 2 else ""
    if len(location_parts) == 2:
        city = re.sub(r"\s+(North|South|East|West|Central)$", "", city, flags=re.I)

    floor_area = None
    land_area = None
    for area in soup.select(".area"):
        value = parse_measurement(area.get_text(" ", strip=True))
        if value is None:
            continue
        image = area.find("img")
        if image and image.get("alt", "").strip().lower() == "land area":
            land_area = value
        elif floor_area is None:
            floor_area = value

    description = ""
    description_heading = next(
        (
            heading
            for heading in soup.find_all(["h2", "h3"])
            if heading.get_text(" ", strip=True).lower() == "description"
        ),
        None,
    )
    if description_heading:
        description_block = description_heading.find_next(
            class_="article"
        )
        if description_block:
            description = description_block.get_text(" ", strip=True)

    amenities = []
    amenities_heading = next(
        (
            heading
            for heading in soup.find_all(["h2", "h3"])
            if heading.get_text(" ", strip=True).lower() == "amenities"
        ),
        None,
    )
    if amenities_heading and amenities_heading.parent:
        for image in amenities_heading.parent.select("img[alt]"):
            label = image.get("alt", "").strip()
            if label and label.lower() not in {"amenities"}:
                amenities.append(label)

    return {
        **card,
        "title": title.get_text(" ", strip=True),
        "price_text": price_text,
        "price_amount": price_amount,
        "currency": currency,
        "location_text": location_text,
        "suburb": suburb,
        "city": city,
        "bedrooms": first_integer(soup.select_one(".bed")),
        "bathrooms": first_integer(soup.select_one(".bath")),
        "floor_area_m2": floor_area,
        "land_area_m2": land_area,
        "description": description,
        "amenities": sorted(set(amenities)),
        "photos": get_photo_records(soup),
    }


def download_photo(client, robots, listing_id, photo, images_per_listing):
    if images_per_listing <= 0:
        return []

    local_paths = []
    for item in photo[:images_per_listing]:
        require_allowed(robots, item["photo_url"])
        filename = f"{listing_id}_{item['photo_id']}.webp"
        image_path = IMAGES_DIR / filename

        if not image_path.exists():
            response = client.get(item["photo_url"])
            if not response.headers.get("Content-Type", "").lower().startswith("image/"):
                raise ValueError(f"Expected an image from {item['photo_url']}")
            if len(response.content) > 25 * 1024 * 1024:
                raise ValueError(f"Image exceeds 25 MB: {item['photo_url']}")
            image_path.write_bytes(response.content)

        local_paths.append(image_path.relative_to(ROOT).as_posix())

    return local_paths


def load_existing_ids(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            row["listing_id"]
            for row in csv.DictReader(file)
            if row.get("listing_id")
        }


def save_record(path, record):
    is_new_file = not path.exists() or path.stat().st_size == 0
    row = {
        **record,
        "amenities_json": json.dumps(record["amenities"], ensure_ascii=False),
        "photo_ids_json": json.dumps(
            [photo["photo_id"] for photo in record["photos"]], ensure_ascii=False
        ),
        "photo_urls_json": json.dumps(
            [photo["photo_url"] for photo in record["photos"]], ensure_ascii=False
        ),
        "photo_paths_json": json.dumps(record["photo_paths"], ensure_ascii=False),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        file.flush()


def scrape_sitemap(max_listings, images_per_listing, delay_seconds):
    client = PoliteClient(delay_seconds)
    robots = load_robots(client)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(DATA_PATH)
    seen_ids = set(existing_ids)
    saved_count = 0

    require_allowed(robots, SITEMAP_URL)
    sitemap_response = client.get(SITEMAP_URL)
    listing_urls = get_sitemap_listing_urls(sitemap_response.content)
    print(f"Found {len(listing_urls)} for-sale house and townhouse URLs in sitemap.")
    if not listing_urls:
        raise RuntimeError(f"No house listing URLs found in {SITEMAP_URL}")
    skipped_count = 0

    for index, property_url in enumerate(listing_urls, start=1):
        listing_id = get_listing_id({}, property_url)
        if listing_id in seen_ids:
            skipped_count += 1
            continue

        route = urlparse(property_url).path.rsplit("/", 1)[-1]
        property_type = "Townhouse" if route.startswith("townhouses-") else "House"
        card = {
            "listing_id": listing_id,
            "property_url": property_url,
            "title": "",
            "price_text": "",
            "price_amount": None,
            "currency": None,
            "location_text": "",
            "property_type": property_type,
        }

        require_allowed(robots, property_url)
        try:
            detail_response = client.get(property_url)
        except requests.RequestException as error:
            print(f"Could not fetch listing {listing_id} ({property_url}): {error}")
            continue

        record = parse_detail(detail_response.text, card)
        record["photo_paths"] = download_photo(
            client, robots, record["listing_id"], record["photos"], images_per_listing
        )
        save_record(DATA_PATH, record)
        seen_ids.add(listing_id)
        saved_count += 1
        print(f"[{index}/{len(listing_urls)}] Saved {listing_id}: {record['title']}")

        if max_listings and saved_count >= max_listings:
            print(f"Reached the requested limit of {max_listings} new listings.")
            return

    print(
        f"Finished. Added {saved_count} listings; skipped {skipped_count} IDs "
        f"already in the CSV. Dataset: {DATA_PATH}"
    )


def scrape_search_pages(max_pages, max_listings, images_per_listing, client, robots):
    seen_ids = load_existing_ids(DATA_PATH)
    saved_count = 0

    for page_number in range(1, max_pages + 1):
        page_url = f"{BASE_URL}?page={page_number}"
        require_allowed(robots, page_url)
        response = client.get(page_url)
        soup = BeautifulSoup(response.text, "html.parser")
        result_summary = soup.select_one(".result-description")
        if result_summary:
            summary = result_summary.get_text(" ", strip=True)
            total_match = re.search(r"of\s+([\d,]+)\s+properties", summary, re.I)
            if total_match and page_number == 1:
                total = int(total_match.group(1).replace(",", ""))
                print(f"Currently advertised properties: {total}")
                if total < 6000:
                    print("Notice: this search currently has fewer than 6,000 listings.")

        cards = parse_cards(response.text)
        if not cards:
            print(f"No listing cards on page {page_number}; stopping pagination.")
            break

        fresh_cards = [card for card in cards if card["listing_id"] not in seen_ids]
        if not fresh_cards:
            print(f"No new listings on page {page_number}; continuing to the next page.")
            continue

        print(f"Page {page_number}: {len(fresh_cards)} new listings")
        for card in fresh_cards:
            require_allowed(robots, card["property_url"])
            detail_response = client.get(card["property_url"])
            record = parse_detail(detail_response.text, card)
            record["photo_paths"] = download_photo(
                client, robots, record["listing_id"], record["photos"], images_per_listing
            )
            save_record(DATA_PATH, record)
            seen_ids.add(card["listing_id"])
            saved_count += 1
            print(f"Saved {record['listing_id']}: {record['title']}")

            if max_listings and saved_count >= max_listings:
                print(f"Reached the requested limit of {max_listings} new listings.")
                return

    print(f"Finished. Added {saved_count} listings to {DATA_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect permitted Zimbabwe house listing details and photos."
    )
    parser.add_argument(
        "--source",
        choices=("sitemap", "pages"),
        default="sitemap",
        help="Use the site's advertised listing sitemap (default) or search pages.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum result pages to visit (default: 200).",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=0,
        help="Maximum new listings to save; 0 means all available listings.",
    )
    parser.add_argument(
        "--images-per-listing",
        type=int,
        default=1,
        help="Maximum photos to download per listing; 0 records IDs and URLs only.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Minimum seconds between requests (must be at least 1).",
    )
    args = parser.parse_args()

    if args.max_pages < 1 or args.max_listings < 0 or args.images_per_listing < 0:
        parser.error("page count must be positive; listing and image limits cannot be negative")
    if args.delay < 1.0:
        parser.error("--delay must be at least 1 second")

    if args.source == "sitemap":
        scrape_sitemap(args.max_listings, args.images_per_listing, args.delay)
    else:
        client = PoliteClient(args.delay)
        robots = load_robots(client)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        scrape_search_pages(
            args.max_pages,
            args.max_listings,
            args.images_per_listing,
            client,
            robots,
        )


if __name__ == "__main__":
    main()
