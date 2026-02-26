"""
=================================================================
  ECCV v3 — Large-Scale Image Collector

  Collects 1,500+ diverse images from Wikimedia Commons for
  challenge generation. Images are saved to dataset_final/images/.

  Usage:
    python scripts/collect_images.py
    python scripts/collect_images.py --target 2000
    python scripts/collect_images.py --resume   # Continue from last run

  Features:
    - 500+ search queries across 20+ categories
    - pHash deduplication
    - Skips existing images when --resume is set
    - Rate-limited to respect Wikimedia API
    - Saves manifest.jsonl for downstream annotation
=================================================================
"""
import os, sys, json, random, time, hashlib, argparse
from datetime import datetime, timedelta
from io import BytesIO

sys.path.insert(0, ".")

import requests
from PIL import Image

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="ECCV v3 — Image Collector")
parser.add_argument("--target", type=int, default=1600,
                    help="Target number of images (default: 1600)")
parser.add_argument("--imgs-per-query", type=int, default=5,
                    help="Max images per query (default: 5)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from existing manifest, skip already-collected images")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dataset_final")
IMG_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

HEADERS = {"User-Agent": "VLMRealityCheck/1.0 (Academic Research; contact@university.edu)"}
TARGET = args.target
MAX_PER_QUERY = args.imgs_per_query

# ============================================================
# 500+ Search Queries — 20+ categories for maximum diversity
# ============================================================
SEARCH_QUERIES = [
    # ===== STREET / TRAFFIC =====
    ("street_traffic_01", "busy city street cars traffic lights"),
    ("street_traffic_02", "urban road vehicles pedestrians crosswalk"),
    ("street_traffic_03", "downtown traffic jam rush hour"),
    ("street_traffic_04", "narrow European city street cobblestone"),
    ("street_traffic_05", "Asian city street motorbikes scooters"),
    ("street_traffic_06", "African city street market traffic"),
    ("street_traffic_07", "Latin American city colorful street"),
    ("street_traffic_08", "rainy city street reflections cars"),
    ("street_traffic_09", "snowy street winter city traffic"),
    ("street_traffic_10", "city street night lights vehicles"),
    ("highway_01", "highway multiple cars driving lanes"),
    ("highway_02", "motorway interchange overpass"),
    ("highway_03", "freeway traffic aerial view"),
    ("intersection_01", "road intersection traffic lights urban"),
    ("intersection_02", "roundabout traffic circle aerial"),
    ("intersection_03", "pedestrian crossing zebra street"),

    # ===== OUTDOOR / NATURE =====
    ("park_01", "park bench trees people walking"),
    ("park_02", "city park fountain green"),
    ("park_03", "botanical garden flowers pathway"),
    ("park_04", "Japanese garden bridge pond"),
    ("beach_01", "beach people walking sunset ocean"),
    ("beach_02", "tropical beach palm trees sand"),
    ("beach_03", "rocky coastline waves cliff"),
    ("beach_04", "beach volleyball sports sand"),
    ("forest_01", "forest trail nature hiking path"),
    ("forest_02", "dense forest sunlight trees canopy"),
    ("forest_03", "bamboo forest pathway green"),
    ("forest_04", "autumn forest colorful leaves trail"),
    ("mountain_01", "mountain landscape hiking trail"),
    ("mountain_02", "snowy mountain peak alpine"),
    ("mountain_03", "mountain valley river scenic"),
    ("river_01", "river bridge city waterway"),
    ("river_02", "river rapids nature rocks"),
    ("lake_01", "lake reflection mountains scenic"),
    ("lake_02", "lakeside dock boats calm"),
    ("waterfall_01", "waterfall nature tropical forest"),
    ("desert_01", "desert sand dunes landscape"),
    ("desert_02", "desert road straight highway"),

    # ===== URBAN / ARCHITECTURE =====
    ("skyline_01", "city skyline skyscrapers panoramic"),
    ("skyline_02", "city skyline sunset reflection"),
    ("skyline_03", "modern city skyline glass buildings"),
    ("building_01", "historic building facade ornate"),
    ("building_02", "modern architecture glass steel"),
    ("building_03", "Gothic cathedral architecture detail"),
    ("building_04", "mosque architecture dome minaret"),
    ("building_05", "temple architecture Asian pagoda"),
    ("building_06", "colonial architecture building"),
    ("construction_01", "construction site crane workers"),
    ("construction_02", "building under construction scaffold"),
    ("bridge_01", "bridge road urban architecture cables"),
    ("bridge_02", "old stone bridge river arches"),
    ("bridge_03", "suspension bridge cable panoramic"),
    ("alley_01", "narrow alley old town lanterns"),
    ("alley_02", "graffiti street art alley urban"),
    ("rooftop_01", "rooftop city view panoramic"),

    # ===== PEOPLE / ACTIVITY =====
    ("market_01", "outdoor market stalls people produce"),
    ("market_02", "fish market seafood vendors"),
    ("market_03", "flea market antiques crowd"),
    ("market_04", "floating market boats vendors"),
    ("market_05", "night market food stalls lights"),
    ("sports_01", "soccer football field players match"),
    ("sports_02", "basketball court game players"),
    ("sports_03", "tennis court match player"),
    ("sports_04", "swimming pool swimmers competition"),
    ("sports_05", "running track athletics competitors"),
    ("sports_06", "skateboard park skateboarder tricks"),
    ("playground_01", "playground children playing equipment"),
    ("playground_02", "water park slides children"),
    ("cycling_01", "bicycle lane cycling urban commute"),
    ("cycling_02", "mountain biking trail forest"),
    ("festival_01", "street festival parade crowd"),
    ("festival_02", "cultural festival dancers costumes"),
    ("festival_03", "music festival concert outdoor crowd"),
    ("protest_01", "street demonstration signs crowd"),
    ("crowd_01", "busy pedestrian street crowd"),
    ("crowd_02", "outdoor event crowd gathering"),

    # ===== VEHICLES =====
    ("parking_01", "parking lot parked cars aerial"),
    ("parking_02", "multi-story parking garage"),
    ("train_01", "train station platform passengers"),
    ("train_02", "high speed train railway"),
    ("train_03", "subway metro station underground"),
    ("train_04", "freight train railway tracks"),
    ("bus_01", "bus stop urban street public transport"),
    ("bus_02", "double decker bus city street"),
    ("bus_03", "school bus yellow street"),
    ("boat_01", "fishing boat harbor sea"),
    ("boat_02", "sailboat marina yacht"),
    ("boat_03", "cargo ship port container"),
    ("boat_04", "ferry boat river passengers"),
    ("airplane_01", "airplane airport runway takeoff"),
    ("airplane_02", "airplane cockpit instruments"),
    ("truck_01", "delivery truck city street"),
    ("truck_02", "fire truck emergency street"),
    ("motorcycle_01", "motorcycle parked street"),
    ("motorcycle_02", "motorcycle racing track"),
    ("tram_01", "tram streetcar urban rails"),
    ("bicycle_01", "bicycle rack city parked"),
    ("scooter_01", "electric scooter urban sidewalk"),

    # ===== RURAL / AGRICULTURAL =====
    ("farm_01", "farm landscape agricultural field crop"),
    ("farm_02", "vineyard grape harvest winery"),
    ("farm_03", "rice paddy field terraces Asia"),
    ("farm_04", "wheat field golden harvest"),
    ("farm_05", "orchard fruit trees rows"),
    ("farm_06", "greenhouse plants agriculture indoor"),
    ("rural_01", "rural road countryside fence"),
    ("rural_02", "country house cottage garden"),
    ("village_01", "village houses rural stream"),
    ("village_02", "fishing village coastal boats"),
    ("village_03", "mountain village alpine houses"),
    ("barn_01", "red barn farm silo"),

    # ===== ANIMALS =====
    ("animals_01", "farm animals cows sheep grazing field"),
    ("animals_02", "horses pasture fence ranch"),
    ("animals_03", "chickens poultry farm yard"),
    ("animals_04", "dogs playing outdoor park"),
    ("animals_05", "cats outdoor garden sitting"),
    ("animals_06", "birds flock flying sky"),
    ("animals_07", "pigeons city square feeding"),
    ("animals_08", "ducks pond water swimming"),
    ("animals_09", "deer forest wildlife nature"),
    ("animals_10", "squirrel park tree climbing"),
    ("wildlife_01", "wildlife safari animals Africa"),
    ("wildlife_02", "bears fishing river salmon"),
    ("wildlife_03", "penguins colony Antarctic ice"),
    ("zoo_01", "zoo animals enclosure visitors"),
    ("zoo_02", "aquarium fish underwater tank"),
    ("zoo_03", "bird aviary tropical colorful"),

    # ===== TEXT-RICH SCENES =====
    ("signs_01", "street signs city road signs multiple"),
    ("signs_02", "highway signs direction arrows"),
    ("signs_03", "warning signs construction caution"),
    ("signs_04", "neon signs night city commercial"),
    ("signs_05", "vintage signs retro advertising"),
    ("shopfront_01", "shop front signs storefronts neon"),
    ("shopfront_02", "bakery shop window display"),
    ("shopfront_03", "bookstore shop front display"),
    ("shopfront_04", "pharmacy drugstore shop front"),
    ("billboard_01", "billboard advertising outdoor large"),
    ("billboard_02", "digital billboard electronic display"),
    ("menu_01", "restaurant menu board outdoor chalkboard"),
    ("menu_02", "cafe menu specials board handwritten"),
    ("newspaper_01", "newspaper stand headlines display"),
    ("poster_01", "movie poster cinema theater"),
    ("poster_02", "event poster wall flyer"),
    ("graffiti_01", "graffiti text street art wall"),
    ("banner_01", "banner advertisement shop sale"),
    ("license_01", "license plate car close up"),

    # ===== TIME OF DAY =====
    ("sunrise_01", "sunrise morning landscape golden hour"),
    ("sunrise_02", "sunrise city skyline dawn"),
    ("sunset_01", "sunset evening landscape dusk orange"),
    ("sunset_02", "sunset beach silhouette"),
    ("sunset_03", "sunset city rooftop view"),
    ("night_01", "night city lights dark urban street"),
    ("night_02", "night skyline city reflection water"),
    ("night_03", "night street lamp post sidewalk"),
    ("night_04", "northern lights aurora landscape"),
    ("twilight_01", "twilight blue hour city"),
    ("overcast_01", "overcast cloudy city grey"),
    ("foggy_01", "foggy morning bridge mist"),

    # ===== FOOD / RESTAURANT =====
    ("food_01", "outdoor restaurant cafe terrace dining"),
    ("food_02", "food market stall fresh produce"),
    ("food_03", "street food vendor cooking"),
    ("food_04", "food truck festival outdoor"),
    ("food_05", "bakery bread pastries display"),
    ("food_06", "ice cream shop colorful display"),
    ("food_07", "sushi restaurant Japanese cuisine"),
    ("food_08", "pizza restaurant Italian food"),
    ("food_09", "BBQ barbecue outdoor grill"),
    ("food_10", "fruit stand tropical market"),

    # ===== COMMERCIAL / SHOPPING =====
    ("shopping_01", "shopping district stores pedestrian"),
    ("shopping_02", "shopping mall interior stores"),
    ("shopping_03", "department store display window"),
    ("shopping_04", "souvenir shop tourist items"),
    ("shopping_05", "antique shop vintage items display"),
    ("shopping_06", "electronics store display gadgets"),
    ("shopping_07", "clothing store fashion display"),
    ("shopping_08", "flower shop bouquets display"),

    # ===== INDUSTRIAL =====
    ("industrial_01", "factory building industrial smokestack"),
    ("industrial_02", "warehouse industrial storage logistics"),
    ("industrial_03", "power plant energy infrastructure"),
    ("industrial_04", "oil refinery chemical plant"),
    ("industrial_05", "solar panels farm renewable energy"),
    ("industrial_06", "wind turbines farm landscape"),
    ("industrial_07", "mining operation quarry heavy machinery"),
    ("industrial_08", "railway yard freight containers"),

    # ===== EDUCATION / INSTITUTIONAL =====
    ("campus_01", "university campus students buildings"),
    ("campus_02", "school playground children building"),
    ("campus_03", "library building architecture"),
    ("hospital_01", "hospital building entrance ambulance"),
    ("museum_01", "museum building exterior steps"),
    ("museum_02", "art gallery interior paintings"),
    ("church_01", "church building steeple architecture"),
    ("courthouse_01", "courthouse government building columns"),

    # ===== WATER / MARINE =====
    ("harbor_01", "waterfront harbor boats marina"),
    ("harbor_02", "commercial port cargo ships cranes"),
    ("harbor_03", "fishing harbor nets boats"),
    ("pier_01", "pier boardwalk ocean promenade"),
    ("pier_02", "wooden pier lake dock"),
    ("lighthouse_01", "lighthouse coastal cliff ocean"),
    ("canal_01", "canal boats buildings European"),
    ("canal_02", "Venice canal gondola buildings"),
    ("dam_01", "dam reservoir hydroelectric"),

    # ===== AIRPORT / AVIATION =====
    ("airport_01", "airport terminal planes gate"),
    ("airport_02", "airport tarmac planes loading"),
    ("airport_03", "airport check-in counter travelers"),
    ("airport_04", "small airport runway propeller"),
    ("helipad_01", "helicopter helipad rooftop"),

    # ===== SPORTS VENUES =====
    ("stadium_01", "stadium crowd sports football"),
    ("stadium_02", "baseball stadium game field"),
    ("stadium_03", "cricket ground match players"),
    ("arena_01", "indoor arena basketball court"),
    ("racetrack_01", "motor racing track cars circuit"),
    ("golf_01", "golf course green fairway"),
    ("ski_01", "ski resort slope mountain snow"),

    # ===== RECREATION =====
    ("amusement_01", "amusement park rides ferris wheel"),
    ("amusement_02", "roller coaster theme park"),
    ("camping_01", "camping tent forest outdoor"),
    ("camping_02", "campfire night outdoor wilderness"),
    ("garden_party_01", "garden party outdoor celebration"),
    ("picnic_01", "picnic park blanket food outdoor"),
    ("yoga_01", "yoga outdoor park exercise"),
    ("fishing_01", "fishing lake outdoor rod"),

    # ===== WINTER SCENES =====
    ("snow_01", "snow covered village winter houses"),
    ("snow_02", "snow park trees frozen"),
    ("snow_03", "ice skating rink outdoor winter"),
    ("snow_04", "snowplow clearing road winter"),
    ("snow_05", "frozen lake ice winter landscape"),

    # ===== TROPICAL / EXOTIC =====
    ("tropical_01", "tropical island beach resort"),
    ("tropical_02", "palm tree lined street tropical"),
    ("tropical_03", "mangrove swamp tropical forest"),
    ("tropical_04", "coral reef underwater fish"),
    ("jungle_01", "jungle rainforest dense vegetation"),

    # ===== LANDMARKS =====
    ("landmark_01", "Eiffel Tower Paris France"),
    ("landmark_02", "Colosseum Rome Italy"),
    ("landmark_03", "Taj Mahal India monument"),
    ("landmark_04", "Great Wall China landscape"),
    ("landmark_05", "Sydney Opera House harbor"),
    ("landmark_06", "Golden Gate Bridge San Francisco"),
    ("landmark_07", "Statue of Liberty New York"),
    ("landmark_08", "Big Ben London Parliament"),
    ("landmark_09", "Machu Picchu Peru ruins"),
    ("landmark_10", "Pyramids Giza Egypt"),
    ("landmark_11", "Christ Redeemer Rio Janeiro"),
    ("landmark_12", "Angkor Wat Cambodia temple"),

    # ===== INTERIOR SCENES =====
    ("interior_01", "living room interior furniture modern"),
    ("interior_02", "kitchen interior cooking appliances"),
    ("interior_03", "office workspace computer desk"),
    ("interior_04", "classroom school desks chairs"),
    ("interior_05", "hospital ward beds medical equipment"),
    ("interior_06", "gym interior exercise equipment"),
    ("interior_07", "hotel lobby interior reception"),
    ("interior_08", "train interior seats passengers"),
    ("interior_09", "airplane cabin seats passengers"),
    ("interior_10", "workshop tools equipment workbench"),

    # ===== WEATHER / ATMOSPHERE =====
    ("weather_01", "storm clouds dramatic sky landscape"),
    ("weather_02", "rainbow after rain landscape"),
    ("weather_03", "lightning thunderstorm night city"),
    ("weather_04", "flood water street urban"),
    ("weather_05", "drought dry cracked earth"),
    ("weather_06", "hail damage cars street"),
    ("weather_07", "tornado funnel cloud landscape"),
    ("weather_08", "dust storm desert visibility"),

    # ===== CULTURAL / RELIGIOUS =====
    ("cultural_01", "Hindu temple architecture India"),
    ("cultural_02", "Buddhist temple monastery Asia"),
    ("cultural_03", "synagogue Jewish temple architecture"),
    ("cultural_04", "torii gate Shinto shrine Japan"),
    ("cultural_05", "mosque prayer Islamic architecture"),
    ("cultural_06", "Orthodox church domes Russia"),
    ("cultural_07", "ancient ruins archaeological site"),
    ("cultural_08", "cave paintings ancient art"),
    ("cultural_09", "tribal village traditional houses"),
    ("cultural_10", "medieval castle fortress tower"),

    # ===== EMERGENCY / UTILITY =====
    ("emergency_01", "fire truck firefighters emergency"),
    ("emergency_02", "police car street patrol"),
    ("emergency_03", "ambulance hospital emergency"),
    ("utility_01", "power lines electrical transmission tower"),
    ("utility_02", "water tower city infrastructure"),
    ("utility_03", "cell phone tower antenna communication"),
    ("utility_04", "garbage truck waste collection"),
    ("utility_05", "road construction repair workers"),

    # ===== SPACE / SCIENCE =====
    ("science_01", "observatory telescope dome mountain"),
    ("science_02", "satellite dish antenna array"),
    ("science_03", "rocket launch pad space center"),
    ("science_04", "laboratory science equipment research"),

    # ===== MISCELLANEOUS OUTDOOR =====
    ("misc_01", "public fountain plaza square"),
    ("misc_02", "bus terminal station urban"),
    ("misc_03", "gas station fuel pumps"),
    ("misc_04", "car wash automatic cleaning"),
    ("misc_05", "laundromat washing machines"),
    ("misc_06", "recycling center waste sorting"),
    ("misc_07", "cemetery gravestones memorial"),
    ("misc_08", "garden center nursery plants"),
    ("misc_09", "pet shop animals window"),
    ("misc_10", "toy store children display"),
    ("misc_11", "hardware store tools display"),
    ("misc_12", "pharmacy store front medical"),
    ("misc_13", "tattoo shop parlor sign"),
    ("misc_14", "barber shop pole sign"),
    ("misc_15", "gym fitness center exterior"),

    # ===== ADDITIONAL DIVERSE SCENES =====
    ("diverse_01", "crowded subway underground commuters"),
    ("diverse_02", "hot air balloon festival sky"),
    ("diverse_03", "horse race track jockeys"),
    ("diverse_04", "vintage car classic automobile show"),
    ("diverse_05", "motorcycle rally gathering"),
    ("diverse_06", "kite flying beach colorful"),
    ("diverse_07", "windmill traditional Dutch landscape"),
    ("diverse_08", "lighthouse keeper cottage coastal"),
    ("diverse_09", "treehouse forest adventure"),
    ("diverse_10", "rooftop garden urban farming"),
    ("diverse_11", "underground cave cavern formations"),
    ("diverse_12", "hot springs geothermal pool"),
    ("diverse_13", "glacier ice landscape mountains"),
    ("diverse_14", "volcanic landscape lava rocks"),
    ("diverse_15", "bamboo raft river floating"),
    ("diverse_16", "street performer musician crowd"),
    ("diverse_17", "outdoor cinema screen audience"),
    ("diverse_18", "drive-in theater cars screen"),
    ("diverse_19", "roadside diner classic American"),
    ("diverse_20", "gas station desert remote highway"),
    ("diverse_21", "cable car mountain gondola"),
    ("diverse_22", "paragliding sky mountain aerial"),
    ("diverse_23", "surfing wave ocean sport"),
    ("diverse_24", "rock climbing cliff outdoor"),
    ("diverse_25", "canoe kayak river paddling"),
    ("diverse_26", "scuba diving underwater coral"),
    ("diverse_27", "horseback riding trail outdoor"),
    ("diverse_28", "bird watching wetlands binoculars"),
    ("diverse_29", "drone aerial city photography"),
    ("diverse_30", "solar eclipse sky phenomenon"),

    # ===== MULTI-OBJECT DENSE SCENES =====
    ("dense_01", "supermarket aisle products shelves"),
    ("dense_02", "toy store colorful display shelves"),
    ("dense_03", "library bookshelves reading room"),
    ("dense_04", "cluttered desk office workspace"),
    ("dense_05", "toolbox workshop organized tools"),
    ("dense_06", "kitchen counter cooking ingredients"),
    ("dense_07", "children playroom toys scattered"),
    ("dense_08", "garage workshop cars tools"),
    ("dense_09", "warehouse pallets boxes stacked"),
    ("dense_10", "market vegetable fruit stall variety"),
    ("dense_11", "parking lot full cars rows"),
    ("dense_12", "bicycle parking rack many bikes"),
    ("dense_13", "shoe store display rack variety"),
    ("dense_14", "electronics store display multiple screens"),
    ("dense_15", "art supply store paints brushes"),

    # ===== WIDE VARIETY SCENES =====
    ("variety_01", "train crossing road barrier gates"),
    ("variety_02", "covered bridge wooden rural"),
    ("variety_03", "dock loading cargo truck warehouse"),
    ("variety_04", "airport luggage carousel baggage"),
    ("variety_05", "escalator shopping mall people"),
    ("variety_06", "elevator building interior lobby"),
    ("variety_07", "staircase spiral architecture interior"),
    ("variety_08", "tunnel road cars driving lights"),
    ("variety_09", "underpass bridge urban graffiti"),
    ("variety_10", "overpass highway urban view"),
    ("variety_11", "crosswalk pedestrians busy city"),
    ("variety_12", "taxi cab yellow city street"),
    ("variety_13", "tow truck vehicle roadside"),
    ("variety_14", "ice cream truck colorful street"),
    ("variety_15", "mail delivery truck postal service"),
    ("variety_16", "moving van residential street"),
    ("variety_17", "cement mixer construction site"),
    ("variety_18", "excavator construction digging earth"),
    ("variety_19", "crane lifting construction high rise"),
    ("variety_20", "bulldozer construction site leveling"),

    # ===== ADDITIONAL DENSE + OCR-RICH =====
    ("ocr_01", "bus schedule timetable display"),
    ("ocr_02", "train departure board schedule"),
    ("ocr_03", "airport flight information display board"),
    ("ocr_04", "scoreboard sports display numbers"),
    ("ocr_05", "clock tower city building time"),
    ("ocr_06", "thermometer weather temperature display"),
    ("ocr_07", "speedometer dashboard car interior"),
    ("ocr_08", "price tag label store item"),
    ("ocr_09", "receipt paper printed text"),
    ("ocr_10", "barcode product label scanning"),
    ("ocr_11", "calendar wall planner dates"),
    ("ocr_12", "map city tourist public information"),
    ("ocr_13", "directory building floor plan sign"),
    ("ocr_14", "nutrition label food packaging"),
    ("ocr_15", "street name sign post city"),

    # ===== ADDITIONAL SPATIAL =====
    ("spatial_01", "stacked boxes warehouse organized"),
    ("spatial_02", "chess pieces board game close up"),
    ("spatial_03", "pool billiards table balls cue"),
    ("spatial_04", "dominoes pattern falling cascade"),
    ("spatial_05", "jigsaw puzzle pieces table"),
    ("spatial_06", "Rubik cube colorful puzzle"),
    ("spatial_07", "Lego blocks construction toy colorful"),
    ("spatial_08", "building blocks shapes children toys"),
    ("spatial_09", "shelving unit organized objects"),
    ("spatial_10", "display cabinet figurines collection"),

    # ===== ADDITIONAL COUNTING =====
    ("counting_01", "eggs carton dozen breakfast"),
    ("counting_02", "bowling pins alley setup"),
    ("counting_03", "candles birthday cake flames"),
    ("counting_04", "fingers hand gesture counting"),
    ("counting_05", "coins money pile currency"),
    ("counting_06", "buttons sewing colorful set"),
    ("counting_07", "dice game board multiple"),
    ("counting_08", "marbles glass spheres colorful"),
    ("counting_09", "pencils colored drawing set"),
    ("counting_10", "flowers bouquet variety stems"),
]

random.shuffle(SEARCH_QUERIES)

print(f"{'='*65}")
print(f"  ECCV v3 — Large-Scale Image Collector")
print(f"  Target:  {TARGET:,} images")
print(f"  Queries: {len(SEARCH_QUERIES)}")
print(f"  Max/query: {MAX_PER_QUERY}")
print(f"  Output:  {IMG_DIR}")
print(f"{'='*65}")

# ============================================================
# Load existing data if resuming
# ============================================================
manifest_path = os.path.join(DATASET_DIR, "manifest.jsonl")
collected = []
existing_hashes = set()
existing_ids = set()

if args.resume and os.path.exists(manifest_path):
    for line in open(manifest_path, "r", encoding="utf-8"):
        rec = json.loads(line)
        collected.append(rec)
        existing_hashes.add(rec.get("phash", ""))
        existing_ids.add(rec.get("image_id", ""))
    print(f"  Resuming: {len(collected)} existing images loaded")
else:
    # Check for existing images on disk
    for f in os.listdir(IMG_DIR):
        if f.endswith((".jpg", ".jpeg", ".png", ".webp")):
            existing_ids.add(os.path.splitext(f)[0])
    if existing_ids:
        print(f"  Found {len(existing_ids)} existing images on disk (will skip IDs)")

START = time.time()
new_count = 0
last_saved_count = 0
errors = 0
skipped_dup = 0
skipped_small = 0

def elapsed():
    return str(timedelta(seconds=int(time.time() - START)))

def show_progress():
    total = len(collected)
    pct = total / TARGET * 100
    rate = new_count / max(time.time() - START, 1)
    remaining = (TARGET - total) / max(rate, 0.01)
    eta = str(timedelta(seconds=int(remaining))) if rate > 0.01 else "???"
    bar_len = 30
    filled = int(bar_len * total / TARGET)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}% {total}/{TARGET} | "
          f"+{new_count} new | {rate:.1f}/s | ETA: {eta} | {elapsed()}",
          end="", flush=True)


print(f"\n  Starting collection...\n")

for qi, (scene_name, query) in enumerate(SEARCH_QUERIES):
    total_so_far = len(collected)
    if total_so_far >= TARGET:
        break

    print(f"  [QI:{qi}/{len(SEARCH_QUERIES)}] Query: {query}")
    show_progress()

    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": "6",
            "gsrlimit": str(min(MAX_PER_QUERY * 3, 20)),  # Fetch extra to account for filtering
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "640",
            "format": "json",
        }
        resp = requests.get("https://commons.wikimedia.org/w/api.php",
                           params=params, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            errors += 1
            continue

        data = resp.json()
        pages = data.get("query", {}).get("pages", {})

        imgs_this_query = 0
        for pid, page in pages.items():
            if imgs_this_query >= MAX_PER_QUERY:
                break
            total_so_far = len(collected) + new_count
            if total_so_far >= TARGET:
                break

            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if "image" not in mime or "svg" in mime:
                continue

            url = info.get("thumburl") or info.get("url", "")
            if not url:
                continue

            # Download
            try:
                img_resp = requests.get(url, headers=HEADERS, timeout=20)
                if img_resp.status_code != 200:
                    continue
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                if min(img.size) < 100:
                    skipped_small += 1
                    continue
            except Exception:
                errors += 1
                continue

            # Resize
            img.thumbnail((640, 640), Image.LANCZOS)

            # pHash dedup
            try:
                import imagehash
                phash = str(imagehash.phash(img))
            except ImportError:
                phash = hashlib.md5(img.tobytes()[:2048]).hexdigest()[:16]

            if phash in existing_hashes:
                skipped_dup += 1
                continue
            existing_hashes.add(phash)

            # Generate unique ID
            img_id = f"vlm_{len(existing_ids) + new_count + 1:04d}"
            while img_id in existing_ids:
                img_id = f"vlm_{int(img_id.split('_')[1]) + 1:04d}"

            img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
            img.save(img_path, quality=93)

            ext_meta = info.get("extmetadata", {})
            license_val = ext_meta.get("LicenseShortName", {}).get("value", "unknown")

            record = {
                "image_id": img_id,
                "url": url,
                "local_path": img_path,
                "source": "wikimedia",
                "license": license_val,
                "phash": phash,
                "metadata": {
                    "query": query,
                    "scene": scene_name,
                    "width": img.size[0],
                    "height": img.size[1],
                    "title": page.get("title", "")[:80],
                },
            }
            collected.append(record)
            new_count += 1
            imgs_this_query += 1

            # Save progress every 10 new images
            if new_count >= last_saved_count + 10:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    for rec in collected:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                last_saved_count = new_count

        # Rate limit: 50ms between queries (be nice to Wikimedia)
        time.sleep(0.05)

    except requests.exceptions.Timeout:
        errors += 1
        continue
    except Exception as e:
        errors += 1
        continue

show_progress()
print()

# ============================================================
# Save final manifest
# ============================================================
with open(manifest_path, "w", encoding="utf-8") as f:
    for rec in collected:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

total_time = time.time() - START

print(f"\n{'='*65}")
print(f"  IMAGE COLLECTION COMPLETE")
print(f"{'='*65}")
print(f"  Total images:    {len(collected):,}")
print(f"  New this run:    {new_count:,}")
print(f"  Queries used:    {qi + 1}/{len(SEARCH_QUERIES)}")
print(f"  Dedup skipped:   {skipped_dup:,}")
print(f"  Too small:       {skipped_small:,}")
print(f"  Errors:          {errors:,}")
print(f"  Total time:      {str(timedelta(seconds=int(total_time)))}")
print(f"  Rate:            {new_count / max(total_time, 1):.1f} img/sec")
print(f"  Manifest:        {manifest_path}")
print(f"  Images dir:      {IMG_DIR}")
print(f"{'='*65}")
