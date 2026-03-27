"""
FlatRouteFinder Global Backend v4
- Uses srtm.py to download elevation tiles directly (no API key, no rate limits)
- Grade capped at 35% max
- Cache version v4 forces fresh rebuild
"""

import os, math, gzip, pickle, hashlib, logging, shutil
from flask import Flask, jsonify, request
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import srtm

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
COMFORT_GRADE   = 0.02
K               = 2000
ALPHAS          = [1.0, 0.67, 0.33, 0.0]
CACHE_VERSION   = "v4"
GRAPH_CACHE_DIR = f"graph_cache_{CACHE_VERSION}"
MAX_GRADE       = 0.35      # 35% hard cap

# Purge all old cache versions on startup
for old in ["graph_cache", "graph_cache_v2", "graph_cache_v3"]:
    if os.path.exists(old):
        shutil.rmtree(old, ignore_errors=True)
        log.info(f"Purged old cache: {old}")

os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

# Load SRTM elevation data (downloads tiles on first use, caches locally)
log.info("Loading SRTM elevation data …")
ELEVATION_DATA = srtm.get_data()
log.info("SRTM ready")


# ── Geometry ───────────────────────────────────────────────────────────────────
def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Elevation ──────────────────────────────────────────────────────────────────
def get_elevation(lat, lng):
    """
    Get elevation in metres from local SRTM tiles.
    Returns 0 if data unavailable for this location.
    """
    try:
        elev = ELEVATION_DATA.get_elevation(lat, lng)
        if elev is None:
            return 0.0
        return float(elev)
    except Exception:
        return 0.0


# ── Edge weight formula ────────────────────────────────────────────────────────
def enrich_graph(G):
    """
    Add SRTM elevation to every node, compute grade_abs and impedance
    on every edge.

    impedance = length × penalties × (1 + K × excess_grade²)

    The quadratic term reflects real human physiology: a 10% hill feels
    4× harder than a 5% hill. This is the core of the product.
    """
    nodes = list(G.nodes(data=True))
    log.info(f"Adding elevation to {len(nodes)} nodes …")

    elevs = []
    for nid, data in nodes:
        elev = get_elevation(data["y"], data["x"])
        G.nodes[nid]["elevation"] = elev
        elevs.append(elev)

    valid = [e for e in elevs if e != 0.0]
    if valid:
        log.info(f"Elevation range: {min(valid):.1f}m – {max(valid):.1f}m "
                 f"(span {max(valid)-min(valid):.1f}m, {len(valid)}/{len(elevs)} valid)")
    else:
        log.warning("All elevations are 0 — SRTM tiles may not cover this area")

    ARTERIALS = {"primary", "trunk", "motorway",
                 "primary_link", "trunk_link", "motorway_link"}

    for u, v, k, data in G.edges(keys=True, data=True):
        length = max(float(data.get("length", 1)), 1.0)

        elev_u = float(G.nodes[u].get("elevation") or 0)
        elev_v = float(G.nodes[v].get("elevation") or 0)

        raw_grade = abs(elev_v - elev_u) / length
        grade_abs = min(raw_grade, MAX_GRADE)   # cap at 35%
        G[u][v][k]["grade_abs"] = grade_abs

        highway = data.get("highway", "")
        if isinstance(highway, list):
            highway = highway[0] if highway else ""

        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        name = name or ""

        arterial_penalty = 2.5  if highway in ARTERIALS                else 1.0
        unnamed_penalty  = 8.0  if not name.strip()                    else 1.0
        brt_penalty      = 10.0 if "bus rapid transit" in name.lower() else 1.0

        excess_grade = max(0.0, grade_abs - COMFORT_GRADE)
        impedance = (
            length
            * arterial_penalty
            * unnamed_penalty
            * brt_penalty
            * (1.0 + K * excess_grade**2)
        )
        G[u][v][k]["impedance"] = impedance

    return G


# ── Graph cache ────────────────────────────────────────────────────────────────
def _bbox_key(north, south, east, west):
    s = f"{round(north,3)}_{round(south,3)}_{round(east,3)}_{round(west,3)}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


def get_graph(lat1, lng1, lat2, lng2):
    crow_m  = haversine_m(lat1, lng1, lat2, lng2)
    buf_deg = max(crow_m * 0.6, 600) / 111_000

    north = max(lat1, lat2) + buf_deg
    south = min(lat1, lat2) - buf_deg
    east  = max(lng1, lng2) + buf_deg
    west  = min(lng1, lng2) - buf_deg

    key  = _bbox_key(north, south, east, west)
    path = os.path.join(GRAPH_CACHE_DIR, f"{key}.pkl.gz")

    if os.path.exists(path):
        log.info(f"Cache hit: {key}")
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    log.info(f"Downloading OSM graph …")
    G = ox.graph_from_bbox(
        bbox=(north, south, east, west),
        network_type="walk",
        simplify=True,
    )
    G = enrich_graph(G)

    with gzip.open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Cached: {key} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


# ── Nearest node ───────────────────────────────────────────────────────────────
def nearest_node(G, lat, lng):
    best_node = None
    best_dist = float('inf')
    for n, d in G.nodes(data=True):
        dist = (d['y'] - lat)**2 + (d['x'] - lng)**2
        if dist < best_dist:
            best_dist = dist
            best_node = n
    return best_node


# ── Subgraph extraction ────────────────────────────────────────────────────────
def extract_subgraph(G, start_node, end_node):
    sd = G.nodes[start_node]
    ed = G.nodes[end_node]
    crow = haversine_m(sd["y"], sd["x"], ed["y"], ed["x"])
    budget = crow * 2.0

    keep = {start_node, end_node}
    for n, d in G.nodes(data=True):
        if (haversine_m(d["y"], d["x"], sd["y"], sd["x"]) +
            haversine_m(d["y"], d["x"], ed["y"], ed["x"])) <= budget:
            keep.add(n)

    return G.subgraph(keep).copy()


# ── Route statistics ───────────────────────────────────────────────────────────
def compute_route_stats(G, path_nodes):
    coords = []
    dist_m = 0.0
    gain_m = 0.0
    grades = []

    for n in path_nodes:
        nd = G.nodes[n]
        coords.append({"lat": nd["y"], "lng": nd["x"]})

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge   = min(G[u][v].values(), key=lambda d: d.get("length", 9999))
        length = float(edge.get("length", 0))
        dist_m += length

        elev_u = float(G.nodes[u].get("elevation") or 0)
        elev_v = float(G.nodes[v].get("elevation") or 0)
        if elev_v > elev_u:
            gain_m += elev_v - elev_u

        # Always re-cap when reading back — defensive against old cached data
        raw_grade = float(edge.get("grade_abs", 0))
        grades.append(min(raw_grade, MAX_GRADE))

    avg_grade = sum(grades) / len(grades) if grades else 0.0
    max_grade = max(grades)               if grades else 0.0

    return {
        "coordinates":      coords,
        "distanceInMiles":  round(dist_m * 0.000621371, 3),
        "distanceInMeters": round(dist_m, 1),
        "avgGradePct":      round(avg_grade * 100, 2),
        "maxGradePct":      round(max_grade * 100, 2),
        "elevationGainFt":  round(gain_m * 3.28084, 1),
        "elevationGainM":   round(gain_m, 1),
    }


def routes_are_duplicates(r1, r2):
    return (
        abs(r1["distanceInMiles"] - r2["distanceInMiles"]) < 0.05 and
        abs(r1["avgGradePct"]     - r2["avgGradePct"])     < 0.3  and
        abs(r1["maxGradePct"]     - r2["maxGradePct"])     < 1.0
    )


# ── Route endpoint ─────────────────────────────────────────────────────────────
@app.route("/route")
def route():
    try:
        slat = float(request.args["start_lat"])
        slng = float(request.args["start_lng"])
        elat = float(request.args["end_lat"])
        elng = float(request.args["end_lng"])
    except (KeyError, ValueError) as exc:
        return jsonify({"error": f"Bad parameters: {exc}"}), 400

    try:
        G = get_graph(slat, slng, elat, elng)
    except Exception as exc:
        log.exception("Graph build failed")
        return jsonify({"error": f"Could not load map data: {exc}"}), 500

    start_node = nearest_node(G, slat, slng)
    end_node   = nearest_node(G, elat, elng)

    subG = extract_subgraph(G, start_node, end_node)

    unique_routes = []
    for alpha in ALPHAS:
        for u, v, k in subG.edges(keys=True):
            d = subG[u][v][k]
            subG[u][v][k]["_w"] = (
                alpha * d.get("length", 1) + (1 - alpha) * d.get("impedance", 1)
            )
        try:
            path  = nx.shortest_path(subG, start_node, end_node, weight="_w")
            stats = compute_route_stats(subG, path)
            stats["_alpha"] = alpha
            if not any(routes_are_duplicates(stats, r) for r in unique_routes):
                unique_routes.append(stats)
        except nx.NetworkXNoPath:
            log.warning(f"No path for alpha={alpha}")
        except Exception as exc:
            log.warning(f"Routing error alpha={alpha}: {exc}")

    if not unique_routes:
        return jsonify({"error": "No walkable route found."}), 404

    unique_routes.sort(key=lambda r: r["distanceInMiles"])

    # Drop routes more than 60% longer than shortest
    min_dist = unique_routes[0]["distanceInMiles"]
    unique_routes = [r for r in unique_routes if r["distanceInMiles"] <= min_dist * 1.6]

    for r in unique_routes:
        r.pop("_alpha", None)

    if len(unique_routes) == 1:
        return jsonify({
            "message":     "The shortest and flattest routes are the same for this trip.",
            "singleRoute": unique_routes[0],
        })

    result = {}
    for i, r in enumerate(unique_routes):
        result[f"route{i+1}"] = r

    shortest = unique_routes[0]
    flattest = min(unique_routes, key=lambda r: r["avgGradePct"])
    if flattest is not shortest:
        grade_saved = round(shortest["avgGradePct"] - flattest["avgGradePct"], 1)
        dist_added  = round(flattest["distanceInMiles"] - shortest["distanceInMiles"], 2)
        result["message"] = (
            f"Flattest option saves {grade_saved}% avg grade, adds {dist_added}mi."
        )
    else:
        result["message"] = "Multiple route options found."

    return jsonify(result)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": CACHE_VERSION})


# ── Elevation test ─────────────────────────────────────────────────────────────
@app.route("/test_elevation")
def test_elevation():
    sf   = get_elevation(37.7749, -122.4194)   # SF downtown ~16m
    twin = get_elevation(37.7544, -122.4477)   # Twin Peaks  ~280m
    return jsonify({
        "sf_downtown_m":  sf,
        "twin_peaks_m":   twin,
        "span_m":         round(twin - sf, 1),
        "looks_correct":  twin > sf + 100      # Twin Peaks should be ~260m higher
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
