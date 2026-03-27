"""
FlatRouteFinder Global Backend v3
- Switched to Open-Elevation API (more reliable than OpenTopoData)
- Clears stale caches on startup
- Better elevation validation
"""

import os, math, gzip, pickle, hashlib, logging, time, shutil
from flask import Flask, jsonify, request
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import requests

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
COMFORT_GRADE   = 0.02
K               = 2000
ALPHAS          = [1.0, 0.67, 0.33, 0.0]
CACHE_VERSION   = "v3"
GRAPH_CACHE_DIR = f"graph_cache_{CACHE_VERSION}"
MAX_GRADE       = 0.35      # 35% hard cap — anything above is bad data
ELEV_BATCH_SIZE = 100

# Purge old cache dirs so stale data never survives a redeploy
for old in ["graph_cache", "graph_cache_v2"]:
    if os.path.exists(old):
        shutil.rmtree(old, ignore_errors=True)
        log.info(f"Purged old cache: {old}")

os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)


# ── Geometry ───────────────────────────────────────────────────────────────────
def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Elevation ──────────────────────────────────────────────────────────────────
def fetch_elevations_open_elevation(lats, lngs):
    """
    Open-Elevation API — free, no key, POST endpoint accepts up to 512 points.
    More reliable than OpenTopoData for large batches.
    """
    results = [None] * len(lats)
    url = "https://api.open-elevation.com/api/v1/lookup"

    for i in range(0, len(lats), ELEV_BATCH_SIZE):
        blats = lats[i:i + ELEV_BATCH_SIZE]
        blngs = lngs[i:i + ELEV_BATCH_SIZE]
        payload = {"locations": [{"latitude": la, "longitude": lo}
                                  for la, lo in zip(blats, blngs)]}
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json().get("results", [])
                for j, res in enumerate(data):
                    elev = res.get("elevation")
                    if elev is not None:
                        results[i + j] = float(elev)
                break
            except Exception as e:
                log.warning(f"Open-Elevation batch {i} attempt {attempt+1}: {e}")
                time.sleep(2)

    return results


def fetch_elevations_opentopodata(lats, lngs):
    """
    OpenTopoData — fallback if Open-Elevation fails.
    """
    results = [None] * len(lats)
    url = "https://api.opentopodata.org/v1/srtm90m"

    for i in range(0, len(lats), 50):
        blats = lats[i:i + 50]
        blngs = lngs[i:i + 50]
        loc_str = "|".join(f"{la},{lo}" for la, lo in zip(blats, blngs))
        for attempt in range(2):
            try:
                r = requests.get(url, params={"locations": loc_str}, timeout=20)
                r.raise_for_status()
                data = r.json().get("results", [])
                for j, res in enumerate(data):
                    elev = res.get("elevation")
                    if elev is not None:
                        results[i + j] = float(elev)
                break
            except Exception as e:
                log.warning(f"OpenTopoData batch {i} attempt {attempt+1}: {e}")
                time.sleep(1)

    return results


def fetch_elevations(lats, lngs):
    """
    Try Open-Elevation first, fill gaps with OpenTopoData.
    Remaining Nones get the median of valid values.
    """
    log.info(f"Fetching elevations for {len(lats)} nodes …")
    results = fetch_elevations_open_elevation(lats, lngs)

    # Count how many we got
    got = sum(1 for e in results if e is not None)
    log.info(f"Open-Elevation returned {got}/{len(lats)} values")

    # If less than 50% succeeded, try OpenTopoData for the gaps
    if got < len(lats) * 0.5:
        log.info("Falling back to OpenTopoData for missing values …")
        fallback = fetch_elevations_opentopodata(lats, lngs)
        for i, (r, f) in enumerate(zip(results, fallback)):
            if r is None and f is not None:
                results[i] = f
        got2 = sum(1 for e in results if e is not None)
        log.info(f"After fallback: {got2}/{len(lats)} values")

    # Fill remaining Nones with median of valid values
    valid = sorted(e for e in results if e is not None)
    if valid:
        median = valid[len(valid) // 2]
    else:
        median = 0.0
        log.warning("All elevation fetches failed — grades will be 0")

    return [e if e is not None else median for e in results]


# ── Edge weight formula ────────────────────────────────────────────────────────
def enrich_graph(G):
    nodes = list(G.nodes(data=True))
    lats  = [d["y"] for _, d in nodes]
    lngs  = [d["x"] for _, d in nodes]
    ids   = [n for n, _ in nodes]

    elevs = fetch_elevations(lats, lngs)

    elev_min = min(elevs)
    elev_max = max(elevs)
    log.info(f"Elevation range: {elev_min:.1f}m – {elev_max:.1f}m "
             f"(span {elev_max - elev_min:.1f}m)")

    # Sanity check — if span is < 1m the elevation fetch probably failed
    if elev_max - elev_min < 1.0:
        log.warning("Elevation span < 1m — elevation data may be flat/failed")

    for nid, elev in zip(ids, elevs):
        G.nodes[nid]["elevation"] = elev

    ARTERIALS = {"primary", "trunk", "motorway",
                 "primary_link", "trunk_link", "motorway_link"}

    for u, v, k, data in G.edges(keys=True, data=True):
        length = max(float(data.get("length", 1)), 1.0)

        elev_u = float(G.nodes[u].get("elevation") or 0)
        elev_v = float(G.nodes[v].get("elevation") or 0)

        raw_grade = abs(elev_v - elev_u) / length
        grade_abs = min(raw_grade, MAX_GRADE)
        G[u][v][k]["grade_abs"] = grade_abs

        highway = data.get("highway", "")
        if isinstance(highway, list):
            highway = highway[0] if highway else ""
        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        name = name or ""

        arterial_penalty = 2.5  if highway in ARTERIALS             else 1.0
        unnamed_penalty  = 8.0  if not name.strip()                 else 1.0
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

    log.info(f"Downloading OSM graph for bbox …")
    G = ox.graph_from_bbox(
        bbox=(north, south, east, west),
        network_type="walk",
        simplify=True,
    )
    G = enrich_graph(G)

    with gzip.open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Graph cached: {key} ({G.number_of_nodes()} nodes)")
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

        grades.append(float(edge.get("grade_abs", 0)))

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
        result["message"] = f"Flattest option saves {grade_saved}% avg grade, adds {dist_added}mi."
    else:
        result["message"] = "Multiple route options found."

    return jsonify(result)


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": CACHE_VERSION})


# ── Elevation debug endpoint ───────────────────────────────────────────────────
@app.route("/test_elevation")
def test_elevation():
    """Quick check that elevation APIs are reachable."""
    test_lats = [37.7749, 37.8044]
    test_lngs = [-122.4194, -122.2712]
    results = fetch_elevations(test_lats, test_lngs)
    return jsonify({
        "sf_downtown":  results[0],
        "oakland":      results[1],
        "looks_correct": abs(results[0] - 16) < 30   # SF downtown ~16m
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
