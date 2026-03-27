"""
FlatRouteFinder Global Backend
Elevation-aware pedestrian routing for any city worldwide.
"""

import os, math, gzip, pickle, hashlib, logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import requests

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
COMFORT_GRADE   = 0.02          # 2% grade — feels flat, no penalty
K               = 2000          # quadratic steepness multiplier
ALPHAS          = [1.0, 0.67, 0.33, 0.0]   # shortest → flattest
GRAPH_CACHE_DIR = os.environ.get("GRAPH_CACHE_DIR", "graph_cache")
ELEVATION_URL   = "https://api.opentopodata.org/v1/srtm90m"
ELEV_BATCH_SIZE = 100           # opentopodata max per request

os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

# ── Geometry ───────────────────────────────────────────────────────────────────
def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Elevation ──────────────────────────────────────────────────────────────────
def fetch_elevations(lats: list, lngs: list) -> list:
    """
    Batch-fetch SRTM elevations from OpenTopoData.
    Falls back to 0 on any error so routing always completes.
    """
    elevations = []
    for i in range(0, len(lats), ELEV_BATCH_SIZE):
        blats = lats[i : i + ELEV_BATCH_SIZE]
        blngs = lngs[i : i + ELEV_BATCH_SIZE]
        loc_str = "|".join(f"{la},{lo}" for la, lo in zip(blats, blngs))
        try:
            r = requests.get(ELEVATION_URL, params={"locations": loc_str}, timeout=20)
            r.raise_for_status()
            elevations += [res.get("elevation") or 0 for res in r.json()["results"]]
        except Exception as e:
            log.warning(f"Elevation batch {i} failed: {e} — defaulting to 0")
            elevations += [0] * len(blats)
    return elevations


# ── Edge weight formula ────────────────────────────────────────────────────────
def enrich_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add elevation to every node, then compute grade_abs and impedance on
    every edge.  This is where the physics lives.

    impedance = length × arterial_penalty × unnamed_penalty × (1 + K × excess_grade²)

    The quadratic term captures that humans find steep hills
    disproportionately harder — perceived effort scales super-linearly
    with grade, not linearly.
    """
    nodes = list(G.nodes(data=True))
    lats  = [d["y"] for _, d in nodes]
    lngs  = [d["x"] for _, d in nodes]
    ids   = [n for n, _ in nodes]

    log.info(f"Fetching elevation for {len(ids)} nodes …")
    elevs = fetch_elevations(lats, lngs)
    for nid, elev in zip(ids, elevs):
        G.nodes[nid]["elevation"] = elev

    ARTERIALS = {"primary", "trunk", "motorway", "primary_link", "trunk_link", "motorway_link"}

    for u, v, k, data in G.edges(keys=True, data=True):
        length = max(float(data.get("length", 1)), 1.0)

        elev_u = float(G.nodes[u].get("elevation") or 0)
        elev_v = float(G.nodes[v].get("elevation") or 0)
        raw_grade = abs(elev_v - elev_u) / length
        grade_abs = min(raw_grade, 0.6)  # cap at 60% — steeper = bad data
        G[u][v][k]["grade_abs"] = grade_abs

        highway = data.get("highway", "")
        if isinstance(highway, list):
            highway = highway[0] if highway else ""

        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        name = name or ""

        arterial_penalty = 2.5  if highway in ARTERIALS else 1.0
        unnamed_penalty  = 8.0  if not name.strip()     else 1.0
        brt_penalty      = 10.0 if "bus rapid transit" in name.lower() else 1.0

        excess_grade = max(0.0, grade_abs - COMFORT_GRADE)
        impedance = (
            length
            * arterial_penalty
            * unnamed_penalty
            * brt_penalty
            * (1.0 + K * excess_grade ** 2)
        )
        G[u][v][k]["impedance"] = impedance

    return G


# ── Graph cache ────────────────────────────────────────────────────────────────
def _bbox_key(north, south, east, west) -> str:
    s = f"{round(north,3)}_{round(south,3)}_{round(east,3)}_{round(west,3)}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


def get_graph(lat1, lng1, lat2, lng2) -> nx.MultiDiGraph:
    """
    Return an enriched pedestrian graph covering the route bounding box.
    Graphs are cached to disk — first request for an area is slow (30-90s),
    subsequent requests are fast (< 1s).
    """
    crow_m = haversine_m(lat1, lng1, lat2, lng2)
    buf_deg = max(crow_m * 0.6, 600) / 111_000   # at least 600m buffer

    north = max(lat1, lat2) + buf_deg
    south = min(lat1, lat2) - buf_deg
    east  = max(lng1, lng2) + buf_deg
    west  = min(lng1, lng2) - buf_deg

    key   = _bbox_key(north, south, east, west)
    path  = os.path.join(GRAPH_CACHE_DIR, f"{key}.pkl.gz")

    if os.path.exists(path):
        log.info(f"Graph cache hit: {key}")
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    log.info(f"Downloading graph for N{north:.4f} S{south:.4f} E{east:.4f} W{west:.4f}")
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


# ── Subgraph extraction (ellipse constraint) ───────────────────────────────────
def extract_subgraph(G, start_node, end_node) -> nx.MultiDiGraph:
    """
    Keep only nodes inside the ellipse with foci at start and end.
    budget_factor = 2.0 means: allow routes up to 2× crow-flies distance.
    This keeps the subgraph small for fast Dijkstra.
    """
    BUDGET_FACTOR = 2.0
    sd = G.nodes[start_node]
    ed = G.nodes[end_node]
    crow = haversine_m(sd["y"], sd["x"], ed["y"], ed["x"])
    budget = crow * BUDGET_FACTOR

    keep = set()
    for n, d in G.nodes(data=True):
        d_start = haversine_m(d["y"], d["x"], sd["y"], sd["x"])
        d_end   = haversine_m(d["y"], d["x"], ed["y"], ed["x"])
        if d_start + d_end <= budget:
            keep.add(n)

    keep.add(start_node)
    keep.add(end_node)
    return G.subgraph(keep).copy()


# ── Route statistics ───────────────────────────────────────────────────────────
def compute_route_stats(G, path_nodes: list) -> dict:
    coords  = []
    dist_m  = 0.0
    gain_m  = 0.0
    grades  = []

    for n in path_nodes:
        nd = G.nodes[n]
        coords.append({"lat": nd["y"], "lng": nd["x"]})

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        # Pick the edge with minimum length between parallel edges
        edge = min(G[u][v].values(), key=lambda d: d.get("length", 9999))
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


def routes_are_duplicates(r1: dict, r2: dict) -> bool:
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

    try:
        def nearest_node(graph, lat, lng):
            best_node = None
            best_dist = float('inf')
            for n, d in graph.nodes(data=True):
                dlat = d['y'] - lat
                dlng = d['x'] - lng
                dist = dlat*dlat + dlng*dlng
                if dist < best_dist:
                    best_dist = dist
                    best_node = n
            return best_node

        start_node = nearest_node(G, slat, slng)
        end_node   = nearest_node(G, elat, elng)
    except Exception as exc:
        return jsonify({"error": f"Could not snap to road network: {exc}"}), 500

    subG = extract_subgraph(G, start_node, end_node)

    unique_routes = []
    for alpha in ALPHAS:
        # Compute combined weight for this alpha on every edge of the subgraph
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
            log.warning(f"No path found for alpha={alpha}")
        except Exception as exc:
            log.warning(f"Routing error alpha={alpha}: {exc}")

    if not unique_routes:
        return jsonify({"error": "No walkable route found between these points."}), 404

    # Sort: shortest distance first
    unique_routes.sort(key=lambda r: r["distanceInMiles"])

    # Strip internal fields
    for r in unique_routes:
        r.pop("_alpha", None)

    # ── Single route ───────────────────────────────────────────────────────────
    if len(unique_routes) == 1:
        return jsonify({
            "message":     "The shortest and flattest routes are the same for this trip.",
            "singleRoute": unique_routes[0],
        })

    # ── Multiple routes ────────────────────────────────────────────────────────
    result = {}
    for i, r in enumerate(unique_routes):
        result[f"route{i + 1}"] = r

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


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "cache_dir": GRAPH_CACHE_DIR})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# Pre-warm cache on startup — comment this out if you want faster cold starts
# import threading
# def prewarm():
#     try:
#         get_graph(37.7984, -122.4268, 37.7956, -122.4072)
#     except:
#         pass
# threading.Thread(target=prewarm, daemon=True).start()
