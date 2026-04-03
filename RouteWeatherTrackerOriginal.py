import time
import math
from datetime import datetime, timedelta
import requests
import pandas as pd
import os


km_to_miles = 0.621371
miles_to_km = 1.0 / km_to_miles

USER_AGENT = "route-towns-weather/1.0 (Jackson Negus: jacksonegus2021@gmail.com)"  # set a real contact if you deploy/share

# --------- helpers ---------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def nominatim_search(query: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"Could not geocode: {query}")
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", query)

def nominatim_reverse(lat: float, lon: float):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    addr = data.get("address", {})

    town = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("hamlet")
        or addr.get("suburb")
        or addr.get("locality")
    )

    if not town:
        return None, None

    state = addr.get("state") or addr.get("region") or ""
    return town, state

def osrm_route(start_lat, start_lon, end_lat, end_lon):
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        raise ValueError(f"OSRM routing failed: {data}")

    route = data["routes"][0]
    coords = route["geometry"]["coordinates"]  # list of [lon, lat]
    duration_s = float(route["duration"])
    distance_m = float(route["distance"])

    legs = route.get("legs", [])
    steps = []
    for leg in legs:
        steps.extend(leg.get("steps", []))

    return coords, duration_s, distance_m, steps


def sample_route_points(coords_lonlat, every_km=10.0):
    # returns list of (lat, lon, distance_km_from_start)
    points = []
    dist_accum = 0.0
    next_mark = 0.0

    # seed with start
    lon0, lat0 = coords_lonlat[0]
    points.append((lat0, lon0, 0.0))

    for i in range(1, len(coords_lonlat)):
        lon1, lat1 = coords_lonlat[i - 1]
        lon2, lat2 = coords_lonlat[i]
        seg_km = haversine_km(lat1, lon1, lat2, lon2)
        if seg_km <= 0:
            continue

        while next_mark + every_km <= dist_accum + seg_km:
            target = next_mark + every_km
            frac = (target - dist_accum) / seg_km
            lat = lat1 + (lat2 - lat1) * frac
            lon = lon1 + (lon2 - lon1) * frac
            points.append((lat, lon, target))
            next_mark = target

        dist_accum += seg_km

    # add end point
    lonE, latE = coords_lonlat[-1]
    points.append((latE, lonE, dist_accum))
    return points

def step_points_from_osrm(steps):
    # Returns list of (lat, lon) points representing maneuver locations
    pts = []
    for st in steps:
        man = st.get("maneuver", {})
        loc = man.get("location")  # [lon, lat]
        if loc and len(loc) == 2:
            lon, lat = loc
            pts.append((lat, lon))
    return pts

def nws_hourly_forecast(lat, lon):
    # NWS: first get gridpoint endpoint
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    p = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=30)
    p.raise_for_status()
    props = p.json()["properties"]
    forecast_hourly_url = props["forecastHourly"]

    f = requests.get(forecast_hourly_url, headers=headers, timeout=30)
    f.raise_for_status()
    periods = f.json()["properties"]["periods"]
    return periods  # list with startTime, temperature, shortForecast, etc.

def pick_forecast_for_eta(periods, eta_dt):
    # find hourly period whose startTime is closest to eta_dt
    best = None
    best_diff = None
    for pr in periods:
        start = pr.get("startTime")
        if not start:
            continue
        # NWS times include offset, parse with fromisoformat
        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        diff = abs((dt - eta_dt).total_seconds())
        if best is None or diff < best_diff:
            best = pr
            best_diff = diff
    return best
def route_cumulative_dist_km(coords_lonlat):
    # returns list cumdist_km aligned with coords_lonlat indices
    cum = [0.0]
    for i in range(1, len(coords_lonlat)):
        lon1, lat1 = coords_lonlat[i - 1]
        lon2, lat2 = coords_lonlat[i]
        cum.append(cum[-1] + haversine_km(lat1, lon1, lat2, lon2))
    return cum

def nearest_cumdist_km(lat, lon, coords_lonlat, cumdist_km):
    # find nearest route vertex, return its cumulative distance
    best_i = 0
    best_d = float("inf")
    for i, (rlon, rlat) in enumerate(coords_lonlat):
        d = haversine_km(lat, lon, rlat, rlon)
        if d < best_d:
            best_d = d
            best_i = i
    return cumdist_km[best_i]

# --------- main program ---------
def build_town_weather_table(start_query, end_query, depart_local_str):

    # depart_local_str example: "2026-01-05 08:00"
    depart_local = datetime.strptime(depart_local_str, "%Y-%m-%d %H:%M")

    start_lat, start_lon, start_name = nominatim_search(start_query)
    time.sleep(1)
    end_lat, end_lon, end_name = nominatim_search(end_query)
    time.sleep(1)

    coords, duration_s, distance_m, steps = osrm_route(start_lat, start_lon, end_lat, end_lon)
    step_pts = step_points_from_osrm(steps)  # list of (lat, lon)

    # --- Build sparse points first ---
    total_km = distance_m / 1000.0
    total_miles = total_km * km_to_miles
    cumdist_km = route_cumulative_dist_km(coords)

    sparse_pts = []
    SPARSE_KM = 15  # ~25 miles*Changed to 15 km for denser coverage (Change?)

    dist = 0.0
    next_mark = SPARSE_KM

    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        seg_km = haversine_km(lat1, lon1, lat2, lon2)

        if seg_km <= 0:
            continue

        while dist + seg_km >= next_mark:
            frac = (next_mark - dist) / seg_km
            lat = lat1 + (lat2 - lat1) * frac
            lon = lon1 + (lon2 - lon1) * frac
            sparse_pts.append((lat, lon))
            next_mark += SPARSE_KM

        dist += seg_km

    # --- Merge step points + sparse points, then dedupe once ---
    all_pts = step_pts + sparse_pts

    seen = set()
    deduped = []
    for lat, lon in all_pts:
        k = (round(lat, 4), round(lon, 4))  # ~10–15m
        if k in seen:
            continue
        seen.add(k)
        deduped.append((lat, lon))

    step_pts = deduped


    towns_by_label = {}  # key: "Town, State" → best (closest) row
    speed_km_per_s = total_km / duration_s if duration_s > 0 else 0

    # Cache forecasts by town to avoid repeated calls
    forecast_cache = {}

    reverse_cache = {}

    MIN_KM_BETWEEN_REVERSE = 10.0  # start with 3 km; adjust later (Change?)
    last_rev_lat = None
    last_rev_lon = None


    for lat, lon in step_pts:
         # ---- Step B: distance gate ----
        if last_rev_lat is not None:
            moved_km = haversine_km(last_rev_lat, last_rev_lon, lat, lon)
            if moved_km < MIN_KM_BETWEEN_REVERSE:
                continue

        km_from_start = nearest_cumdist_km(lat, lon, coords, cumdist_km)
        # round coordinates to reduce duplicate reverse geocode calls
        cache_key = (round(lat, 2), round(lon, 2))  # ~1 km, reduces reverse calls a lot (Change?)

        if cache_key in reverse_cache:
            town, state = reverse_cache[cache_key]
        else:
            town, state = nominatim_reverse(lat, lon)
            reverse_cache[cache_key] = (town, state)
            time.sleep(1)  # only sleep on real Nominatim calls

        if town is None:
            continue

        last_rev_lat, last_rev_lon = lat, lon


        label = f"{town}, {state}".strip().strip(",")

        # ETA estimation using average speed (simple first version)
        eta = depart_local + timedelta(seconds=(km_from_start / speed_km_per_s) if speed_km_per_s else 0)
        # Treat ETA as local naive; NWS gives zoned times. For a first version, we will compare as naive UTC offset later.
        # Easiest practical approach: assume your machine local time aligns; for multi-timezone routes you will refine later.

        weather = None
        temp = None

        if label not in forecast_cache:
            try:
                periods = nws_hourly_forecast(lat, lon)
                forecast_cache[label] = periods
                time.sleep(0.25)
            except Exception:
                forecast_cache[label] = None

        periods = forecast_cache.get(label)
        if periods:
            # Convert eta to an aware datetime by attaching local offset from the first period if possible
            try:
                first_dt = datetime.fromisoformat(periods[0]["startTime"].replace("Z", "+00:00"))
                # If first_dt is aware, make eta aware using its tzinfo
                eta_aware = eta.replace(tzinfo=first_dt.tzinfo)
                pick = pick_forecast_for_eta(periods, eta_aware)
                if pick:
                    temp = pick.get("temperature")
                    weather = pick.get("shortForecast")
            except Exception:
                pass

            miles_from_start = round(km_from_start * km_to_miles, 1)

            row = {
                "Town": town,
                "State": state,
                "Approx Route Miles": miles_from_start,
                "ETA (local machine time)": eta.strftime("%Y-%m-%d %H:%M"),
                "Temp (F)": temp,
                "Weather": weather,
            }

            # Keep the closest occurrence of each town
            if label not in towns_by_label or miles_from_start < towns_by_label[label]["Approx Route Miles"]:
                towns_by_label[label] = row


    meta = {
        "Start": start_name,
        "Destination": end_name,
        "Depart (local)": depart_local_str,
        "Route Distance (miles)": round(total_miles, 1),
        "Route Duration (hr)": round(duration_s / 3600.0, 2),
        "Signal points": "OSRM steps (maneuvers)",
    }

    # Sort towns by distance from start
    sorted_rows = sorted(
        towns_by_label.values(),
        key=lambda r: r["Approx Route Miles"]
    )

    return meta, pd.DataFrame(sorted_rows)


# --------- run + save ---------
start = input("Start location (e.g., Boise, ID): ").strip()
dest = input("Destination (e.g., Spokane, WA): ").strip()
depart = input("Departure local time (YYYY-MM-DD HH:MM): ").strip()

meta, df = build_town_weather_table(start, dest, depart)

safe_dest = dest.lower().replace(",", "").replace(" ", "_")

output_dir = r"C:\Users\negus\OneDrive\Jackson - Personal\VS_Code\Routes"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = os.path.join(output_dir, f"route_to_{safe_dest}_{timestamp}.xlsx")

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    pd.DataFrame([meta]).to_excel(writer, sheet_name="Meta", index=False)
    df.to_excel(writer, sheet_name="Towns", index=False)

print("Created:", out_file)
