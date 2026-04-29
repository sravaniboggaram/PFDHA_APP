import tempfile
from pathlib import Path

def generate_leaflet_html(coordinates):
    if coordinates is None or coordinates.empty:
        return "<html><body><p>No coordinates provided.</p></body></html>"

    first_row = coordinates.iloc[0]
    center_lat = first_row['latitude']
    center_lon = first_row['longitude']

    profile_data = []
    for _, row in coordinates.iterrows():
        profile_data.append({
            "lat": row["latitude"],
            "lon": row["longitude"],
            "profile_id": str(row["Profile ID"]),
            "disp": round(row["rescaled disp"], 3),
            "width": round(row["actual width"], 3)
        })
    profile_json = str(profile_data).replace("'", '"')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Leaflet Map Viewer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
        <style>
            html, body, #container {{
                height: 100%;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            #container {{
                display: flex;
                height: 100%;
            }}
            #sidebar {{
                width: 320px;
                padding: 10px;
                overflow-y: auto;
                background: #f8f8f8;
                border-right: 1px solid #ccc;
                box-sizing: border-box;
            }}
            #map {{
                flex-grow: 1;
            }}
            .profile-row {{
                padding: 10px;
                margin-bottom: 8px;
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                cursor: pointer;
            }}
            .profile-row:hover {{
                background: #eef;
            }}
        </style>
        <script src="filtering.js"></script>
    </head>
    <body>
        <div id="container">
            <div id="sidebar">
                <h3>Filters</h3>
                <div class="filter-group">
                    <input type="text" id="searchId" placeholder="Search Profile ID">
                    <input type="number" step="0.01" id="dispMin" placeholder="Min Disp">
                    <input type="number" step="0.01" id="dispMax" placeholder="Max Disp">
                    <input type="number" step="0.01" id="widthMin" placeholder="Min Width">
                    <input type="number" step="0.01" id="widthMax" placeholder="Max Width">
                    <input type="number" step="0.0001" id="latMin" placeholder="Min Lat">
                    <input type="number" step="0.0001" id="latMax" placeholder="Max Lat">
                    <input type="number" step="0.0001" id="lonMin" placeholder="Min Lon">
                    <input type="number" step="0.0001" id="lonMax" placeholder="Max Lon">
                    <button onclick="applyFilters()">Apply Filters</button>
                    <button onclick="exportFilteredToCSV()">Export to CSV</button>
                    <button onclick="selectAllMarkers(true)">Select All</button>
                    <button onclick="selectAllMarkers(false)">Deselect All</button>
                </div>
                <hr>
                <div id="profileList"></div>
            </div>
            <div id="map"></div>
        </div>

        <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
        <script>
            let map = L.map('map').setView([{center_lat}, {center_lon}], 13);
            let allMarkers = [];
            allProfiles = {profile_json};

            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);

            function clearMarkers() {{
                allMarkers.forEach(m => map.removeLayer(m));
                allMarkers = [];
                markerMap = {{}};
            }}

            function applyFilters() {{
                clearMarkers();
                const filtered = getFilteredProfiles();
                filtered.forEach(p => {{
                    const marker = L.marker([p.lat, p.lon]).addTo(map);
                    markerMap[p.profile_id] = marker;
                    allMarkers.push(marker);
                }});
                renderSidebarList(filtered, focusOnMarker, toggleMarkerVisibilityLeaflet);
            }}

            function focusOnMarker(profileId) {{
                const marker = markerMap[profileId];
                if (marker) {{
                    map.setView(marker.getLatLng(), 15);
                }}
            }}

            applyFilters();
        </script>
    </body>
    </html>
    """
    #return html

    with open("leaflet_map.html", "w", encoding="utf-8") as f:
        f.write(html)

    return "leaflet_map.html"

    # temp_path = Path(tempfile.gettempdir()) / "leaflet_map.html"
    # temp_path.write_text(html, encoding="utf-8")

    # return temp_path


def generate_google_maps_html(coords_df, api_key):
    if coords_df is None or coords_df.empty:
        return "<html><body><p>No coordinates provided.</p></body></html>"

    first_row = coords_df.iloc[0]
    center_lat = first_row['latitude']
    center_lon = first_row['longitude']

    profile_data = []
    for _, row in coords_df.iterrows():
        profile_data.append({
            "lat": row['latitude'],
            "lon": row['longitude'],
            "profile_id": str(row['Profile ID']),
            "disp": round(row['rescaled disp'], 3),
            "width": round(row['actual width'], 3),
        })
    profile_json = str(profile_data).replace("'", '"')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Google Maps Viewer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            html, body, #container {{
                height: 100%;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            #container {{
                display: flex;
                height: 100%;
            }}
            #sidebar {{
                width: 320px;
                padding: 10px;
                overflow-y: auto;
                background: #f8f8f8;
                border-right: 1px solid #ccc;
                box-sizing: border-box;
            }}
            #map {{
                flex-grow: 1;
            }}
            .profile-row {{
                padding: 10px;
                margin-bottom: 8px;
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                cursor: pointer;
            }}
            .profile-row:hover {{
                background: #eef;
            }}
        </style>
        <script src="filtering.js"></script>
    </head>
    <body>
        <div id="container">
            <div id="sidebar">
                <h3>Filters</h3>
                <div class="filter-group">
                    <input type="text" id="searchId" placeholder="Search Profile ID">
                    <input type="number" step="0.01" id="dispMin" placeholder="Min Disp">
                    <input type="number" step="0.01" id="dispMax" placeholder="Max Disp">
                    <input type="number" step="0.01" id="widthMin" placeholder="Min Width">
                    <input type="number" step="0.01" id="widthMax" placeholder="Max Width">
                    <input type="number" step="0.0001" id="latMin" placeholder="Min Lat">
                    <input type="number" step="0.0001" id="latMax" placeholder="Max Lat">
                    <input type="number" step="0.0001" id="lonMin" placeholder="Min Lon">
                    <input type="number" step="0.0001" id="lonMax" placeholder="Max Lon">
                    <button onclick="applyFilters()">Apply Filters</button>
                    <button onclick="exportFilteredToCSV()">Export to CSV</button>
                    <button onclick="selectAllMarkers(true)">Select All</button>
                    <button onclick="selectAllMarkers(false)">Deselect All</button>
                </div>
                <hr>
                <div id="profileList"></div>
            </div>
            <div id="map"></div>
        </div>

        <script>
            let map;
            let allMarkers = [];
            allProfiles = {profile_json};

            function initMap() {{
                map = new google.maps.Map(document.getElementById("map"), {{
                    center: {{ lat: {center_lat}, lng: {center_lon} }},
                    zoom: 13
                }});
                applyFilters();
            }}

            function clearMarkers() {{
                allMarkers.forEach(marker => marker.setMap(null));
                allMarkers = [];
                markerMap = {{}};
            }}

            function applyFilters() {{
                clearMarkers();
                const filtered = getFilteredProfiles();

                filtered.forEach(p => {{
                    const marker = new google.maps.Marker({{
                        position: {{ lat: p.lat, lng: p.lon }},
                        map: map,
                        title: p.profile_id
                    }});
                    markerMap[p.profile_id] = marker;
                    allMarkers.push(marker);
                }});

                renderSidebarList(filtered, focusOnMarker, toggleMarkerVisibilityGoogle);
            }}

            function focusOnMarker(profileId) {{
                const marker = markerMap[profileId];
                if (marker) {{
                    map.panTo(marker.getPosition());
                    map.setZoom(15);
                }}
            }}
        </script>

        <script async defer
            src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap">
        </script>
    </body>
    </html>
    """
    #return html

    with open("google_map.html", "w", encoding="utf-8") as f:
        f.write(html)

    return "google_map.html"

    # temp_path = Path(tempfile.gettempdir()) / "google_map.html"
    # temp_path.write_text(html, encoding="utf-8")

    # return temp_path
