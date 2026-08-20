import json
import math
from html import escape

import pandas as pd


REQUIRED_MAP_COLUMNS = {
    "LAT",
    "LONG",
    "ID",
    "file_idx",
    "profile_idx",
    "WIDTH"
}


def _safe_float(value):
    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def get_available_displacement_columns(coords_df):
    """Return DISP columns that contain at least one numeric value."""
    if not isinstance(coords_df, pd.DataFrame):
        return []

    columns = []
    for column in coords_df.columns:
        name = str(column)
        if name.startswith("DISP") and name[4:].isdigit() and pd.to_numeric(coords_df[column], errors="coerce").notna().any():
            columns.append(column)

    return sorted(columns, key=lambda column: int(str(column)[4:]))


def _dimension_label(displacement_column):
    suffix = str(displacement_column)[4:]
    return f"Dimension {suffix}" if suffix.isdigit() else str(displacement_column)


def _prepare_profile_data(coords_df, displacement_column):
    if not isinstance(coords_df, pd.DataFrame):
        raise TypeError(
            "coords_df must be a pandas DataFrame."
        )

    required_columns = REQUIRED_MAP_COLUMNS | {displacement_column}
    missing = required_columns - set(coords_df.columns)

    if missing:
        raise ValueError(
            "Map table is missing columns: "
            + ", ".join(sorted(missing))
        )

    profiles = []

    for _, row in coords_df.iterrows():
        latitude = _safe_float(row["LAT"])
        longitude = _safe_float(row["LONG"])

        if latitude is None or longitude is None:
            continue

        profiles.append({
            "profile_id": str(row["ID"]),
            "lat": latitude,
            "lon": longitude,
            "width": _safe_float(row.get("WIDTH")),
            "disp": _safe_float(row.get(displacement_column)),
            "file_idx": int(row["file_idx"]),
            "profile_idx": int(row["profile_idx"]),
        })

    return profiles


def _empty_html(message):
    return f"""
    <!DOCTYPE html>
    <html>
    <body style="
        height: 100vh;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: Arial, sans-serif;
    ">
        <p>{escape(message)}</p>
    </body>
    </html>
    """


def _build_shared_content(profiles, displacement_label):
    profile_json = json.dumps(
        profiles,
        allow_nan=False,
    )

    safe_displacement_label = escape(displacement_label)
    sidebar_html = f"""
    <aside id="sidebar">
        <h3>Profile selection</h3>

        <label class="search-label">
            Profile ID
            <input
                id="searchId"
                type="text"
                placeholder="Search profile ID"
            >
        </label>

        <div class="range-section">
            <div class="range-title">
                Width range
            </div>

            <div class="range-row">
                <input
                    id="widthMin"
                    type="number"
                    step="any"
                    placeholder="Minimum"
                >

                <input
                    id="widthMax"
                    type="number"
                    step="any"
                    placeholder="Maximum"
                >
            </div>
        </div>

        <div class="range-section">
            <div class="range-title">
                {safe_displacement_label} displacement range
            </div>

            <div class="range-row">
                <input
                    id="dispMin"
                    type="number"
                    step="any"
                    placeholder="Minimum"
                >

                <input
                    id="dispMax"
                    type="number"
                    step="any"
                    placeholder="Maximum"
                >
            </div>
        </div>

        <div class="button-row">
            <button onclick="applyFilters()">
                Apply filters
            </button>

            <button onclick="resetFilters()">
                Reset filters
            </button>
        </div>

        <div class="button-row">
            <button onclick="selectAllFiltered()">
                Select all
            </button>

            <button onclick="deselectAllFiltered()">
                Deselect all
            </button>
        </div>

        <div id="filterSummary" class="summary">
        </div>

        <hr>

        <div id="profileList"></div>
    </aside>
    """

    hover_html = f"""
    <div id="hoverInfo" class="hover-info">
        Hover over a marker or profile row.
    </div>

    <div class="map-legends">
        <div class="legend-block">
            <div class="legend-title">Width (marker color)</div>
            <div class="width-gradient"></div>
            <div class="legend-scale"><span id="widthLegendMin"></span><span id="widthLegendMax"></span></div>
        </div>

        <div class="legend-block">
            <div class="legend-title">{safe_displacement_label} displacement (marker size)</div>
            <div id="displacementLegend" class="size-legend"></div>
        </div>
    </div>
    """

    css = """
    html, body, #container {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }

    #container {
        display: flex;
        position: relative;
    }

    #sidebar {
        width: 350px;
        min-width: 300px;
        max-width: 430px;
        box-sizing: border-box;
        padding: 12px;
        overflow-y: auto;
        background: #f7f7f7;
        border-right: 1px solid #c5c5c5;
    }

    #mapArea {
        position: relative;
        flex: 1;
        min-width: 0;
    }

    #map {
        position: absolute;
        inset: 0;
        z-index: 1;
        width: 100%;
        height: 100%;
    }

    .search-label {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 13px;
    }

    .search-label input {
        padding: 7px;
        box-sizing: border-box;
    }

    .range-section {
        margin-top: 12px;
    }

    .range-title {
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 13px;
    }

    .range-row {
        display: flex;
        gap: 6px;
    }

    .range-row input {
        width: 50%;
        min-width: 0;
        padding: 7px;
        box-sizing: border-box;
    }

    .button-row {
        display: flex;
        gap: 7px;
        margin-top: 10px;
    }

    .button-row button {
        flex: 1;
        padding: 7px;
        cursor: pointer;
    }

    .summary {
        margin-top: 10px;
        font-size: 13px;
        color: #555;
    }

    .profile-row {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 9px;
        margin-bottom: 8px;
        background: white;
        border: 1px solid #d0d0d0;
        border-radius: 5px;
        cursor: pointer;
    }

    .profile-row:hover {
        background: #eaf2ff;
        border-color: #7fa5d8;
    }

    .profile-checkbox {
        margin-top: 3px;
        cursor: pointer;
    }

    .profile-content {
        flex: 1;
        min-width: 0;
    }

    .profile-title {
        margin-bottom: 4px;
        font-weight: bold;
        word-break: break-word;
    }

    .profile-values {
        color: #444;
        line-height: 1.45;
        font-size: 12px;
    }

    .marker-key {
        display: inline-block;
        width: 11px;
        height: 11px;
        margin-right: 5px;
        border: 1px solid #333;
        border-radius: 50%;
        vertical-align: middle;
    }

    .hover-info {
        position: absolute;
        z-index: 2000;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        min-width: 390px;
        max-width: calc(100% - 40px);
        padding: 10px 14px;
        box-sizing: border-box;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #999;
        border-radius: 6px;
        box-shadow: 0 2px 9px rgba(0, 0, 0, 0.25);
        pointer-events: none;
        text-align: center;
        font-size: 13px;
    }

    .map-legends {
        position: absolute;
        z-index: 2000;
        top: 12px;
        right: 12px;
        width: 225px;
        padding: 10px 12px;
        box-sizing: border-box;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #999;
        border-radius: 6px;
        box-shadow: 0 2px 9px rgba(0, 0, 0, 0.25);
        pointer-events: none;
        font-size: 12px;
    }

    .legend-block + .legend-block {
        margin-top: 13px;
        padding-top: 10px;
        border-top: 1px solid #ccc;
    }

    .legend-title {
        margin-bottom: 7px;
        font-weight: bold;
    }

    .width-gradient {
        height: 12px;
        border: 1px solid #555;
        background: linear-gradient(to right, hsl(240, 85%, 47%), hsl(180, 85%, 47%), hsl(120, 85%, 47%), hsl(60, 85%, 47%), hsl(0, 85%, 47%));
    }

    .legend-scale {
        display: flex;
        justify-content: space-between;
        margin-top: 3px;
        color: #444;
    }

    .size-legend {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        min-height: 62px;
    }

    .size-sample {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        min-width: 48px;
        color: #444;
    }

    .size-circle {
        box-sizing: border-box;
        margin-bottom: 4px;
        border: 1px solid #222;
        border-radius: 50%;
        background: #888;
    }

    .cluster-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid rgba(0, 0, 0, 0.55);
        border-radius: 50%;
        color: white;
        background: rgba(60, 90, 150, 0.85);
        font-weight: bold;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.4);
    }
    """

    js = f"""
    const allProfiles = {profile_json};
    const displacementLabel = {json.dumps(displacement_label)};

    /*
     * Selection is independent of filtering.
     *
     * selected=true:
     *     eligible to appear on map
     *
     * passes filters:
     *     appears in current sidebar subset
     *
     * Marker is shown only when both are true.
     */
    allProfiles.forEach(profile => {{
        profile.selected = true;
    }});

    let filteredProfiles = [...allProfiles];
    let displayedProfiles = [...allProfiles];
    let markerMap = {{}};
    let qtBridge = null;

    const numericWidths = allProfiles
        .map(profile => profile.width)
        .filter(Number.isFinite);

    const numericDisplacements = allProfiles
        .map(profile => Math.abs(profile.disp))
        .filter(Number.isFinite);

    const widthDataMin = numericWidths.length
        ? Math.min(...numericWidths)
        : 0;

    const widthDataMax = numericWidths.length
        ? Math.max(...numericWidths)
        : 1;

    const dispDataMin = numericDisplacements.length
        ? Math.min(...numericDisplacements)
        : 0;

    const dispDataMax = numericDisplacements.length
        ? Math.max(...numericDisplacements)
        : 1;

    new QWebChannel(
        qt.webChannelTransport,
        function(channel) {{
            qtBridge = channel.objects.bridge;
        }}
    );

    function clamp(value, minimum, maximum) {{
        return Math.max(
            minimum,
            Math.min(maximum, value)
        );
    }}

    function normalize(value, minimum, maximum) {{
        if (!Number.isFinite(value)) {{
            return 0.5;
        }}

        if (maximum <= minimum) {{
            return 0.5;
        }}

        return clamp(
            (value - minimum) /
            (maximum - minimum),
            0,
            1
        );
    }}

    function widthColor(width) {{
        const t = normalize(
            width,
            widthDataMin,
            widthDataMax
        );

        /*
         * Small width: blue
         * Medium width: green/yellow
         * Large width: red
         */
        const hue = 240 * (1 - t);

        return `hsl(${{hue}}, 85%, 47%)`;
    }}

    function displacementRadius(displacement) {{
        const magnitude = Number.isFinite(displacement)
            ? Math.abs(displacement)
            : dispDataMin;

        const t = normalize(
            magnitude,
            dispDataMin,
            dispDataMax
        );

        /*
         * Radius between 5 and 20 pixels.
         * Square root scaling avoids giant markers
         * when displacement is strongly skewed.
         */
        return 5 + 15 * Math.sqrt(t);
    }}

    function formatNumber(value, digits=3) {{
        if (!Number.isFinite(value)) {{
            return "N/A";
        }}

        return value.toFixed(digits);
    }}

    function escapeHtml(value) {{
        const element =
            document.createElement("div");

        element.textContent = String(value);

        return element.innerHTML;
    }}

    function initializeLegends() {{
        document.getElementById("widthLegendMin").textContent = formatNumber(widthDataMin);
        document.getElementById("widthLegendMax").textContent = formatNumber(widthDataMax);

        const values = dispDataMax > dispDataMin
            ? [dispDataMin, (dispDataMin + dispDataMax) / 2, dispDataMax]
            : [dispDataMin];

        document.getElementById("displacementLegend").innerHTML = values.map(value => {{
            const diameter = 2 * displacementRadius(value);
            return `<div class="size-sample"><div class="size-circle" style="width:${{diameter}}px;height:${{diameter}}px"></div><span>${{formatNumber(value)}}</span></div>`;
        }}).join("");
    }}

    function profileInfoHtml(profile) {{
        return `
            <strong>
                ${{escapeHtml(profile.profile_id)}}
            </strong>

            &nbsp;|&nbsp;

            Coordinates:
            ${{formatNumber(profile.lat, 5)}},
            ${{formatNumber(profile.lon, 5)}}

            &nbsp;|&nbsp;

            Width:
            ${{formatNumber(profile.width)}}

            &nbsp;|&nbsp;

            ${{escapeHtml(displacementLabel)}} displacement:
            ${{formatNumber(profile.disp)}}
        `;
    }}

    function showHoverInfo(profile) {{
        document
            .getElementById("hoverInfo")
            .innerHTML = profileInfoHtml(profile);
    }}

    function clearHoverInfo() {{
        document
            .getElementById("hoverInfo")
            .textContent =
            "Hover over a marker or profile row.";
    }}

    function optionalNumber(elementId) {{
        const text = document
            .getElementById(elementId)
            .value
            .trim();

        if (text === "") {{
            return null;
        }}

        const value = Number(text);

        return Number.isFinite(value)
            ? value
            : null;
    }}

    function valueInRange(
        value,
        minimum,
        maximum
    ) {{
        if (!Number.isFinite(value)) {{
            /*
             * Profiles without a numeric value remain
             * visible only if no range is requested.
             */
            return (
                minimum === null &&
                maximum === null
            );
        }}

        if (
            minimum !== null &&
            value < minimum
        ) {{
            return false;
        }}

        if (
            maximum !== null &&
            value > maximum
        ) {{
            return false;
        }}

        return true;
    }}

    function getFilteredProfiles() {{
        const search = document
            .getElementById("searchId")
            .value
            .trim()
            .toLowerCase();

        const widthMin =
            optionalNumber("widthMin");

        const widthMax =
            optionalNumber("widthMax");

        const dispMin =
            optionalNumber("dispMin");

        const dispMax =
            optionalNumber("dispMax");

        return allProfiles.filter(profile => {{
            const nameMatches = (
                !search ||
                profile.profile_id
                    .toLowerCase()
                    .includes(search)
            );

            return (
                nameMatches &&
                valueInRange(
                    profile.width,
                    widthMin,
                    widthMax
                ) &&
                valueInRange(
                    profile.disp,
                    dispMin,
                    dispMax
                )
            );
        }});
    }}

    function getDisplayedProfiles() {{
        return filteredProfiles.filter(
            profile => profile.selected
        );
    }}

    function updateSummary() {{
        const selectedFiltered =
            filteredProfiles.filter(
                profile => profile.selected
            ).length;

        const selectedTotal =
            allProfiles.filter(
                profile => profile.selected
            ).length;

        document
            .getElementById("filterSummary")
            .textContent =
            `${{filteredProfiles.length}} match filters; ` +
            `${{selectedFiltered}} displayed; ` +
            `${{selectedTotal}} selected total`;
    }}

    function renderSidebar() {{
        const list =
            document.getElementById("profileList");

        list.innerHTML = "";

        filteredProfiles.forEach(profile => {{
            const row =
                document.createElement("div");

            row.className = "profile-row";

            const checkbox =
                document.createElement("input");

            checkbox.type = "checkbox";
            checkbox.className =
                "profile-checkbox";

            checkbox.checked =
                profile.selected;

            checkbox.addEventListener(
                "click",
                event => {{
                    /*
                     * Do not allow checkbox click to also
                     * trigger row zoom behavior.
                     */
                    event.stopPropagation();
                }}
            );

            checkbox.addEventListener(
                "change",
                event => {{
                    profile.selected =
                        event.target.checked;

                    updateMarkersFromCurrentState();
                    updateSummary();
                }}
            );

            const content =
                document.createElement("div");

            content.className =
                "profile-content";

            const color =
                widthColor(profile.width);

            content.innerHTML = `
                <div class="profile-title">
                    <span
                        class="marker-key"
                        style="background:${{color}}">
                    </span>

                    ${{escapeHtml(
                        profile.profile_id
                    )}}
                </div>

                <div class="profile-values">
                    Coordinates:
                    ${{formatNumber(
                        profile.lat,
                        5
                    )}},
                    ${{formatNumber(
                        profile.lon,
                        5
                    )}}
                    <br>

                    Width:
                    ${{formatNumber(
                        profile.width
                    )}}
                    <br>

                    ${{escapeHtml(displacementLabel)}} displacement:
                    ${{formatNumber(
                        profile.disp
                    )}}
                </div>
            `;

            row.appendChild(checkbox);
            row.appendChild(content);

            row.addEventListener(
                "mouseenter",
                () => showHoverInfo(profile)
            );

            row.addEventListener(
                "mouseleave",
                clearHoverInfo
            );

            row.addEventListener(
                "click",
                () => focusOnProfile(profile)
            );

            row.addEventListener(
                "dblclick",
                () => openProfile(profile)
            );

            list.appendChild(row);
        }});

        updateSummary();
    }}

    function applyFilters() {{
        filteredProfiles =
            getFilteredProfiles();

        renderSidebar();
        updateMarkersFromCurrentState();
        fitMapToDisplayedProfiles();
    }}

    function resetFilters() {{
        [
            "searchId",
            "widthMin",
            "widthMax",
            "dispMin",
            "dispMax"
        ].forEach(id => {{
            document
                .getElementById(id)
                .value = "";
        }});

        applyFilters();
    }}

    function selectAllFiltered() {{
        filteredProfiles.forEach(profile => {{
            profile.selected = true;
        }});

        renderSidebar();
        updateMarkersFromCurrentState();
        fitMapToDisplayedProfiles();
    }}

    function deselectAllFiltered() {{
        filteredProfiles.forEach(profile => {{
            profile.selected = false;
        }});

        renderSidebar();
        updateMarkersFromCurrentState();
    }}

    function openProfile(profile) {{
        if (!qtBridge) {{
            console.warn(
                "Qt bridge is not ready."
            );
            return;
        }}

        qtBridge.show_profile(
            profile.file_idx,
            profile.profile_idx
        );
    }}

    [
        "searchId",
        "widthMin",
        "widthMax",
        "dispMin",
        "dispMax"
    ].forEach(id => {{
        document
            .getElementById(id)
            .addEventListener(
                "keydown",
                event => {{
                    if (event.key === "Enter") {{
                        applyFilters();
                    }}
                }}
            );
    }});

    initializeLegends();
    """

    return (
        sidebar_html,
        hover_html,
        css,
        js,
    )


def generate_leaflet_html(coords_df, displacement_column="DISP1"):
    if coords_df is None or coords_df.empty:
        return _empty_html(
            "No coordinates provided."
        )

    profiles = _prepare_profile_data(coords_df, displacement_column)

    if not profiles:
        return _empty_html(
            "No valid coordinate rows were found."
        )

    center_lat = profiles[0]["lat"]
    center_lon = profiles[0]["lon"]

    displacement_label = _dimension_label(displacement_column)
    sidebar_html, hover_html, css, shared_js = _build_shared_content(profiles, displacement_label)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">

        <meta
            name="viewport"
            content="width=device-width,
                     initial-scale=1.0"
        >

        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        >

        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"
        >

        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"
        >

        <script
            src="qrc:///qtwebchannel/qwebchannel.js">
        </script>

        <style>
            {css}
        </style>
    </head>

    <body>
        <div id="container">
            {sidebar_html}

            <main id="mapArea">
                <div id="map"></div>
                {hover_html}
            </main>
        </div>

        <script
            src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
        </script>

        <script
            src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js">
        </script>

        <script>
            {shared_js}

            const map = L.map("map").setView(
                [{center_lat}, {center_lon}],
                5
            );

            L.tileLayer(
                "https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
                {{
                    attribution:
                        "© OpenStreetMap contributors"
                }}
            ).addTo(map);

            /*
             * Marker clustering prevents thousands of
             * individual markers from rendering initially.
             */
            const markerCluster =
                L.markerClusterGroup({{
                    chunkedLoading: true,
                    chunkInterval: 100,
                    chunkDelay: 20,
                    removeOutsideVisibleBounds: true,
                    spiderfyOnMaxZoom: true,
                    showCoverageOnHover: false,
                    zoomToBoundsOnClick: true,
                    disableClusteringAtZoom: 17,
                    maxClusterRadius: 55
                }});

            map.addLayer(markerCluster);

            function createMarker(profile) {{
                const marker = L.circleMarker(
                    [profile.lat, profile.lon],
                    {{
                        radius:
                            displacementRadius(
                                profile.disp
                            ),
                        color: "#222",
                        weight: 1,
                        fillColor:
                            widthColor(
                                profile.width
                            ),
                        fillOpacity: 0.85
                    }}
                );

                marker.profileData = profile;

                marker.on(
                    "mouseover",
                    () => showHoverInfo(profile)
                );

                marker.on(
                    "mouseout",
                    clearHoverInfo
                );

                marker.on(
                    "dblclick",
                    event => {{
                        L.DomEvent.stopPropagation(
                            event
                        );

                        openProfile(profile);
                    }}
                );

                return marker;
            }}

            function clearMarkers() {{
                markerCluster.clearLayers();
                markerMap = {{}};
            }}

            function updateMarkersFromCurrentState() {{
                clearMarkers();

                displayedProfiles =
                    getDisplayedProfiles();

                const markers = [];

                displayedProfiles.forEach(
                    profile => {{
                        const marker =
                            createMarker(profile);

                        markerMap[
                            profile.profile_id
                        ] = marker;

                        markers.push(marker);
                    }}
                );

                /*
                 * addLayers is much faster than adding
                 * thousands of markers one at a time.
                 */
                markerCluster.addLayers(markers);

                updateSummary();
            }}

            function fitMapToDisplayedProfiles() {{
                if (
                    displayedProfiles.length === 0
                ) {{
                    return;
                }}

                if (
                    displayedProfiles.length === 1
                ) {{
                    map.setView(
                        [
                            displayedProfiles[0].lat,
                            displayedProfiles[0].lon
                        ],
                        15
                    );

                    return;
                }}

                const bounds =
                    L.latLngBounds(
                        displayedProfiles.map(
                            profile => [
                                profile.lat,
                                profile.lon
                            ]
                        )
                    );

                map.fitBounds(
                    bounds,
                    {{
                        padding: [30, 30],
                        maxZoom: 14
                    }}
                );
            }}

            function focusOnProfile(profile) {{
                const marker =
                    markerMap[profile.profile_id];

                if (!marker) {{
                    return;
                }}

                /*
                 * zoomToShowLayer expands clusters until
                 * the individual marker becomes visible.
                 */
                markerCluster.zoomToShowLayer(
                    marker,
                    () => {{
                        map.panTo(
                            marker.getLatLng()
                        );

                        showHoverInfo(profile);
                    }}
                );
            }}

            filteredProfiles =
                getFilteredProfiles();

            renderSidebar();
            updateMarkersFromCurrentState();
            fitMapToDisplayedProfiles();
        </script>
    </body>
    </html>
    """


def generate_google_maps_html(coords_df, api_key, displacement_column="DISP1"):
    if coords_df is None or coords_df.empty:
        return _empty_html(
            "No coordinates provided."
        )

    if not api_key:
        return _empty_html(
            "No Google Maps API key was provided."
        )

    profiles = _prepare_profile_data(coords_df, displacement_column)

    if not profiles:
        return _empty_html(
            "No valid coordinate rows were found."
        )

    center_lat = profiles[0]["lat"]
    center_lon = profiles[0]["lon"]

    displacement_label = _dimension_label(displacement_column)
    sidebar_html, hover_html, css, shared_js = _build_shared_content(profiles, displacement_label)

    escaped_key = escape(api_key)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">

        <meta
            name="viewport"
            content="width=device-width,
                     initial-scale=1.0"
        >

        <script
            src="qrc:///qtwebchannel/qwebchannel.js">
        </script>

        <style>
            {css}
        </style>
    </head>

    <body>
        <div id="container">
            {sidebar_html}

            <main id="mapArea">
                <div id="map"></div>
                {hover_html}
            </main>
        </div>

        <script>
            {shared_js}

            let map;
            let markerCluster = null;

            function createMarker(profile) {{
                const marker =
                    new google.maps.Marker({{
                        position: {{
                            lat: profile.lat,
                            lng: profile.lon
                        }},
                        title:
                            profile.profile_id,
                        icon: {{
                            path:
                                google.maps
                                    .SymbolPath
                                    .CIRCLE,
                            scale:
                                displacementRadius(
                                    profile.disp
                                ),
                            fillColor:
                                widthColor(
                                    profile.width
                                ),
                            fillOpacity: 0.85,
                            strokeColor: "#222",
                            strokeOpacity: 1,
                            strokeWeight: 1
                        }}
                    }});

                marker.profileData = profile;

                marker.addListener(
                    "mouseover",
                    () => showHoverInfo(profile)
                );

                marker.addListener(
                    "mouseout",
                    clearHoverInfo
                );

                marker.addListener(
                    "dblclick",
                    () => openProfile(profile)
                );

                return marker;
            }}

            function clearMarkers() {{
                if (markerCluster) {{
                    markerCluster.clearMarkers();
                }}

                Object.values(
                    markerMap
                ).forEach(marker => {{
                    marker.setMap(null);
                }});

                markerMap = {{}};
            }}

            function updateMarkersFromCurrentState() {{
                clearMarkers();

                displayedProfiles =
                    getDisplayedProfiles();

                const markers =
                    displayedProfiles.map(
                        profile => {{
                            const marker =
                                createMarker(profile);

                            markerMap[
                                profile.profile_id
                            ] = marker;

                            return marker;
                        }}
                    );

                markerCluster =
                    new markerClusterer
                        .MarkerClusterer({{
                            map: map,
                            markers: markers
                        }});

                updateSummary();
            }}

            function fitMapToDisplayedProfiles() {{
                if (
                    displayedProfiles.length === 0
                ) {{
                    return;
                }}

                if (
                    displayedProfiles.length === 1
                ) {{
                    map.setCenter({{
                        lat:
                            displayedProfiles[0]
                                .lat,
                        lng:
                            displayedProfiles[0]
                                .lon
                    }});

                    map.setZoom(15);
                    return;
                }}

                const bounds =
                    new google.maps.LatLngBounds();

                displayedProfiles.forEach(
                    profile => {{
                        bounds.extend({{
                            lat: profile.lat,
                            lng: profile.lon
                        }});
                    }}
                );

                map.fitBounds(bounds);
            }}

            function focusOnProfile(profile) {{
                map.panTo({{
                    lat: profile.lat,
                    lng: profile.lon
                }});

                map.setZoom(16);
                showHoverInfo(profile);
            }}

            function initMap() {{
                map = new google.maps.Map(
                    document.getElementById(
                        "map"
                    ),
                    {{
                        center: {{
                            lat: {center_lat},
                            lng: {center_lon}
                        }},
                        zoom: 5
                    }}
                );

                filteredProfiles =
                    getFilteredProfiles();

                renderSidebar();
                updateMarkersFromCurrentState();
                fitMapToDisplayedProfiles();
            }}
        </script>

        <script
            src="https://unpkg.com/@googlemaps/markerclusterer/dist/index.min.js">
        </script>

        <script
            async
            defer
            src="https://maps.googleapis.com/maps/api/js?key={escaped_key}&callback=initMap">
        </script>
    </body>
    </html>
    """
