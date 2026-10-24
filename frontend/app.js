/**
 * GeoAI Assistant Standalone Web GUI
 * Vanilla JavaScript interacting asynchronously with FastAPI /recommend
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const recommendForm = document.getElementById("recommendForm");
    const queryInput = document.getElementById("queryInput");
    const radiusSelect = document.getElementById("radiusSelect");
    const useSlmToggle = document.getElementById("useSlmToggle");
    const submitBtn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const resetBtn = document.getElementById("resetBtn");
    
    // Health Checkboxes
    const healthAsthma = document.getElementById("healthAsthma");
    const healthSenior = document.getElementById("healthSenior");
    const healthChild = document.getElementById("healthChild");
    const healthPregnant = document.getElementById("healthPregnant");

    // Dynamic Areas
    const loadingIndicator = document.getElementById("loadingIndicator");
    const slmNotice = document.getElementById("slmNotice");
    const errorCard = document.getElementById("errorCard");
    const errorMessage = document.getElementById("errorMessage");
    const clarificationCard = document.getElementById("clarificationCard");
    const clarificationQuestion = document.getElementById("clarificationQuestion");
    const resultsSection = document.getElementById("resultsSection");
    const systemStatusText = document.getElementById("systemStatusText");
    const statusDot = document.querySelector(".status-dot");

    // Leaflet Map instance
    let map = null;
    let mapMarkersLayer = null;
    let radiusCircleLayer = null;

    // Check System Health on Page Load
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                systemStatusText.textContent = `Online · ${data.supported_activities_count} Activities · Live OSM & Open-Meteo`;
                statusDot.classList.remove("offline");
            } else {
                systemStatusText.textContent = "Service Unreachable";
                statusDot.classList.add("offline");
            }
        } catch (e) {
            systemStatusText.textContent = "Backend Offline";
            statusDot.classList.add("offline");
        }
    }
    checkHealth();

    // Quick suggestion chips
    document.querySelectorAll(".btn-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                queryInput.focus();
            }
        });
    });

    // Reset Form
    resetBtn.addEventListener("click", () => {
        recommendForm.reset();
        hideAllCards();
        queryInput.focus();
    });

    // Toggle SLM notice when checkbox changes
    useSlmToggle.addEventListener("change", () => {
        if (useSlmToggle.checked) {
            slmNotice.classList.remove("hidden");
        } else {
            slmNotice.classList.add("hidden");
        }
    });

    function hideAllCards() {
        errorCard.classList.add("hidden");
        clarificationCard.classList.add("hidden");
        resultsSection.classList.add("hidden");
    }

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.textContent = "Analyzing Geospatial Data...";
            loadingIndicator.classList.remove("hidden");
            if (useSlmToggle.checked) {
                slmNotice.classList.remove("hidden");
            } else {
                slmNotice.classList.add("hidden");
            }
            hideAllCards();
        } else {
            submitBtn.disabled = false;
            btnText.textContent = "Analyze & Recommend";
            loadingIndicator.classList.add("hidden");
        }
    }

    // Initialize or Reset Leaflet Map
    function initMap(lat, lon, zoom = 13) {
        const mapContainer = document.getElementById("map");
        if (!mapContainer) return;

        if (!map) {
            map = L.map("map").setView([lat, lon], zoom);
            L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
            }).addTo(map);
            mapMarkersLayer = L.layerGroup().addTo(map);
        } else {
            map.setView([lat, lon], zoom);
            if (mapMarkersLayer) {
                mapMarkersLayer.clearLayers();
            }
            if (radiusCircleLayer) {
                map.removeLayer(radiusCircleLayer);
                radiusCircleLayer = null;
            }
        }

        // Invalidate map size after rendering container
        setTimeout(() => {
            if (map) map.invalidateSize();
        }, 150);
    }

    // Submit Handler
    recommendForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const rawQuery = queryInput.value.trim();
        if (!rawQuery) {
            showError("Please enter an outdoor activity question.");
            return;
        }

        // Safe search radius: default 5 km, never send 0 or NaN
        let radius = parseFloat(radiusSelect.value);
        if (isNaN(radius) || radius <= 0) {
            radius = 5.0;
        }

        const useSlm = useSlmToggle.checked;

        // Construct Pydantic-compliant request payload
        const payload = {
            query: rawQuery,
            search_radius_km: radius,
            user_profile: {
                has_asthma: healthAsthma.checked,
                is_senior: healthSenior.checked,
                is_child: healthChild.checked,
                is_pregnant: healthPregnant.checked
            },
            use_slm: useSlm
        };

        setLoading(true);

        try {
            // Asynchronous fetch with no artificial short timeout (allows CPU inference to finish)
            const response = await fetch("/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            renderResult(data, radius);

        } catch (err) {
            console.error("Recommendation request failed:", err);
            showError(err.message || "Network error. Failed to retrieve recommendation from server.");
        } finally {
            setLoading(false);
        }
    });

    function showError(msg) {
        hideAllCards();
        errorMessage.textContent = msg;
        errorCard.classList.remove("hidden");
    }

    function renderResult(data, requestedRadiusKm) {
        hideAllCards();

        // 1. Handle Needs Clarification
        if (data.status === "needs_clarification") {
            clarificationQuestion.textContent = data.clarification_question || "Essential location or activity missing.";
            clarificationCard.classList.remove("hidden");
            clarificationCard.scrollIntoView({ behavior: "smooth" });
            return;
        }

        // 2. Render Results Section
        resultsSection.classList.remove("hidden");

        const hasSuitable = data.has_suitable_option === true;
        const best = data.best_recommendation;

        // Status Banner Configuration
        const statusBanner = document.getElementById("statusBanner");
        const statusIcon = document.getElementById("statusIcon");
        const statusBadge = document.getElementById("statusBadge");
        const suitableBadge = document.getElementById("suitableBadge");
        const engineBadge = document.getElementById("engineBadge");
        const statusHeadline = document.getElementById("statusHeadline");
        const statusReason = document.getElementById("statusReason");

        statusBanner.className = "status-banner";

        let recStatus = best ? best.recommendation_status : "not_recommended";
        if (!hasSuitable) {
            recStatus = "not_recommended";
        }

        if (recStatus === "recommended") {
            statusBanner.classList.add("status-recommended");
            statusIcon.textContent = "✅";
            statusBadge.textContent = "RECOMMENDED";
            suitableBadge.textContent = "SUITABLE OPTION FOUND";
            statusHeadline.textContent = "Optimal Outdoor Conditions Found";
            statusReason.textContent = "Weather, air quality, and spatial buffer meet ideal activity criteria.";
        } else if (recStatus === "caution") {
            statusBanner.classList.add("status-caution");
            statusIcon.textContent = "⚠️";
            statusBadge.textContent = "PROCEED WITH CAUTION";
            suitableBadge.textContent = "SUITABLE WITH ADVISORIES";
            statusHeadline.textContent = "Marginal Conditions Detected";
            statusReason.textContent = "Environmental conditions are acceptable but warrant caution or timing adjustments.";
        } else {
            statusBanner.classList.add("status-not_recommended");
            statusIcon.textContent = "🚫";
            statusBadge.textContent = "NOT RECOMMENDED";
            suitableBadge.textContent = "NO SUITABLE OPTION FOUND";
            statusHeadline.textContent = "Outdoor Exertion Unadvisable";
            statusReason.textContent = data.no_suitable_option_reason || "All evaluated place-and-time slots exhibit unfavorable air quality or weather.";
        }

        // Engine source badge
        if (data.explanation_source === "slm_qwen2.5_1.5b") {
            engineBadge.textContent = "ENGINE: QWEN 2.5 1.5B (SLM)";
            engineBadge.style.color = "#0284c7";
            engineBadge.style.borderColor = "#7dd3fc";
        } else {
            engineBadge.textContent = "ENGINE: DETERMINISTIC HYBRID";
            engineBadge.style.color = "#475569";
            engineBadge.style.borderColor = "#cbd5e1";
        }

        // 3. Render Best Venue Card
        if (best) {
            const isAnchorFallback = best.venue_id && best.venue_id.startsWith("anchor/");
            const venueNameEl = document.getElementById("venueName");
            const venueTypeEl = document.getElementById("venueType");
            const venueDistanceEl = document.getElementById("venueDistance");
            const venueScoreEl = document.getElementById("venueScore");
            const slotTimeEl = document.getElementById("slotTime");

            if (isAnchorFallback) {
                // Rule requirement: Clearly label anchor reference points, NOT verified venues
                venueNameEl.textContent = `${best.venue_name} (Reference Anchor)`;
                venueTypeEl.textContent = "GEOGRAPHIC REFERENCE POINT";
                venueTypeEl.style.background = "#fef3c7";
                venueTypeEl.style.color = "#92400e";
                venueDistanceEl.textContent = "Fallback search anchor (no specific sports facility/park discovered in radius)";
            } else {
                venueNameEl.textContent = best.venue_name;
                venueTypeEl.textContent = (best.place_type || "Venue").toUpperCase();
                venueTypeEl.style.background = "#f1f5f9";
                venueTypeEl.style.color = "#475569";
                venueDistanceEl.textContent = `${best.distance_km.toFixed(2)} km from search anchor`;
            }

            venueScoreEl.textContent = Math.round(best.suitability_score);
            slotTimeEl.textContent = best.slot_time_ist;

            // Environmental Metrics
            const w = best.weather_summary || {};
            const a = best.aqi_summary || {};

            document.getElementById("valTemp").textContent = w.temperature_c !== undefined ? `${w.temperature_c}°C` : "--";
            document.getElementById("valRainProb").textContent = w.rain_probability_percent !== undefined ? `${w.rain_probability_percent}%` : "--";
            document.getElementById("valWind").textContent = w.wind_speed_kmh !== undefined ? `${w.wind_speed_kmh} km/h` : "--";
            document.getElementById("valHumidity").textContent = w.humidity_percent !== undefined ? `${w.humidity_percent}%` : "--";

            document.getElementById("valPm25").textContent = a.pm2_5 !== undefined ? `${a.pm2_5} µg/m³` : "--";
            document.getElementById("valUsAqi").textContent = a.us_aqi !== undefined ? a.us_aqi : "--";

            // AQI category label
            const aqiVal = a.us_aqi;
            const aqiCatEl = document.getElementById("valAqiCategory");
            if (aqiVal !== undefined) {
                if (aqiVal <= 50) { aqiCatEl.textContent = "Good"; aqiCatEl.style.color = "#16a34a"; }
                else if (aqiVal <= 100) { aqiCatEl.textContent = "Moderate"; aqiCatEl.style.color = "#ca8a04"; }
                else if (aqiVal <= 150) { aqiCatEl.textContent = "Unhealthy for Sensitive"; aqiCatEl.style.color = "#ea580c"; }
                else { aqiCatEl.textContent = "Unhealthy"; aqiCatEl.style.color = "#dc2626"; }
            } else {
                aqiCatEl.textContent = "--";
            }

            // Geographic Context Indicators
            const geo = best.geographic_context || {};
            document.getElementById("geoAdvisory").textContent = geo.summary_advisory || "Standard terrain evaluated.";

            const contextTags = document.getElementById("contextTags");
            contextTags.innerHTML = "";
            if (geo.has_green_canopy) {
                const t = document.createElement("span");
                t.className = "ctx-tag";
                t.textContent = "🌳 Green Canopy Cover";
                contextTags.appendChild(t);
            }
            if (geo.is_near_water) {
                const t = document.createElement("span");
                t.className = "ctx-tag";
                t.textContent = "💧 Near Water Body";
                contextTags.appendChild(t);
            }
            if (geo.is_downwind_highway) {
                const t = document.createElement("span");
                t.className = "ctx-tag";
                t.style.background = "#fee2e2";
                t.style.color = "#991b1b";
                t.textContent = "🚗 Highway Downwind Plume Risk";
                contextTags.appendChild(t);
            }
            if (geo.is_downwind_industry) {
                const t = document.createElement("span");
                t.className = "ctx-tag";
                t.style.background = "#fee2e2";
                t.style.color = "#991b1b";
                t.textContent = "🏭 Industrial Plume Risk";
                contextTags.appendChild(t);
            }

            // Timing Note
            const timingNote = document.getElementById("timingNote");
            if (data.temporal_comparison) {
                timingNote.textContent = data.temporal_comparison;
                document.getElementById("timingNoteBox").classList.remove("hidden");
            } else {
                document.getElementById("timingNoteBox").classList.add("hidden");
            }

            // Initialize Map with best coordinates
            if (best.coordinate && best.coordinate.latitude && best.coordinate.longitude) {
                initMap(best.coordinate.latitude, best.coordinate.longitude, 13);

                // Add search radius circle
                radiusCircleLayer = L.circle([best.coordinate.latitude, best.coordinate.longitude], {
                    color: '#0284c7',
                    fillColor: '#38bdf8',
                    fillOpacity: 0.08,
                    radius: (requestedRadiusKm || 5.0) * 1000
                }).addTo(map);

                // Add Primary Marker
                const markerColor = isAnchorFallback ? "#f59e0b" : "#10b981";
                const markerIcon = L.divIcon({
                    className: "custom-div-icon",
                    html: `<div style="background-color:${markerColor}; width:16px; height:16px; border-radius:50%; border:3px solid #ffffff; box-shadow:0 0 6px rgba(0,0,0,0.4);"></div>`,
                    iconSize: [16, 16],
                    iconAnchor: [8, 8]
                });

                const popupContent = document.createElement("div");
                const pTitle = document.createElement("strong");
                pTitle.textContent = isAnchorFallback ? `${best.venue_name} (Reference Anchor)` : best.venue_name;
                popupContent.appendChild(pTitle);

                const pMeta = document.createElement("div");
                pMeta.style.fontSize = "12px";
                pMeta.style.color = "#475569";
                pMeta.style.marginTop = "4px";
                pMeta.textContent = isAnchorFallback
                    ? "Geographic search reference anchor point (Not a verified sports venue)."
                    : `Score: ${best.suitability_score}/100 (${best.recommendation_status.toUpperCase()}) | Distance: ${best.distance_km.toFixed(2)} km`;
                popupContent.appendChild(pMeta);

                const marker = L.marker([best.coordinate.latitude, best.coordinate.longitude], { icon: markerIcon })
                    .bindPopup(popupContent)
                    .addTo(mapMarkersLayer);

                marker.openPopup();
            }
        }

        // 4. Render Alternatives Table & Map Markers
        const tbody = document.getElementById("alternativesTbody");
        tbody.innerHTML = "";
        const alts = data.top_alternatives || [];

        if (alts.length > 0) {
            document.getElementById("alternativesCard").classList.remove("hidden");
            alts.forEach((alt, idx) => {
                const tr = document.createElement("tr");

                const isAltAnchor = alt.venue_id && alt.venue_id.startsWith("anchor/");
                const displayName = isAltAnchor ? `${alt.venue_name} (Anchor)` : alt.venue_name;

                tr.innerHTML = `
                    <td>${idx + 2}</td>
                    <td><strong>${escapeHtml(displayName)}</strong></td>
                    <td>${escapeHtml(alt.place_type || "venue")}</td>
                    <td>${alt.distance_km ? alt.distance_km.toFixed(2) + " km" : "--"}</td>
                    <td>${escapeHtml(alt.slot_time_ist || "--")}</td>
                    <td>${alt.weather_summary ? `${alt.weather_summary.temperature_c}°C / ${alt.weather_summary.rain_probability_percent}%` : "--"}</td>
                    <td>${alt.aqi_summary && alt.aqi_summary.pm2_5 ? `${alt.aqi_summary.pm2_5} µg/m³` : "--"}</td>
                    <td><strong>${Math.round(alt.suitability_score)}</strong>/100</td>
                    <td><span class="badge ${alt.recommendation_status === 'recommended' ? 'status-green' : (alt.recommendation_status === 'caution' ? 'status-amber' : 'status-red')}">${escapeHtml(alt.recommendation_status.toUpperCase())}</span></td>
                `;
                tbody.appendChild(tr);

                // Add alternative marker to map
                if (map && mapMarkersLayer && alt.coordinate && alt.coordinate.latitude && alt.coordinate.longitude) {
                    const altIcon = L.divIcon({
                        className: "custom-div-icon",
                        html: `<div style="background-color:#0284c7; width:12px; height:12px; border-radius:50%; border:2px solid #ffffff; box-shadow:0 0 4px rgba(0,0,0,0.3);"></div>`,
                        iconSize: [12, 12],
                        iconAnchor: [6, 6]
                    });

                    const altPopup = document.createElement("div");
                    const apTitle = document.createElement("strong");
                    apTitle.textContent = displayName;
                    altPopup.appendChild(apTitle);

                    const apMeta = document.createElement("div");
                    apMeta.style.fontSize = "12px";
                    apMeta.textContent = `Score: ${alt.suitability_score}/100 | Time: ${alt.slot_time_ist}`;
                    altPopup.appendChild(apMeta);

                    L.marker([alt.coordinate.latitude, alt.coordinate.longitude], { icon: altIcon })
                        .bindPopup(altPopup)
                        .addTo(mapMarkersLayer);
                }
            });
        } else {
            document.getElementById("alternativesCard").classList.add("hidden");
        }

        // 5. Render Explanation (Safely using textContent)
        const expEl = document.getElementById("explanationText");
        expEl.textContent = data.explanation || "No explanation text provided.";

        const expBadge = document.getElementById("explanationSourceBadge");
        if (data.explanation_source === "slm_qwen2.5_1.5b") {
            expBadge.textContent = "Generated by Qwen 2.5 1.5B (Local SLM on CPU)";
        } else {
            expBadge.textContent = "Generated by Deterministic Grounded Engine";
        }

        // Diagnostics
        const diagBadge = document.getElementById("diagnosticsBadge");
        const diagDetails = document.getElementById("diagnosticsDetails");
        if (data.model_diagnostics) {
            const d = data.model_diagnostics;
            diagBadge.textContent = `⚡ ${d.tokens_per_second || 0} tok/sec · ${d.inference_duration_seconds || 0}s CPU`;
            diagDetails.classList.remove("hidden");
            diagDetails.textContent = `Diagnostics: Model ${d.model_id || 'Qwen'} (${d.dtype || 'bfloat16'}) | Net RAM: ${d.model_ram_mb || '--'} MB | Generated: ${d.generated_tokens || 0} tokens in ${d.inference_duration_seconds || 0}s | Post-validation: ${d.validation_passed ? 'Passed' : 'Failed'}`;
        } else {
            diagBadge.textContent = "⚡ Fast Deterministic Response";
            diagDetails.classList.add("hidden");
        }

        // 6. Footer Disclosures
        if (data.data_sources && data.data_sources.length > 0) {
            document.getElementById("sourcesText").textContent = data.data_sources.join(", ");
        }
        if (data.attribution) {
            document.getElementById("attributionText").textContent = data.attribution;
        }
        if (data.disclaimer) {
            document.getElementById("disclaimerText").textContent = data.disclaimer;
        }

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
