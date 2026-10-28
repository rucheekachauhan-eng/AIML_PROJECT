/**
 * ==============================================================================
 * MediCost AI — Interactive Frontend Controller (Vanilla JavaScript)
 * Handles:
 * - Dynamic BMI classification & live badge updates
 * - Built-in BMI calculator (Height & Weight)
 * - Quick patient demo presets (Youth, Adult, High-Risk Smoker)
 * - Asynchronous Fetch API predictions with probability animation
 * - Prediction History audit log search and auto-refresh
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    // Form Elements
    const form = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");
    const resetFormBtn = document.getElementById("resetFormBtn");

    const ageInput = document.getElementById("age");
    const sexInput = document.getElementById("sex");
    const bmiInput = document.getElementById("bmi");
    const childrenInput = document.getElementById("children");
    const smokerInput = document.getElementById("smoker");
    const regionInput = document.getElementById("region");

    // BMI Components
    const bmiStatusBadge = document.getElementById("bmiStatusBadge");
    const toggleBmiCalcBtn = document.getElementById("toggleBmiCalcBtn");
    const bmiCalcDrawer = document.getElementById("bmiCalcDrawer");
    const calcHeight = document.getElementById("calcHeight");
    const calcWeight = document.getElementById("calcWeight");
    const applyBmiBtn = document.getElementById("applyBmiBtn");

    // Presets
    const presetHealthy = document.getElementById("presetHealthy");
    const presetModerate = document.getElementById("presetModerate");
    const presetHighRisk = document.getElementById("presetHighRisk");

    // Result Elements
    const liveStatusBadge = document.getElementById("liveStatusBadge");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const outcomeBanner = document.getElementById("outcomeBanner");
    const resCategory = document.getElementById("resCategory");
    const resConfidence = document.getElementById("resConfidence");
    const resChargesRange = document.getElementById("resChargesRange");

    const probLowVal = document.getElementById("probLowVal");
    const probLowBar = document.getElementById("probLowBar");
    const probMedVal = document.getElementById("probMedVal");
    const probMedBar = document.getElementById("probMedBar");
    const probHighVal = document.getElementById("probHighVal");
    const probHighBar = document.getElementById("probHighBar");

    const resSummaryText = document.getElementById("resSummaryText");
    const resRiskList = document.getElementById("resRiskList");
    const resRecText = document.getElementById("resRecText");

    const printReportBtn = document.getElementById("printReportBtn");
    const newPredBtn = document.getElementById("newPredBtn");

    // History Table Elements
    const historyTableBody = document.getElementById("historyTableBody");
    const historySearch = document.getElementById("historySearch");
    const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

    let allHistoryRecords = [];

    // ==========================================================================
    // 1. DYNAMIC BMI BADGE & CALCULATOR
    // ==========================================================================
    function updateBmiBadge() {
        const val = parseFloat(bmiInput.value);
        if (isNaN(val) || val <= 0) {
            bmiStatusBadge.textContent = "BMI --";
            bmiStatusBadge.className = "bmi-badge";
            return;
        }

        if (val < 18.5) {
            bmiStatusBadge.textContent = `Underweight (${val.toFixed(1)})`;
            bmiStatusBadge.className = "bmi-badge bmi-underweight";
        } else if (val < 25.0) {
            bmiStatusBadge.textContent = `Normal (${val.toFixed(1)})`;
            bmiStatusBadge.className = "bmi-badge bmi-normal";
        } else if (val < 30.0) {
            bmiStatusBadge.textContent = `Overweight (${val.toFixed(1)})`;
            bmiStatusBadge.className = "bmi-badge bmi-overweight";
        } else {
            bmiStatusBadge.textContent = `Obese (${val.toFixed(1)})`;
            bmiStatusBadge.className = "bmi-badge bmi-obese";
        }
    }

    bmiInput.addEventListener("input", updateBmiBadge);
    updateBmiBadge(); // Initial call

    // Toggle Collapsible BMI Calculator
    toggleBmiCalcBtn.addEventListener("click", () => {
        bmiCalcDrawer.classList.toggle("hidden");
    });

    // Calculate BMI from Height (cm) & Weight (kg)
    applyBmiBtn.addEventListener("click", () => {
        const h = parseFloat(calcHeight.value);
        const w = parseFloat(calcWeight.value);

        if (!h || !w || h <= 0 || w <= 0) {
            alert("Please enter valid height (cm) and weight (kg).");
            return;
        }

        const heightMeters = h / 100;
        const computedBmi = (w / (heightMeters * heightMeters)).toFixed(1);
        bmiInput.value = computedBmi;
        updateBmiBadge();
        bmiCalcDrawer.classList.add("hidden");
    });

    // ==========================================================================
    // 2. PRESET PROFILES
    // ==========================================================================
    presetHealthy.addEventListener("click", () => {
        ageInput.value = 22;
        sexInput.value = "female";
        bmiInput.value = 21.5;
        childrenInput.value = 0;
        smokerInput.value = "no";
        regionInput.value = "northwest";
        updateBmiBadge();
    });

    presetModerate.addEventListener("click", () => {
        ageInput.value = 42;
        sexInput.value = "male";
        bmiInput.value = 27.5;
        childrenInput.value = 2;
        smokerInput.value = "no";
        regionInput.value = "southwest";
        updateBmiBadge();
    });

    presetHighRisk.addEventListener("click", () => {
        ageInput.value = 52;
        sexInput.value = "male";
        bmiInput.value = 35.8;
        childrenInput.value = 2;
        smokerInput.value = "yes";
        regionInput.value = "southeast";
        updateBmiBadge();
    });

    resetFormBtn.addEventListener("click", () => {
        form.reset();
        ageInput.value = 30;
        bmiInput.value = 25.0;
        updateBmiBadge();
    });

    // ==========================================================================
    // 3. ASYNCHRONOUS FORM SUBMISSION & INFERENCE (FETCH API)
    // ==========================================================================
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. Gather and parse values
        const payload = {
            age: parseInt(ageInput.value),
            sex: sexInput.value,
            bmi: parseFloat(bmiInput.value),
            children: parseInt(childrenInput.value),
            smoker: smokerInput.value,
            region: regionInput.value
        };

        // Client-side quick validation
        if (isNaN(payload.age) || payload.age < 18 || payload.age > 100) {
            alert("Please enter an age between 18 and 100.");
            ageInput.focus();
            return;
        }

        if (isNaN(payload.bmi) || payload.bmi < 10 || payload.bmi > 70) {
            alert("Please enter a realistic BMI between 10.0 and 70.0.");
            bmiInput.focus();
            return;
        }

        // 2. Button Loading State
        submitBtn.disabled = true;
        const originalBtnHTML = submitBtn.innerHTML;
        submitBtn.innerHTML = `<span>⏳ Evaluating Scikit-Learn Pipeline...</span>`;
        liveStatusBadge.textContent = "Processing";
        liveStatusBadge.className = "live-status-badge";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok && data.status === "success") {
                renderPredictionOutcome(data.prediction);
                fetchHistory(); // Automatically refresh SQLite table
            } else {
                alert(`Prediction Error: ${data.message || "Failed to process prediction."}`);
            }
        } catch (err) {
            console.error("Network Error:", err);
            alert("Could not communicate with the Flask backend. Please ensure 'python app.py' is running.");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnHTML;
        }
    });

    // ==========================================================================
    // 4. RENDER PREDICTION RESULTS & ANIMATIONS
    // ==========================================================================
    function renderPredictionOutcome(pred) {
        // Toggle view
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.remove("hidden");

        liveStatusBadge.textContent = "Predicted";
        liveStatusBadge.className = "live-status-badge active";

        // Set Outcome Banner Color Class
        outcomeBanner.className = "outcome-banner";
        if (pred.category === "Low") {
            outcomeBanner.classList.add("outcome-low");
        } else if (pred.category === "Medium") {
            outcomeBanner.classList.add("outcome-med");
        } else {
            outcomeBanner.classList.add("outcome-high");
        }

        resCategory.textContent = `${pred.category} Cost Tier`;
        resChargesRange.textContent = pred.charges_range;
        resConfidence.textContent = `${pred.confidence.toFixed(1)}% Confidence`;

        // Animate Probabilities
        const probs = pred.probabilities || {};
        const lowP = probs["Low"] || 0;
        const medP = probs["Medium"] || 0;
        const highP = probs["High"] || 0;

        probLowVal.textContent = `${lowP.toFixed(1)}%`;
        probMedVal.textContent = `${medP.toFixed(1)}%`;
        probHighVal.textContent = `${highP.toFixed(1)}%`;

        // Delay bar animation slightly for smooth transition
        setTimeout(() => {
            probLowBar.style.width = `${lowP}%`;
            probMedBar.style.width = `${medP}%`;
            probHighBar.style.width = `${highP}%`;
        }, 50);

        // Clinical Insights
        const insights = pred.insights || {};
        resSummaryText.textContent = insights.summary_text || "Cost assessment based on current risk factors.";

        resRiskList.innerHTML = "";
        (insights.risk_factors || []).forEach(factor => {
            const li = document.createElement("li");
            li.textContent = factor;
            resRiskList.appendChild(li);
        });

        const recs = insights.recommendations || [];
        resRecText.textContent = recs.length > 0 ? recs[0] : "Maintain regular medical follow-ups.";

        // On mobile, scroll cleanly to result
        if (window.innerWidth < 960) {
            resultContainer.scrollIntoView({ behavior: "smooth" });
        }
    }

    // Action Buttons
    printReportBtn.addEventListener("click", () => {
        window.print();
    });

    newPredBtn.addEventListener("click", () => {
        resultContent.classList.add("hidden");
        resultPlaceholder.classList.remove("hidden");
        liveStatusBadge.textContent = "Standby";
        liveStatusBadge.className = "live-status-badge";
        ageInput.focus();
    });

    // ==========================================================================
    // 5. PREDICTION HISTORY TABLE (SQLITE) & FILTER
    // ==========================================================================
    async function fetchHistory() {
        try {
            const res = await fetch("/api/history?limit=15");
            const json = await res.json();
            if (json.status === "success" && Array.isArray(json.data)) {
                allHistoryRecords = json.data;
                renderHistoryTable(allHistoryRecords);
            }
        } catch (err) {
            console.error("Failed to fetch history:", err);
            historyTableBody.innerHTML = `
                <tr><td colspan="9" class="text-center text-muted">Error loading history from SQLite database.</td></tr>
            `;
        }
    }

    function renderHistoryTable(records) {
        if (!records || records.length === 0) {
            historyTableBody.innerHTML = `
                <tr><td colspan="9" class="text-center text-muted">No predictions stored in database yet. Run an assessment above!</td></tr>
            `;
            return;
        }

        historyTableBody.innerHTML = records.map(r => {
            const cat = r.predicted_category || "Low";
            const badgeClass = cat.toLowerCase();
            const timeDisplay = r.timestamp ? r.timestamp.replace("T", " ").slice(5, 19) : "-";

            return `
                <tr>
                    <td><code>${timeDisplay}</code></td>
                    <td>${r.age}</td>
                    <td>${r.sex}</td>
                    <td>${parseFloat(r.bmi).toFixed(1)}</td>
                    <td>${r.children}</td>
                    <td><strong>${r.smoker}</strong></td>
                    <td>${r.region}</td>
                    <td><span class="badge-tier ${badgeClass}">${cat}</span></td>
                    <td>${r.confidence ? parseFloat(r.confidence).toFixed(1) + "%" : "-"}</td>
                </tr>
            `;
        }).join("");
    }

    // Live search filter on history table
    historySearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (!query) {
            renderHistoryTable(allHistoryRecords);
            return;
        }

        const filtered = allHistoryRecords.filter(r => 
            (r.predicted_category && r.predicted_category.toLowerCase().includes(query)) ||
            (r.smoker && r.smoker.toLowerCase().includes(query)) ||
            (r.sex && r.sex.toLowerCase().includes(query)) ||
            (r.region && r.region.toLowerCase().includes(query))
        );
        renderHistoryTable(filtered);
    });

    refreshHistoryBtn.addEventListener("click", fetchHistory);

    // ==========================================================================
    // 6. STAGE 5: CHART.JS ANALYTICS DASHBOARD CONTROLLER
    // ==========================================================================
    const kpiTotal = document.getElementById("kpiTotal");
    const kpiAvgConf = document.getElementById("kpiAvgConf");
    const kpiAvgAge = document.getElementById("kpiAvgAge");
    const kpiSmokers = document.getElementById("kpiSmokers");

    const refreshAnalyticsBtn = document.getElementById("refreshAnalyticsBtn");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");

    // Chart instances references
    let chartCategoryInstance = null;
    let chartSmokerInstance = null;
    let chartBmiInstance = null;
    let chartTimelineInstance = null;

    async function fetchAnalytics() {
        try {
            const res = await fetch("/api/analytics");
            const json = await res.json();

            if (json.status === "success" && json.data) {
                renderAnalyticsDashboard(json.data);
            }
        } catch (err) {
            console.error("Failed to load analytics data:", err);
        }
    }

    function renderAnalyticsDashboard(data) {
        // 1. Update KPI Cards
        const kpis = data.kpis || {};
        kpiTotal.textContent = kpis.total_predictions || 0;
        kpiAvgConf.textContent = `${(kpis.avg_confidence || 0).toFixed(1)}%`;
        kpiAvgAge.textContent = `${(kpis.avg_age || 0).toFixed(1)} yrs`;
        kpiSmokers.textContent = kpis.smoker_count || 0;

        // Verify Chart.js is loaded
        if (typeof Chart === "undefined") {
            console.warn("Chart.js library not yet loaded.");
            return;
        }

        // Global Chart Defaults
        Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
        Chart.defaults.color = "#475569";

        // 2. Chart 1: Cost Tier Distribution (Doughnut)
        const catCanvas = document.getElementById("chartCategoryDist");
        if (catCanvas) {
            const catDist = data.category_distribution || { Low: 0, Medium: 0, High: 0 };
            if (chartCategoryInstance) chartCategoryInstance.destroy();

            chartCategoryInstance = new Chart(catCanvas, {
                type: "doughnut",
                data: {
                    labels: ["Low Tier (<= $6,250)", "Medium Tier ($6,250 - $12,733)", "High Tier (> $12,733)"],
                    datasets: [{
                        data: [catDist.Low || 0, catDist.Medium || 0, catDist.High || 0],
                        backgroundColor: ["#059669", "#d97706", "#dc2626"],
                        borderWidth: 2,
                        borderColor: "#ffffff"
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: "bottom", labels: { boxWidth: 12, padding: 12, font: { size: 11.5 } } }
                    },
                    cutout: "65%"
                }
            });
        }

        // 3. Chart 2: Smoking Impact by Cost Tier (Grouped Bar)
        const smokerCanvas = document.getElementById("chartSmokerCross");
        if (smokerCanvas) {
            const cross = data.smoker_cross || {};
            const nonSmokerData = cross.non_smoker || { Low: 0, Medium: 0, High: 0 };
            const smokerData = cross.smoker || { Low: 0, Medium: 0, High: 0 };

            if (chartSmokerInstance) chartSmokerInstance.destroy();

            chartSmokerInstance = new Chart(smokerCanvas, {
                type: "bar",
                data: {
                    labels: ["Low Tier", "Medium Tier", "High Tier"],
                    datasets: [
                        {
                            label: "Non-Smokers",
                            data: [nonSmokerData.Low || 0, nonSmokerData.Medium || 0, nonSmokerData.High || 0],
                            backgroundColor: "#0284c7",
                            borderRadius: 6
                        },
                        {
                            label: "Smokers",
                            data: [smokerData.Low || 0, smokerData.Medium || 0, smokerData.High || 0],
                            backgroundColor: "#dc2626",
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, ticks: { precision: 0 } }
                    },
                    plugins: {
                        legend: { position: "bottom", labels: { boxWidth: 12, padding: 12 } }
                    }
                }
            });
        }

        // 4. Chart 3: BMI Clinical Classification (Polar Area / Pie)
        const bmiCanvas = document.getElementById("chartBmiDist");
        if (bmiCanvas) {
            const bmiDist = data.bmi_distribution || {};
            if (chartBmiInstance) chartBmiInstance.destroy();

            chartBmiInstance = new Chart(bmiCanvas, {
                type: "polarArea",
                data: {
                    labels: Object.keys(bmiDist),
                    datasets: [{
                        data: Object.values(bmiDist),
                        backgroundColor: ["rgba(16, 185, 129, 0.75)", "rgba(245, 158, 11, 0.75)", "rgba(239, 68, 68, 0.75)"],
                        borderColor: ["#059669", "#d97706", "#dc2626"],
                        borderWidth: 1.5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: "bottom", labels: { boxWidth: 12, padding: 10, font: { size: 11 } } }
                    }
                }
            });
        }

        // 5. Chart 4: Confidence Score Telemetry (Line)
        const timelineCanvas = document.getElementById("chartTimeline");
        if (timelineCanvas) {
            const timeline = data.timeline || [];
            if (chartTimelineInstance) chartTimelineInstance.destroy();

            chartTimelineInstance = new Chart(timelineCanvas, {
                type: "line",
                data: {
                    labels: timeline.map(t => `#${t.id} (${t.category})`),
                    datasets: [{
                        label: "Confidence %",
                        data: timeline.map(t => t.confidence),
                        borderColor: "#0e7490",
                        backgroundColor: "rgba(14, 116, 144, 0.12)",
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: "#0e7490",
                        pointRadius: 4,
                        borderWidth: 2.5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: 50, max: 100, ticks: { callback: v => v + "%" } }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    }

    refreshAnalyticsBtn.addEventListener("click", fetchAnalytics);

    clearHistoryBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to reset demo history records in the SQLite database?")) {
            return;
        }

        try {
            const res = await fetch("/api/clear-history", { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                fetchHistory();
                fetchAnalytics();
                alert("Database reset to clean demo records.");
            }
        } catch (err) {
            console.error("Failed to reset history:", err);
        }
    });

    // Update fetchHistory in form submission to also update analytics
    const originalSubmitListener = form.onsubmit;
    form.addEventListener("submit", () => {
        // After slight delay for DB write, refresh charts
        setTimeout(fetchAnalytics, 600);
    });

    // Initial fetch on page load
    fetchHistory();
    fetchAnalytics();
});
