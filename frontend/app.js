// ========================= BACKEND CONFIG =========================
const API = "http://localhost:8000/api/v1"; // change host/port if needed
const VEHICLE_ID = 12;                      // as per EY spec
const WORKSHOP_ID = 3;                      // workshop "Hero RT Nagar"
const USER_ID = 7;                          // demo user
let LAST_ANOMALY_ID = 101;                  // matches dummy data from docs

// ========================= SPA NAVIGATION =========================
document.addEventListener("DOMContentLoaded", () => {
  // default page
  setActivePage("landing");

  // nav click handlers
  document.querySelectorAll("[data-nav]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-nav");
      setActivePage(target);
    });
  });

  // action buttons
  document
    .querySelectorAll("[data-action='simulate-anomaly']")
    .forEach((btn) => btn.addEventListener("click", submitAnomaly));

  document
    .querySelectorAll("[data-action='book-job']")
    .forEach((btn) => btn.addEventListener("click", bookJob));

  document
    .querySelectorAll("[data-action='refresh-jobs']")
    .forEach((btn) => btn.addEventListener("click", loadWorkshopConsole));

  document
    .querySelectorAll("[data-action='simulate-whatsapp']")
    .forEach((btn) => btn.addEventListener("click", simulateWhatsAppYes));
});

function setActivePage(page) {
  document
    .querySelectorAll(".page-section")
    .forEach((p) => p.classList.remove("active"));

  const active = document.getElementById(`page-${page}`);
  if (active) active.classList.add("active");

  if (page === "rider") loadRiderDashboard();
  if (page === "workshop") loadWorkshopConsole();
}

// =================================================================
//                         RIDER DASHBOARD
// =================================================================
async function loadRiderDashboard() {
  try {
    // 1️⃣ GET /vehicles/{id}/telemetry/latest
    const telemetry = await fetchJSON(
      `${API}/vehicles/${VEHICLE_ID}/telemetry/latest`
    );

    // 2️⃣ GET /vehicles/{id}/anomalies
    const anomalies = await fetchJSON(
      `${API}/vehicles/${VEHICLE_ID}/anomalies`
    );

    updateRiderSnapshot(telemetry);
    updateRiderProtections(anomalies);
  } catch (err) {
    console.error("Failed to load rider dashboard:", err);
  }
}

function updateRiderSnapshot(t) {
  // t = telemetry JSON from EY doc:
  // {
  //   "vehicle_id": 12,
  //   "vehicle_name": "...",
  //   "timestamp": "2025-12-07T10:20:00Z",
  //   "speed": 43,
  //   "rpm": 3200,
  //   "engine_temp": 103.5,
  //   "battery_voltage": 12.6,
  //   "tyre_fl": 30.5, ...
  //   "health_index": 0.62,
  //   "has_open_anomaly": true
  // }

  const soc = Math.round((t.health_index ?? 0.8) * 100);
  const estRange = Math.round(soc * 1.2); // simple demo formula

  document.getElementById("rider-soc").textContent = soc + "%";
  document.getElementById("rider-range").textContent = estRange + " km";
  document.getElementById("rider-eff").textContent = "33 Wh/km";

  document.getElementById("rider-last-sync").textContent =
    "Last sync · " + (t.timestamp || "--");

  document.getElementById("rider-mode").textContent = t.has_open_anomaly
    ? "Risk Mode · anomaly detected"
    : "Normal mode";

  document.getElementById("rider-tyre-status").textContent =
    `Tyres: ${t.tyre_fl} / ${t.tyre_fr} / ${t.tyre_rl} / ${t.tyre_rr} psi`;
}

function updateRiderProtections(anomalies) {
  // anomalies list from EY doc:
  // [
  //   {
  //     "id": 101,
  //     "vehicle_id": 12,
  //     "type": "engine_overheat",
  //     "severity": "high",
  //     "first_ts": "...",
  //     "status": "open",
  //     "message": "Engine temperature above safe limit for 12 minutes."
  //   }
  // ]

  const listEl = document.getElementById("rider-protections-list");
  listEl.innerHTML = "";

  if (!Array.isArray(anomalies) || anomalies.length === 0) {
    listEl.innerHTML = `
      <li class="list-item">
        <div class="list-main">
          <span class="list-title">No active anomalies</span>
          <span class="list-meta">Your vehicle looks healthy right now.</span>
        </div>
      </li>
    `;
    return;
  }

  anomalies.forEach((a) => {
    LAST_ANOMALY_ID = a.id;

    const li = document.createElement("li");
    li.className = "list-item";

    const title = formatAnomaly(a.type);
    const severity = a.severity || "high";

    li.innerHTML = `
      <div class="list-main">
        <span class="list-title">${title}</span>
        <span class="list-meta">${a.message}</span>
      </div>
      <span class="pill-severity">${severity}</span>
    `;

    listEl.appendChild(li);
  });
}

function formatAnomaly(type) {
  if (!type) return "Anomaly";
  return type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// =================================================================
//                         WORKSHOP CONSOLE
// =================================================================
async function loadWorkshopConsole() {
  try {
    // 5️⃣ GET /workshops/{id}/jobs
    const jobs = await fetchJSON(`${API}/workshops/${WORKSHOP_ID}/jobs`);
    updateWorkshopJobs(jobs);
  } catch (err) {
    console.error("Failed to load workshop console:", err);
  }
}

function updateWorkshopJobs(jobs) {
  // jobs from EY doc:
  // [
  //   {
  //     "id": 88,
  //     "vehicle_id": 12,
  //     "vehicle_reg": "KA-01-AB-1234",
  //     "customer_name": "Suresh",
  //     "anomaly_type": "engine_overheat",
  //     "severity": "high",
  //     "status": "booked",
  //     "slot_start": "...",
  //     "slot_end": "..."
  //   }
  // ]

  const listEl = document.getElementById("workshop-jobs-list");
  listEl.innerHTML = "";

  if (!Array.isArray(jobs) || jobs.length === 0) {
    listEl.innerHTML = `
      <div class="list-item">
        <div class="list-main">
          <span class="list-title">No open jobs</span>
          <span class="list-meta">You’re all caught up for now.</span>
        </div>
      </div>
    `;
    return;
  }

  jobs.forEach((job) => {
    const row = document.createElement("div");
    row.className = "list-item";

    row.innerHTML = `
      <div class="list-main">
        <span class="list-title">
          ${job.vehicle_reg || "Vehicle " + job.vehicle_id} · ${formatAnomaly(
      job.anomaly_type
    )}
        </span>
        <span class="list-meta">
          ${job.customer_name || "Customer"} · ${job.status} ·
          ${job.slot_start} → ${job.slot_end}
        </span>
      </div>
      <span class="pill-severity">${job.severity || "high"}</span>
    `;

    listEl.appendChild(row);
  });
}

// =================================================================
//                       ACTION: POST /anomalies
// =================================================================
async function submitAnomaly() {
  try {
    // 3️⃣ POST /api/v1/anomalies
    const body = {
      vehicle_id: VEHICLE_ID,
      type: "engine_overheat",
      severity: "high",
      score: 0.91,
    };

    const response = await fetchJSON(`${API}/anomalies`, "POST", body);
    LAST_ANOMALY_ID = response.id || LAST_ANOMALY_ID;

    alert("Anomaly created with ID " + response.id);
    // refresh rider view to see it
    loadRiderDashboard();
  } catch (err) {
    console.error("Error submitting anomaly:", err);
    alert("Failed to submit anomaly. Check console.");
  }
}

// =================================================================
//                       ACTION: POST /jobs
// =================================================================
async function bookJob() {
  try {
    // 4️⃣ POST /api/v1/jobs
    const body = {
      user_id: USER_ID,
      vehicle_id: VEHICLE_ID,
      anomaly_id: LAST_ANOMALY_ID,
      workshop_id: WORKSHOP_ID,
      source: "app",
      preferred_slot: {
        start: "2025-12-07T16:00:00Z",
        end: "2025-12-07T18:00:00Z",
      },
    };

    const response = await fetchJSON(`${API}/jobs`, "POST", body);
    alert(`Job booked! Job ID = ${response.id}`);
    // refresh workshop console to show job
    loadWorkshopConsole();
  } catch (err) {
    console.error("Error booking job:", err);
    alert("Failed to book job. Check console.");
  }
}

// =================================================================
//               OPTIONAL: POST /webhooks/whatsapp/inbound
// =================================================================
async function simulateWhatsAppYes() {
  try {
    const body = {
      from: "+919876543210",
      message: "Yes",
      metadata: {
        user_id: USER_ID,
        vehicle_id: VEHICLE_ID,
        anomaly_id: LAST_ANOMALY_ID,
      },
    };

    const response = await fetchJSON(
      `${API}/webhooks/whatsapp/inbound`,
      "POST",
      body
    );

    alert(
      `Webhook processed. Status: ${response.status}, Job ID: ${response.job_id}`
    );
    loadWorkshopConsole();
  } catch (err) {
    console.error("Error simulating WhatsApp webhook:", err);
    alert("Failed to simulate WhatsApp webhook. Check console.");
  }
}

// =================================================================
//                          HELPERS
// =================================================================
async function fetchJSON(url, method = "GET", body = null) {
  const res = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  if (!res.ok) {
    throw new Error(`API Error: ${res.status} @ ${url}`);
  }
  return res.json();
}
