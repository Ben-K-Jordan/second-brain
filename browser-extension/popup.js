/* Polls the local dashboard health endpoint and updates the popup. */

const API = "http://127.0.0.1:8765";
const statusEl = document.getElementById("status");

async function check() {
    try {
        const r = await fetch(`${API}/api/extension/health`, { method: "GET" });
        if (r.ok) {
            const j = await r.json();
            statusEl.textContent = j.ok ? "reachable" : "unhealthy";
            statusEl.className = "v " + (j.ok ? "ok" : "bad");
            return;
        }
    } catch (e) {}
    statusEl.textContent = "offline";
    statusEl.className = "v bad";
}

check();
