/* second-brain — service worker.
 *
 * The bearer token for the local /api/extension/* endpoints is held here in
 * chrome.storage.local and is NEVER exposed to content scripts. Content
 * scripts message-pass requests through this worker; only the worker reads
 * the token, attaches Authorization, and proxies the response back. This way
 * a content script (or any other JS on the page's origin) cannot exfiltrate
 * the token to a third-party.
 */

const API = "http://127.0.0.1:8765";

async function getToken() {
    const r = await chrome.storage.local.get(["sb_token"]);
    return r && r.sb_token ? r.sb_token : "";
}

async function authedFetch(path, init) {
    const token = await getToken();
    if (!token) {
        return { error: "no_token" };
    }
    const headers = Object.assign({}, (init && init.headers) || {});
    headers["Authorization"] = `Bearer ${token}`;
    let r;
    try {
        r = await fetch(`${API}${path}`, Object.assign({}, init || {}, { headers }));
    } catch (e) {
        return { error: "fetch_failed", detail: String(e) };
    }
    if (r.status === 401) return { error: "unauthorized" };
    if (!r.ok) return { error: `http_${r.status}` };
    try {
        return { ok: true, body: await r.json() };
    } catch (e) {
        return { error: "bad_json" };
    }
}

chrome.runtime.onInstalled.addListener(() => {
    console.log("[second-brain] extension installed");
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !msg.type) return false;
    if (msg.type === "ping") {
        sendResponse({ ok: true });
        return true;
    }
    if (msg.type === "health") {
        authedFetch("/api/extension/health", { method: "GET" }).then(sendResponse);
        return true;
    }
    if (msg.type === "search") {
        const q = encodeURIComponent(msg.q || "");
        const k = Math.max(1, Math.min(10, msg.k || 5));
        authedFetch(`/api/extension/search?q=${q}&k=${k}`, { method: "GET" }).then(sendResponse);
        return true;
    }
    if (msg.type === "set_token") {
        chrome.storage.local.set({ sb_token: msg.token || "" }).then(() => {
            sendResponse({ ok: true });
        });
        return true;
    }
    if (msg.type === "has_token") {
        getToken().then((t) => sendResponse({ ok: true, has_token: !!t }));
        return true;
    }
    return false;
});
