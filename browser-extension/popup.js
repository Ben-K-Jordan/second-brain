/* Polls the local dashboard health endpoint and updates the popup. Also
 * lets the user paste in the per-install bearer token, which we save to
 * chrome.storage.local via the background service worker. The token is
 * never readable from any content script. */

const statusEl = document.getElementById("status");
const tokStateEl = document.getElementById("tokstate");
const tokEl = document.getElementById("tok");
const saveBtn = document.getElementById("save");
const savedMsg = document.getElementById("savedmsg");

function send(msg) {
    return new Promise((resolve) => {
        try {
            chrome.runtime.sendMessage(msg, (resp) => {
                if (chrome.runtime.lastError) {
                    resolve({ error: chrome.runtime.lastError.message });
                    return;
                }
                resolve(resp || {});
            });
        } catch (e) {
            resolve({ error: String(e) });
        }
    });
}

async function refresh() {
    const tok = await send({ type: "has_token" });
    tokStateEl.textContent = tok && tok.has_token ? "yes" : "no";
    tokStateEl.className = "v " + (tok && tok.has_token ? "ok" : "bad");

    const h = await send({ type: "health" });
    if (h && h.ok && h.body && h.body.ok) {
        statusEl.textContent = "reachable";
        statusEl.className = "v ok";
    } else if (h && h.error === "no_token") {
        statusEl.textContent = "no token set";
        statusEl.className = "v bad";
    } else if (h && h.error === "unauthorized") {
        statusEl.textContent = "unauthorized";
        statusEl.className = "v bad";
    } else {
        statusEl.textContent = "offline";
        statusEl.className = "v bad";
    }
}

saveBtn.addEventListener("click", async () => {
    const v = (tokEl.value || "").trim();
    await send({ type: "set_token", token: v });
    tokEl.value = "";
    savedMsg.style.display = "block";
    setTimeout(() => { savedMsg.style.display = "none"; }, 1500);
    refresh();
});

refresh();
