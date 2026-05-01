/* second-brain — service worker. Mostly a no-op; content scripts do the
 * real work. Kept around so the popup can ping it for state and the
 * extension surface is conventional.
 */

chrome.runtime.onInstalled.addListener(() => {
    console.log("[second-brain] extension installed");
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === "ping") {
        sendResponse({ ok: true });
    }
    return true; // keep channel open for async sendResponse
});
