/* second-brain content-script common module
 *
 * Each site-specific script (chatgpt.js, gemini.js, ...) calls
 * SecondBrain.attach({ findInput, findSendButton, prependToInput }) with
 * three site-specific functions. This module handles the rest:
 *
 *   - polls / observes the DOM until the input area appears
 *   - injects a small "🧠 Brain" pill button next to the send button
 *   - on click: queries the local dashboard for context, shows a
 *     preview (the "approval UI"), then on confirm prepends the
 *     context block to the user's input
 *
 * No frameworks. Plain DOM. Defensive — every selector lookup is
 * wrapped in try/catch so the extension never breaks the host site.
 */

(function () {
    "use strict";
    if (window.__SecondBrainAttached) return;
    window.__SecondBrainAttached = true;

    const API = "http://127.0.0.1:8765";
    const POLL_MS = 1500;
    const STORAGE_K_AUTOATTACH = "sb_autoattach";

    const SecondBrain = {};

    function log(...args) {
        try { console.debug("[second-brain]", ...args); } catch (_) {}
    }

    async function fetchHealth() {
        try {
            const r = await fetch(`${API}/api/extension/health`, { method: "GET" });
            if (!r.ok) return false;
            const j = await r.json();
            return !!j.ok;
        } catch (e) { return false; }
    }

    async function fetchContext(query, k = 5) {
        try {
            const url = `${API}/api/extension/search?q=${encodeURIComponent(query)}&k=${k}`;
            const r = await fetch(url, { method: "GET" });
            if (!r.ok) return { error: `${r.status} ${r.statusText}` };
            return await r.json();
        } catch (e) {
            return { error: String(e) };
        }
    }

    function makePill(label) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.setAttribute("data-second-brain", "pill");
        btn.style.cssText = `
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 10px; margin: 0 6px;
            border: 1px solid #2c2c2c; border-radius: 999px;
            background: #0e0e0e; color: #7fff7f;
            font: 600 12px 'JetBrains Mono', 'SF Mono', monospace;
            cursor: pointer; user-select: none;
            box-shadow: 0 0 8px rgba(127,255,127,0.15);
        `;
        btn.addEventListener("mouseenter", () => {
            btn.style.background = "#181818";
            btn.style.boxShadow = "0 0 12px rgba(127,255,127,0.35)";
        });
        btn.addEventListener("mouseleave", () => {
            btn.style.background = "#0e0e0e";
            btn.style.boxShadow = "0 0 8px rgba(127,255,127,0.15)";
        });
        return btn;
    }

    function buildContextBlock(results) {
        // Compose the user-facing context preface. Kept short so it doesn't
        // dominate the prompt; LLM still sees plenty of room for the actual
        // question.
        const lines = ["[Context from your second brain — verify before relying on it]"];
        results.forEach((r, i) => {
            lines.push(``);
            lines.push(`(${i + 1}) ${r.path} (chunk ${r.chunk_index})`);
            lines.push(r.snippet || "");
        });
        lines.push("");
        lines.push("[End of context]");
        lines.push("");
        return lines.join("\n");
    }

    function buildPreview(results, host) {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed; inset: 0; z-index: 2147483647;
            background: rgba(0,0,0,0.7); backdrop-filter: blur(4px);
            display: flex; align-items: center; justify-content: center;
            font: 14px 'JetBrains Mono', 'SF Mono', monospace;
            color: #d4d4c8;
        `;
        const card = document.createElement("div");
        card.style.cssText = `
            width: min(640px, 92vw); max-height: 80vh;
            background: #111; border: 1px solid #4abe4a; border-radius: 4px;
            box-shadow: 0 0 60px rgba(127,255,127,0.25), 0 14px 50px rgba(0,0,0,0.85);
            overflow: hidden; display: flex; flex-direction: column;
        `;
        const header = document.createElement("div");
        header.style.cssText = "padding: 14px 18px; border-bottom: 1px solid #2c2c2c; color: #7fff7f;";
        header.textContent = `Brain context preview · ${results.length} chunk(s) for ${host}`;
        const body = document.createElement("div");
        body.style.cssText = "padding: 12px 18px; overflow: auto; flex: 1; white-space: pre-wrap; line-height: 1.6;";
        const previewText = results.map((r, i) =>
            `── (${i + 1}) ${r.path} · chunk ${r.chunk_index} · score ${r.score}\n${r.snippet || ""}`
        ).join("\n\n");
        body.textContent = previewText || "(no results)";
        const footer = document.createElement("div");
        footer.style.cssText = "padding: 12px 18px; border-top: 1px solid #2c2c2c; display: flex; gap: 10px; justify-content: flex-end;";
        const cancel = document.createElement("button");
        cancel.textContent = "CANCEL";
        cancel.style.cssText = "padding: 6px 14px; background: transparent; color: #888; border: 1px solid #2c2c2c; border-radius: 2px; font: 600 11px monospace; cursor: pointer; letter-spacing: 0.06em;";
        const inject = document.createElement("button");
        inject.textContent = "INJECT CONTEXT";
        inject.style.cssText = "padding: 6px 14px; background: #0e0e0e; color: #7fff7f; border: 1px solid #4abe4a; border-radius: 2px; font: 600 11px monospace; cursor: pointer; letter-spacing: 0.06em;";

        footer.appendChild(cancel);
        footer.appendChild(inject);
        card.appendChild(header);
        card.appendChild(body);
        card.appendChild(footer);
        overlay.appendChild(card);

        return new Promise((resolve) => {
            const close = (decision) => {
                overlay.remove();
                resolve(decision);
            };
            cancel.addEventListener("click", () => close(false));
            inject.addEventListener("click", () => close(true));
            overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });
            document.addEventListener("keydown", function onKey(e) {
                if (e.key === "Escape") {
                    document.removeEventListener("keydown", onKey);
                    close(false);
                }
            });
            document.body.appendChild(overlay);
        });
    }

    SecondBrain.attach = function ({ findInput, findSendButton, prependToInput, host }) {
        const handlerName = `__sb_${host || "site"}`;

        async function onPillClick(input) {
            const text = (input.value !== undefined ? input.value : input.innerText) || "";
            const trimmed = text.trim();
            if (trimmed.length < 3) {
                alert("[second-brain] Type your prompt first; the brain searches for context related to it.");
                return;
            }
            // Health check — gives a useful error instead of silent failure.
            const ok = await fetchHealth();
            if (!ok) {
                alert("[second-brain] Couldn't reach http://127.0.0.1:8765. Is `secondbrain dashboard` running?");
                return;
            }
            const data = await fetchContext(trimmed, 5);
            if (data.error) {
                alert(`[second-brain] Search failed: ${data.error}`);
                return;
            }
            const results = data.results || [];
            if (results.length === 0) {
                alert("[second-brain] No relevant context found in your brain for this prompt.");
                return;
            }
            const ok2 = await buildPreview(results, host || "AI host");
            if (!ok2) return;
            const block = buildContextBlock(results);
            try {
                prependToInput(input, block);
            } catch (e) {
                log("prependToInput failed:", e);
                alert("[second-brain] Could not inject — site DOM may have changed. See console for details.");
            }
        }

        function tryInject() {
            try {
                const input = findInput();
                const sendBtn = findSendButton();
                if (!input || !sendBtn) return false;
                if (input.dataset.sbInjected === "1") return true;
                if (sendBtn.parentElement && !sendBtn.parentElement.querySelector('[data-second-brain="pill"]')) {
                    const pill = makePill("🧠 brain");
                    pill.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); onPillClick(input); });
                    sendBtn.parentElement.insertBefore(pill, sendBtn);
                    input.dataset.sbInjected = "1";
                    log("pill injected on", host);
                    return true;
                }
            } catch (e) {
                log("inject error:", e);
            }
            return false;
        }

        // Poll until the chat UI mounts; many of these sites SSR a shell
        // and hydrate the chat composer asynchronously.
        const interval = setInterval(() => {
            if (tryInject()) {
                // Keep watching — user might navigate without a reload, the
                // input element can be replaced. Check periodically and
                // re-inject if needed.
            }
        }, POLL_MS);
        // Observer also catches React rerenders that swap the DOM.
        const obs = new MutationObserver(() => { tryInject(); });
        obs.observe(document.body, { childList: true, subtree: true });
        // Initial attempt
        tryInject();
    };

    window.SecondBrain = SecondBrain;
})();
