/* second-brain — Gemini (gemini.google.com) integration. */

(function () {
    "use strict";

    function findInput() {
        return (
            document.querySelector('rich-textarea div[contenteditable="true"]')
            || document.querySelector('div[role="textbox"][contenteditable="true"]')
            || document.querySelector('div.ql-editor[contenteditable="true"]')
            || document.querySelector('textarea[aria-label*="prompt" i]')
        );
    }

    function findSendButton() {
        return (
            document.querySelector('button[aria-label*="Send message" i]')
            || document.querySelector('button[aria-label*="Send" i]')
            || document.querySelector('button[mat-icon-button][aria-label]')
        );
    }

    function prependToInput(input, block) {
        if (input.tagName === "TEXTAREA") {
            input.value = block + (input.value || "");
            input.dispatchEvent(new Event("input", { bubbles: true }));
            return;
        }
        const lines = block.split("\n");
        const frag = document.createDocumentFragment();
        lines.forEach((line) => {
            const p = document.createElement("p");
            p.textContent = line;
            frag.appendChild(p);
        });
        input.insertBefore(frag, input.firstChild);
        input.dispatchEvent(new InputEvent("input", { bubbles: true }));
        const range = document.createRange();
        range.selectNodeContents(input);
        range.collapse(false);
        const sel = window.getSelection();
        if (sel) { sel.removeAllRanges(); sel.addRange(range); }
    }

    if (window.SecondBrain) {
        window.SecondBrain.attach({ findInput, findSendButton, prependToInput, host: "Gemini" });
    }
})();
