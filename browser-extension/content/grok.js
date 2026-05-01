/* second-brain — Grok (x.com/i/grok or grok.com) integration. */

(function () {
    "use strict";

    function findInput() {
        return (
            document.querySelector('textarea[placeholder*="Ask" i]')
            || document.querySelector('textarea[placeholder*="Grok" i]')
            || document.querySelector('div[contenteditable="true"][role="textbox"]')
            || document.querySelector('textarea')
        );
    }

    function findSendButton() {
        return (
            document.querySelector('button[aria-label*="Send" i]')
            || document.querySelector('button[aria-label*="Submit" i]')
            || document.querySelector('button[type="submit"]')
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
        lines.forEach((line, i) => {
            if (i > 0) frag.appendChild(document.createElement("br"));
            frag.appendChild(document.createTextNode(line));
        });
        input.insertBefore(frag, input.firstChild);
        input.dispatchEvent(new InputEvent("input", { bubbles: true }));
    }

    if (window.SecondBrain) {
        window.SecondBrain.attach({ findInput, findSendButton, prependToInput, host: "Grok" });
    }
})();
