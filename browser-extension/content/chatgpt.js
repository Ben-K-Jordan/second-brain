/* second-brain — ChatGPT (chatgpt.com / chat.openai.com) integration.
 *
 * The composer is a contenteditable ProseMirror div, not a textarea.
 * Multiple selectors tried so this survives minor UI shuffles.
 */

(function () {
    "use strict";

    function findInput() {
        return (
            document.getElementById("prompt-textarea")
            || document.querySelector('div[contenteditable="true"][role="textbox"]')
            || document.querySelector('div[contenteditable="true"].ProseMirror')
            || document.querySelector('textarea[data-id="root"]')
            || document.querySelector('textarea[placeholder*="Message"]')
        );
    }

    function findSendButton() {
        return (
            document.querySelector('button[data-testid="send-button"]')
            || document.querySelector('button[data-testid="fruitjuice-send-button"]')
            || document.querySelector('button[aria-label*="Send"]')
            || document.querySelector('form button[type="submit"]')
        );
    }

    function prependToInput(input, block) {
        // ProseMirror contenteditable: insert as text nodes split by <br>.
        if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
            input.value = block + (input.value || "");
            input.dispatchEvent(new Event("input", { bubbles: true }));
            return;
        }
        // contenteditable path
        const lines = block.split("\n");
        const frag = document.createDocumentFragment();
        lines.forEach((line, i) => {
            if (i > 0) frag.appendChild(document.createElement("br"));
            frag.appendChild(document.createTextNode(line));
        });
        // Append a final break separating context from existing user text
        frag.appendChild(document.createElement("br"));
        input.insertBefore(frag, input.firstChild);
        input.dispatchEvent(new InputEvent("input", { bubbles: true }));
        // Move cursor to the end so the user keeps typing where they were.
        const range = document.createRange();
        range.selectNodeContents(input);
        range.collapse(false);
        const sel = window.getSelection();
        if (sel) { sel.removeAllRanges(); sel.addRange(range); }
    }

    if (window.SecondBrain) {
        window.SecondBrain.attach({ findInput, findSendButton, prependToInput, host: "ChatGPT" });
    }
})();
