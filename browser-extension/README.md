# second-brain context bridge

Browser extension that injects context from your local second-brain into
ChatGPT, Gemini, Perplexity, Grok, and DeepSeek prompts. Works alongside
the MCP integration (which is the better path for Claude / Cursor / Cline) —
this fills the gap for AI tools that don't speak MCP.

## How it works

1. You open one of the supported AI sites and type your question.
2. A small `🧠 brain` pill appears next to the send button.
3. Click it. The extension queries your local dashboard at `http://127.0.0.1:8765`
   for chunks relevant to what you're typing.
4. A preview overlay shows you exactly what context will be injected
   (file paths + snippets). This is the **approval step**.
5. Confirm, and the context block is prepended to your prompt before
   you send it. Cancel and nothing happens.

The extension never sends your prompt anywhere except your local dashboard.

## Install (developer / unpacked)

1. Make sure your dashboard is running:

       secondbrain dashboard

2. Open Chrome / Edge / Brave → `chrome://extensions/` (or `edge://extensions/`).
3. Enable **Developer mode** (top-right toggle).
4. Click **Load unpacked** → pick the `browser-extension/` folder of this repo.
5. Visit any supported AI site (see below). The pill should appear next
   to the send button within a few seconds.

## Supported sites

- ChatGPT (`chatgpt.com`, `chat.openai.com`)
- Gemini (`gemini.google.com`)
- Perplexity (`perplexity.ai`)
- Grok (`x.com`, `grok.com`)
- DeepSeek (`chat.deepseek.com`)

Each site has its own content script with site-specific DOM selectors. If
a site updates its UI, those selectors may need to be adjusted in
`content/<site>.js`. Open DevTools console on the site to see
`[second-brain]` log messages diagnosing what the script is finding.

## Privacy

- The extension only fetches from `http://127.0.0.1:8765` and from no
  other host. The dashboard binds to localhost only, so this traffic
  never leaves your machine.
- Every search you trigger is logged to `~/.secondbrain/queries.jsonl`
  and visible at `http://127.0.0.1:8765/queries` so you have a full
  audit trail of what context has been retrieved on your behalf.
- The preview overlay always shows you what's about to be injected —
  there is no silent injection.

## Customizing

Default: 5 chunks per query. To change, edit `content/common.js`
(`fetchContext(trimmed, 5)`).

To disable on a specific site, just don't load the extension on it
(`chrome://extensions/` → details → block site access). Or edit
`manifest.json`'s `content_scripts.matches` entries.
