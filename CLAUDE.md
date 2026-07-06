# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A personal documentation site published via GitHub Pages at **https://zwang96-dl.github.io**.
Each document is a standalone HTML page; the homepage is a generated portal that lists them.
The most common task here is **"help me turn X into a beginner-friendly learning doc"** — so read
the *House style* section below before writing one.

## Core convention: one project = one folder + index.html

- Each document lives in its own folder in the repo root, as `<folder>/index.html`.
- The folder name is BOTH the card title and the URL path
  (`vllm-tutorial/index.html` → `https://zwang96-dl.github.io/vllm-tutorial/`).
- Ignored by the build: folders without an `index.html`, dot-folders (e.g. `.github`), and the
  names in `IGNORE_DIRS` in `build.py` — `node_modules`, `assets`, `scripts`, `templates`.

## Build & deploy (automatic)

- `build.py` (pure Python stdlib, no dependencies) scans the project folders and regenerates the
  root `index.html` — a portal with search, category filter, and dark mode.
- `.github/workflows/deploy.yml` runs `build.py` and deploys on every push to `main`. Pushing is
  the only step needed to update the live site; the build runs on GitHub's servers, not locally.

## Optional per-document meta

Set in a project's `index.html` `<head>`: `doc-title`, `doc-category`, `doc-description`,
`doc-date`, `doc-tags`. Fallbacks: title → folder name; date → git commit date; no category → no
filter bar shown.

## House style — how to write a learning doc

Docs are for **learners**. Keep new docs consistent with `vllm-tutorial/` and
`rocm-inference-optimization/`:

- **Start from the template.** Copy `templates/doc-template.html` → `<your-folder>/index.html`,
  then replace the meta + body. It already carries the shared CSS, light/dark theme, sidebar TOC,
  callouts, tables, command blocks, an inline-SVG figure, and a click-to-expand block.
- **Plain language, beginner-first.** Explain every term in plain words the first time it appears;
  don't pile on jargon. Bilingual (中/EN) headings are welcome.
- **Show, don't wall-of-text.** Prefer tables and simple diagrams for anything complex, and draw
  diagrams as **inline HTML/CSS/SVG**, never as images.
- **Be concise; short beats complete.** Explain each idea once, in the single best form — a table
  OR a diagram OR a callout, not all three. Max ~1–2 callouts per section; fold the rest into prose.
  Don't restate the same point across sections. Keep every command / param / table / takeaway, but
  cut the wording around them. If a doc feels long, it's too long — trim it.
- **Self-contained, zero external deps.** Inline all CSS/JS; no CDN links, no heavy libraries
  (e.g. do NOT pull in mermaid, ~3 MB). Any doc-specific asset goes inside that doc's own folder.
- **Theme-aware.** Support light & dark (follow system + a toggle), like the template.

## Gotchas (learned the hard way)

- **build.py auto-injects** favicon, the `<title>` suffix (`Doc · Site`), a copyright footer, and
  the analytics snippet into every doc. **Don't write these yourself**, and leave the
  `<!--mydocs-*-->` markers alone.
- **Keep your own home link** (`<a href="..">`) in the doc — the template has one. Without it,
  build.py injects a floating "← Home" button that can clash with your layout.
- **`doc-category` is all-or-nothing:** if *any* doc sets it, the homepage grows a category filter
  bar and every doc without one shows as "Uncategorized". So give all docs a category, or none
  (currently: none).

## Common tasks

- **Add a document:** copy `templates/doc-template.html` → `<folder>/index.html`, edit it, then
  commit & push. Do not touch the root `index.html`.
- **Change site title / subtitle / author / GitHub link:** edit the `SITE` dict at the top of
  `build.py`, then push (or run `python3 build.py` to preview first).
- **Preview locally:** `python3 build.py` (it should list your new doc among the collected
  projects), then `python3 -m http.server 8000` and open a browser.

## Do NOT

- Do **not** hand-edit the root `index.html` — it is generated and overwritten on every build.
- Do **not** delete `.nojekyll` (disables Jekyll) or `.github/workflows/deploy.yml`.
