# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A personal documentation site published via GitHub Pages at **https://zwang96-dl.github.io**.
Each document is a standalone HTML page; the homepage is a generated portal that lists them.

## Core convention: one project = one folder + index.html

- Each document lives in its own folder in the repo root, as `<folder>/index.html`.
- The folder name is BOTH the card title and the URL path
  (`vllm-paged-attention/index.html` → `https://zwang96-dl.github.io/vllm-paged-attention/`).
- Folders without an `index.html`, and dot-folders (e.g. `.github`), are ignored by the build.

## Build & deploy (automatic)

- `build.py` (pure Python stdlib, no dependencies) scans the project folders and regenerates the
  root `index.html` — a portal with search, category filter, and dark mode.
- `.github/workflows/deploy.yml` runs `build.py` and deploys on every push to `main`. Pushing is
  the only step needed to update the live site; the build runs on GitHub's servers, not locally.

## Optional per-document meta

Set in a project's `index.html` `<head>`: `doc-title`, `doc-category`, `doc-description`,
`doc-date`, `doc-tags`. Fallbacks: title → folder name; date → git commit date; no category → no
filter bar shown.

## Common tasks

- **Add a document:** create `<folder>/index.html`, then commit & push. Do not touch the root `index.html`.
- **Change site title / subtitle / author / GitHub link:** edit the `SITE` dict at the top of
  `build.py`, then push (or run `python3 build.py` to preview first). `author` is currently the
  username `zwang96-dl`.
- **Preview locally:** `python3 build.py && python3 -m http.server 8000`.

## Do NOT

- Do **not** hand-edit the root `index.html` — it is generated and overwritten on every build.
- Do **not** delete `.nojekyll` (disables Jekyll) or `.github/workflows/deploy.yml`.
