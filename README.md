# My Docs 📚

A personal documentation site hosted on **GitHub Pages**. Drop an HTML document into its own
folder, push to GitHub, and a portal homepage — with search, categories, and dark mode — is
regenerated and deployed automatically. Live at **https://zwang96-dl.github.io**.

## How it works

**One project = one folder with an `index.html` inside.** The folder name becomes both the card
title on the homepage and the URL path.

```
.
├── index.html                  # Auto-generated portal homepage (do NOT edit by hand)
├── build.py                    # Scans project folders and regenerates index.html
├── .nojekyll                   # Disables Jekyll processing (keep it)
├── welcome/
│   └── index.html              # ← one project = one folder, always named index.html
├── vllm-paged-attention/
│   └── index.html
├── .github/workflows/
│   └── deploy.yml              # Builds + deploys on every push
└── README.md
```

Every push triggers GitHub Actions, which runs `build.py` on GitHub's servers and deploys the
result — **you never run the build locally**. Any root-level folder that contains an `index.html`
becomes a card; folders without one (and dot-folders like `.github`) are ignored.

## Add a new document

```bash
mkdir cuda-warp-primitives                 # create a project folder
# write your document as  cuda-warp-primitives/index.html
git add -A
git commit -m "Add: CUDA warp primitives"
git push
```

About a minute later a new card appears on the homepage, reachable at
`https://zwang96-dl.github.io/cuda-warp-primitives/`.

> Tip: use lowercase English + hyphens for folder names to get the cleanest URLs; non-ASCII
> names work too.

## Optional: control how a card looks

Add any of these `<meta>` tags in a project's `index.html` `<head>` (all optional):

| meta | Purpose | Fallback if omitted |
|------|---------|---------------------|
| `doc-title` | Card title | `<title>`, then the folder name |
| `doc-category` | Category (adds a filter bar when any doc has one) | none |
| `doc-description` | Card blurb | empty |
| `doc-date` | Date (sorts newest first) | git commit date |
| `doc-tags` | Tags (comma-separated) | none |

```html
<meta name="doc-category"    content="Inference">
<meta name="doc-description" content="Warp-level primitives and shuffle instructions cheat sheet">
<meta name="doc-tags"        content="GPU,parallel,cheatsheet">
```

## Preview locally

```bash
python3 build.py                  # regenerate index.html
python3 -m http.server 8000       # then open http://localhost:8000
```

## Customize the site

Edit the `SITE` dict at the top of `build.py` (site title, subtitle, author name, GitHub link),
then push. The homepage is regenerated on deploy.

<details>
<summary>First-time setup from scratch (already done for this repo)</summary>

1. Edit the `SITE` dict at the top of `build.py`.
2. Create a **public** repo named exactly `<username>.github.io`.
3. Push it:
   ```bash
   git init && git add -A && git commit -m "init"
   git branch -M main
   git remote add origin git@github.com:<username>/<username>.github.io.git
   git push -u origin main
   ```
4. In the repo: **Settings → Pages → Build and deployment → Source → GitHub Actions**.
5. Wait ~1 minute, then visit `https://<username>.github.io`.
</details>
