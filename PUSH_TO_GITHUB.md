# Pushing this directory as a GitHub Repository

This file documents the exact commands to push the `code-release/` directory to a fresh GitHub repository. **The actual push is not automated** — the URL must be configured by the user before any external write.

Replace the placeholder `<GITHUB_CODE_REPO_URL>` everywhere below with the real URL once a repository has been created on GitHub. The URL should look like one of:

- `https://github.com/<your-user>/tabular-cp-failure-modes.git`
- `git@github.com:<your-user>/tabular-cp-failure-modes.git`

## Option 1 — separate repository (recommended)

This treats `code-release/` as its own Git project, independent of the Overleaf manuscript branch.

```bash
cd code-release
git init
git add -A
git commit -m "Initial code release for tabular CP failure-modes paper"
git branch -M main
git remote add origin <GITHUB_CODE_REPO_URL>
git push -u origin main
```

Pros: cleanest history; the code-release repo on GitHub is self-contained, can be cloned in isolation, and can have its own issue tracker / releases.

## Option 2 — subtree push from the main project repository

This pushes `code-release/` to a separate GitHub repo from the parent repository, keeping a single working copy locally.

Run from the **parent project root** (the directory containing `code-release/`):

```bash
git subtree push --prefix=code-release <GITHUB_CODE_REPO_URL> main
```

Pros: avoids juggling two `.git/` directories; one local checkout serves both the manuscript work and the code release.

Caveat: subtree pushes can be slow on large histories. The first push of a small repository is fast; later pushes are incremental.

## After the URL is known

1. Replace the placeholder in `CITATION.cff` and in `README.md` (citation block) with the real URL.
2. Edit `../sections/A_reproducibility.tex` (the Overleaf manuscript tree) — replace "The GitHub URL will be inserted before arXiv submission." with the real URL.
3. Recompile `../main.pdf` so the arXiv-bound PDF embeds the URL.

## What NOT to push

The `.gitignore` in this directory already excludes:

- `__pycache__/`, `*.pyc` — Python bytecode.
- `.venv/`, `venv/` — local virtual environments.
- `.cache/`, `~/.cache/huggingface/`, `~/scikit_learn_data/` — dataset caches.
- `data/` — raw ACS PUMS downloads (the scripts re-fetch from public hosts).
- `*.log` — run-time logs (rerun `reproduce.sh` to regenerate).
- `.DS_Store`, editor junk.

Sanity-check before the first push:

```bash
cd code-release
git status              # should show only the intended files
git ls-files | head -40 # should list .py sources, README.md, requirements.txt, reproduce.sh, results/*.{parquet,json,md}, paper-figures/*.pdf, LICENSE, CITATION.cff, DATA.md
```
