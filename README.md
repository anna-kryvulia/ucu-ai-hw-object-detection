# UCU AI Homework Report Template

Your report is published as a website in UCU colors. The page is generated from `report.md`, and it shows the date of your repository's last push.

## How to fill

1. Click **Use this template** (or fork) to create your own repository.
2. Go to **Settings → Pages** and set **Source** to **GitHub Actions**.
3. Edit **`report.md`**:
   - Fill in the header block at the top (`title`, `author`, `course`, `date`).
   - Write the report in Markdown. You can use tables, code blocks, and LaTeX math (`$...$` inline, `$$...$$` for display).
   - Put images in `images/` and reference them as `![caption](images/plot.png)`.
4. Commit and push to `main`. GitHub Actions deploys the site automatically within about a minute, at `https://<username>.github.io/<repo>/`.
5. Submit that link.

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` directly from disk will not work.

## ⚠️ Rules

You may change **only** `report.md` and the `images/` folder.

**Any change to the core website (`index.html`, `assets/`, `.github/`) will annul the grade for the homework.** This includes edits that change how the last-push date is shown.

The last-push date is read from GitHub and is used to check deadlines.
