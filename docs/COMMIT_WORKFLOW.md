# Reusable commit workflow (with Entire)

Use this when committing on feature branches (e.g. `test/from-main/v1`) so changes show on the **Entire checkpoint dashboard**.

---

## 1. Make your changes

Edit files as usual. Avoid committing `.DS_Store` (add `**/.DS_Store` to `.gitignore` if needed).

---

## 2. Stage only what you want

```bash
# Stage specific paths (recommended)
git add path/to/file1 path/to/dir/

# Or stage all except .DS_Store
git add .
git reset HEAD -- **/.DS_Store .DS_Store
```

---

## 3. Commit **from Cursor** (so Entire links the checkpoint)

- Open **Source Control** in Cursor (not the terminal).
- Write a clear commit message (e.g. `feat: add X`, `fix: Y`, `chore: Z`).
- Commit via the **✓ Commit** button.
- When prompted **"Link this commit to Claude Code session context?"** → choose **Yes** so the commit gets the `Entire-Checkpoint` trailer and appears on the dashboard.

If you commit from the terminal instead, the trailer is usually not added and the commit won’t show as a checkpoint on Entire.

---

## 4. Push (and fix “behind remote” if needed)

```bash
git push origin <your-branch>
```

If you see **rejected (non-fast-forward)** because your branch is behind the remote:

```bash
git pull --rebase origin <your-branch>
git push origin <your-branch>
```

Then push again; no force push needed.

---

## Quick reference

| Step | Action |
|------|--------|
| 1 | Edit files; don’t commit `.DS_Store` |
| 2 | `git add` only the paths you want |
| 3 | **Commit in Cursor Source Control** → say **Yes** to link session (Entire) |
| 4 | `git push`; if rejected, `git pull --rebase origin <branch>` then `git push` |

---

## Optional: ignore .DS_Store everywhere

Add to `.gitignore`:

```
# macOS
.DS_Store
**/.DS_Store
```

Then future `git add .` won’t pick them up.
