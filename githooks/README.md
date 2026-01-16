# Git Hooks

This directory contains Git hooks that are shared across the team. These hooks are tracked in Git and can be installed using the setup script.

## Installation

Run the install script to copy these hooks to `.git/hooks/`:

```bash
./scripts/install-hooks.sh
```

## Available Hooks

### pre-push

Runs before pushing to GitHub. Ensures:
- Code is formatted (`make format`)
- Code passes linting checks (`ruff check` + `make lint`)

If any check fails, the push is blocked.

**What it does:**
1. Auto-formats code with `make format`
2. Checks if formatting changed files (requires commit)
3. Runs `ruff check` for fast linting feedback
4. Runs `ruff format --check` for formatting validation
5. Runs full `make lint` (including semgrep) for comprehensive checks

**Skip the hook (not recommended):**
```bash
SKIP_HOOKS=1 git push
```

**Common issues:**

- **"ruff not found"**: The hook will try to install dev dependencies automatically, or run:
  ```bash
  pip install -e ".[dev]"
  ```

- **"Formatting modified files"**: The hook auto-formatted your code. Review and commit:
  ```bash
  git diff                    # Review changes
  git add .                   # Stage formatting changes
  git commit -m "Format code"  # Commit them
  ```

- **"Linting failed"**: Fix the errors shown, or run:
  ```bash
  make fix    # Auto-fix issues
  make lint   # See all errors
  ```

## Adding New Hooks

1. Add your hook script to this directory (`githooks/`)
2. Make it executable: `chmod +x githooks/your-hook-name`
3. Run `./scripts/install-hooks.sh` to install it
4. Commit both the hook and the updated install script

## Note

Git hooks are **local-only** - they run on your machine, not on GitHub. However, by storing them here and providing the install script, we ensure all team members can easily set them up.

