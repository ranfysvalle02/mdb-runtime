# Git Hooks

This directory contains Git hooks that are shared across the team. These hooks are tracked in Git and can be installed using the setup script.

## Installation

Run the install script to copy these hooks to `.git/hooks/`:

```bash
./scripts/install-hooks.sh
```

## Available Hooks

### pre-commit

Runs before committing. Automatically formats and lints code:
- Auto-formats code with `make format`
- Stages formatted files automatically
- Runs linting checks (`make lint-local`)

If any check fails, the commit is blocked.

**What it does:**
1. Auto-formats code with `make format`
2. Stages any formatting changes automatically
3. Runs full `make lint-local` (including semgrep) for comprehensive checks

**Note:** Formatted files are automatically staged, so they'll be included in your commit.

### pre-push

Runs before pushing to GitHub. Verifies code quality (read-only checks):
- Verifies code formatting (`ruff format --check`)
- Verifies code quality (`make lint-ci`)

If any check fails, the push is blocked.

**What it does:**
1. Verifies formatting with `ruff format --check` (no auto-fix)
2. Verifies code quality with `make lint-ci` (no auto-fix)
3. Optionally runs integration tests if Docker is available (can be skipped)

**Skip integration tests:**
```bash
SKIP_INTEGRATION_TESTS=1 git push
```

**Note:** Formatting and linting should be handled by the pre-commit hook. This is a final verification step. Integration tests are optional and won't block the push if they fail or Docker isn't available.

**Skip the hook (not recommended):**
```bash
SKIP_HOOKS=1 git push
```

**Common issues:**

- **"ruff not found"**: The hook will try to install dev dependencies automatically, or run:
  ```bash
  pip install -e ".[dev]"
  ```

- **Pre-commit: "Formatting failed"**: Run manually to see errors:
  ```bash
  make format
  ```

- **Pre-commit: "Linting failed"**: Fix the errors shown, or run:
  ```bash
  make fix         # Auto-fix issues
  make lint-local  # See all errors
  ```

- **Pre-push: "Formatting check failed"**: Code wasn't formatted before commit:
  ```bash
  make format
  git add .
  git commit --amend --no-edit  # Add formatting to last commit
  ```

- **Pre-push: "Code quality check failed"**: Linting issues weren't fixed before commit:
  ```bash
  make lint-local  # See issues
  # Fix issues, then:
  git add .
  git commit --amend --no-edit
  ```

## Adding New Hooks

1. Add your hook script to this directory (`githooks/`)
2. Make it executable: `chmod +x githooks/your-hook-name`
3. Run `./scripts/install-hooks.sh` to install it
4. Commit both the hook and the updated install script

## Note

Git hooks are **local-only** - they run on your machine, not on GitHub. However, by storing them here and providing the install script, we ensure all team members can easily set them up.

