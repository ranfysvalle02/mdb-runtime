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
- Code passes linting checks (`make lint`)

If either check fails, the push is blocked.

## Adding New Hooks

1. Add your hook script to this directory (`githooks/`)
2. Make it executable: `chmod +x githooks/your-hook-name`
3. Run `./scripts/install-hooks.sh` to install it
4. Commit both the hook and the updated install script

## Note

Git hooks are **local-only** - they run on your machine, not on GitHub. However, by storing them here and providing the install script, we ensure all team members can easily set them up.

