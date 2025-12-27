# Git Commands for GitHub

This guide contains essential Git commands for committing and pushing your changes to GitHub.

## Initial Setup

### Configure Git (First Time Only)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Initialize Repository (If Not Already Done)
```bash
git init
git remote add origin https://github.com/barada02/Understanding_Tunix.git
```

## Daily Workflow

### Check Status
```bash
git status
```

### Add Files to Staging Area

Add specific file:
```bash
git add filename.md
```

Add all changes:
```bash
git add .
```

Add all Python files:
```bash
git add *.py
```

### Commit Changes
```bash
git commit -m "Your descriptive commit message"
```

### Push to GitHub

Push to main branch:
```bash
git push origin main
```

Push to current branch:
```bash
git push
```

### Pull Latest Changes
```bash
git pull origin main
```

## Common Workflow Example

```bash
# 1. Check current status
git status

# 2. Add all changes
git add .

# 3. Commit with a message
git commit -m "Add new learning materials and update roadmap"

# 4. Push to GitHub
git push origin main
```

## Branch Management

### Create and Switch to New Branch
```bash
git checkout -b feature-branch-name
```

### Switch Between Branches
```bash
git checkout main
git checkout feature-branch-name
```

### Push New Branch to GitHub
```bash
git push -u origin feature-branch-name
```

### List All Branches
```bash
git branch -a
```

## Undoing Changes

### Discard Changes in Working Directory
```bash
git checkout -- filename.md
```

### Unstage Files
```bash
git reset HEAD filename.md
```

### Amend Last Commit
```bash
git commit --amend -m "New commit message"
```

## Viewing History

### View Commit History
```bash
git log
```

### View Compact History
```bash
git log --oneline
```

### View Changes
```bash
git diff
```

## Tips

- **Commit Often**: Make small, focused commits with clear messages
- **Pull Before Push**: Always pull latest changes before pushing
- **Meaningful Messages**: Write descriptive commit messages
- **Check Status**: Use `git status` frequently to stay aware of changes

## Quick Reference

| Command | Description |
|---------|-------------|
| `git status` | Check current status |
| `git add .` | Add all changes |
| `git commit -m "message"` | Commit with message |
| `git push` | Push to remote |
| `git pull` | Pull latest changes |
| `git log` | View history |
| `git diff` | View changes |
