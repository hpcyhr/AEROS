#!/usr/bin/env bash
# AEROS git history cleanup
# Removes large files from git history that block GitHub push (>100MB limit).
# Uses git-filter-repo to rewrite history, then force-pushes.
#
# WARNING: this rewrites history. Only run if:
#   1. The repo is personal (no collaborators relying on shared history), OR
#   2. All collaborators are notified and ready to re-clone.
#
# Usage:
#   bash cleanup_git_history.sh

set -e

REPO_DIR="/data/yhr/AEROS"
REMOTE_URL="git@github.com:hpcyhr/AEROS.git"

echo "================================================================"
echo "AEROS git history cleanup"
echo "Repo: ${REPO_DIR}"
echo "Remote: ${REMOTE_URL}"
echo "================================================================"

cd "${REPO_DIR}"

echo ""
echo "[1/6] Current repo size:"
du -sh .git/ 2>/dev/null || true
echo ""

# -----------------------------------------------------------------------------
# Step 1: backup
# -----------------------------------------------------------------------------
BACKUP="/data/yhr/AEROS_backup_$(date +%Y%m%d_%H%M%S)"
echo "[2/6] Creating backup at ${BACKUP}..."
read -p "  Proceed with backup? [y/N] " yn
case ${yn} in
    [Yy]* ) cp -r "${REPO_DIR}" "${BACKUP}"
            echo "  backup created" ;;
    * ) echo "  skipping backup (risky!)" ;;
esac

# -----------------------------------------------------------------------------
# Step 2: install git-filter-repo
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Verifying git-filter-repo..."
if ! command -v git-filter-repo &> /dev/null; then
    echo "  git-filter-repo not found. Installing via pip..."
    pip install git-filter-repo
fi
git-filter-repo --version 2>&1 | head -1

# -----------------------------------------------------------------------------
# Step 3: rewrite history
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Rewriting history to remove large files..."
echo "  Targets:"
echo "    - Prophesee_Dataset_n_cars.7z (285 MB)"
echo "    - data_extended/ (60-70 MB)"
echo "    - p9_*.json (~200+ MB total)"
echo "    - *.7z, *.tar.gz, *.zip"
echo ""
read -p "  Proceed with history rewrite? [y/N] " yn
case ${yn} in
    [Yy]* ) ;;
    * ) echo "Aborted."; exit 1 ;;
esac

# Remove specific large file
git filter-repo --invert-paths --path Prophesee_Dataset_n_cars.7z --force

# Remove data_extended directory
git filter-repo --invert-paths --path data_extended --force

# Remove all p9_*.json
git filter-repo --invert-paths --path-glob 'p9_*.json' --force

# Catch-all for archives
git filter-repo --invert-paths --path-glob '*.7z' --force
git filter-repo --invert-paths --path-glob '*.tar.gz' --force
git filter-repo --invert-paths --path-glob '*.zip' --force

# Optional: also catch .npz / .pth if they were committed
# git filter-repo --invert-paths --path-glob '*.npz' --force
# git filter-repo --invert-paths --path-glob '*.pth' --force

# -----------------------------------------------------------------------------
# Step 4: verify
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Post-rewrite verification..."
echo "  New repo size:"
du -sh .git/
echo ""
echo "  Commits in history:"
git log --oneline | head -10
echo ""
echo "  Files in HEAD:"
git ls-files | wc -l
echo "  ... files (sample):"
git ls-files | head -5
echo ""
echo "  Should NOT find these in history:"
for f in Prophesee_Dataset_n_cars.7z data_extended p9_iostream_flat_t_full.json; do
    if git log --all --oneline --diff-filter=D --name-only 2>/dev/null | grep -q "${f}"; then
        echo "    ✗ STILL FOUND: ${f}"
    else
        echo "    ✓ removed: ${f}"
    fi
done

# -----------------------------------------------------------------------------
# Step 5: re-add remote and force push
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Re-adding remote and pushing..."
echo ""
echo "  git-filter-repo removes the origin for safety."
echo "  Re-adding: ${REMOTE_URL}"
git remote add origin "${REMOTE_URL}" 2>/dev/null || \
    git remote set-url origin "${REMOTE_URL}"

echo ""
echo "  Ready to force-push to ${REMOTE_URL}"
echo "  This will OVERWRITE the remote main branch with cleaned history."
echo ""
read -p "  Proceed with force push? [y/N] " yn
case ${yn} in
    [Yy]* ) git push -u --force origin main
            echo ""
            echo "================================================================"
            echo "Done. Verify on GitHub: https://github.com/hpcyhr/AEROS"
            echo "================================================================" ;;
    * ) echo ""
        echo "Skipped force push. To push manually later:"
        echo "  cd ${REPO_DIR}"
        echo "  git push -u --force origin main"
        ;;
esac