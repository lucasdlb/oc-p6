#!/usr/bin/env bash
# release.sh — Bump package versions, commit, tag, and push.
#
# Usage:
#   ./scripts/release.sh patch   # 0.1.1 → 0.1.2
#   ./scripts/release.sh minor   # 0.1.1 → 0.2.0
#   ./scripts/release.sh major   # 0.1.1 → 1.0.0
#
# Reads the current version from packages/credit-risk-data/pyproject.toml
# (the source of truth), bumps it in all 3 packages/*/pyproject.toml,
# commits the change, creates 3 annotated tags, and pushes everything.

set -euo pipefail

# ─── Config ─────────────────────────────────────────────────────────────────

PACKAGES=(
    "credit-risk-data"
    "credit-risk-models"
    "credit-risk-processing"
)

REF_PYPROJECT="packages/credit-risk-data/pyproject.toml"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ─── Helpers ───────────────────────────────────────────────────────────────

err() { echo "ERROR: $*" >&2; exit 1; }

log() { echo "==> $*"; }

get_current_version() {
    local version_line
    version_line=$(grep -E '^version = ' "$REF_PYPROJECT")
    echo "$version_line" | sed -E 's/version = "([^"]*)"/\1/'
}

set_version_in_file() {
    local file="$1"
    local new_version="$2"
    # Use sed to replace the version = "..." line
    sed -i -E "s/^version = \"[^\"]*\"/version = \"${new_version}\"/" "$file"
}

bump_version() {
    local current="$1"
    local bump_type="$2"
    local major minor patch

    IFS='.' read -r major minor patch <<<"$current"

    case "$bump_type" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            err "Unknown bump type: '$bump_type'. Use patch, minor, or major."
            ;;
    esac

    echo "${major}.${minor}.${patch}"
}

# ─── Pre-flight checks ─────────────────────────────────────────────────────

[[ -f "$REF_PYPROJECT" ]] || err "Cannot find $REF_PYPROJECT"

[[ -n "$(git status --porcelain)" ]] && \
    err "Working tree is not clean. Commit or stash your changes first."

# Ensure we're on main or master
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" && "$BRANCH" != "master" ]]; then
    err "Not on main/master (currently on '$BRANCH'). Switch to main first."
fi

# ─── Read arguments ─────────────────────────────────────────────────────────

BUMP_TYPE="${1:-}"
[[ -n "$BUMP_TYPE" ]] || err "Usage: $0 <patch|minor|major>"

# ─── Compute new version ───────────────────────────────────────────────────

CURRENT_VERSION=$(get_current_version)
NEW_VERSION=$(bump_version "$CURRENT_VERSION" "$BUMP_TYPE")

log "Current version: $CURRENT_VERSION"
log "New version:     $NEW_VERSION"

# ─── Update all pyproject.toml files ───────────────────────────────────────

for pkg in "${PACKAGES[@]}"; do
    file="packages/${pkg}/pyproject.toml"
    [[ -f "$file" ]] || err "Missing $file"
    set_version_in_file "$file" "$NEW_VERSION"
    log "Updated $file → $NEW_VERSION"
done

# ─── Commit the version bump ───────────────────────────────────────────────

git add packages/*/pyproject.toml
git commit -m "chore: bump packages to v${NEW_VERSION}" --no-verify

# ─── Create annotated tags ─────────────────────────────────────────────────

for pkg in "${PACKAGES[@]}"; do
    tag="${pkg}/v${NEW_VERSION}"
    git tag -a "$tag" -m "Release ${pkg} v${NEW_VERSION}"
    log "Tagged $tag"
done

# ─── Push commit + tags ────────────────────────────────────────────────────

log "Pushing to remote..."
git push
git push --tags

log "Done! Released packages v${NEW_VERSION}"
log "GitHub Actions will pick up the credit-risk-data/v${NEW_VERSION} tag and create the release."
