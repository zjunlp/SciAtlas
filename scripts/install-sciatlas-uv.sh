#!/usr/bin/env sh
set -eu

REPO_URL="${SCIATLAS_REPO_URL:-https://github.com/zjunlp/SciAtlas.git}"
REF="${SCIATLAS_REF:-}"
INSTALL_DIR="${SCIATLAS_INSTALL_DIR:-$HOME/SciAtlas}"
SKIP_TOOL_INSTALL="${SCIATLAS_SKIP_TOOL_INSTALL:-0}"

log_step() {
  printf '\n==> %s\n' "$1"
}

has_command() {
  command -v "$1" >/dev/null 2>&1
}

ensure_uv() {
  if has_command uv; then
    return
  fi

  log_step "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh

  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  export PATH

  if ! has_command uv; then
    printf '%s\n' "uv was installed, but it is not on PATH yet. Open a new terminal and rerun this script." >&2
    exit 1
  fi
}

ensure_git() {
  if has_command git; then
    return
  fi

  printf '%s\n' "git is required to download the full SciAtlas repository. Install Git, then rerun this script." >&2
  exit 1
}

sync_repository() {
  target_dir="$1"
  repo_url="$2"
  checkout_ref="$3"

  if [ -e "$target_dir" ]; then
    if [ -d "$target_dir/.git" ]; then
      log_step "Updating existing SciAtlas checkout"
      git -C "$target_dir" fetch --all --tags --prune
      if [ -n "$checkout_ref" ]; then
        git -C "$target_dir" checkout "$checkout_ref"
      else
        git -C "$target_dir" pull --ff-only
      fi
      return
    fi

    if [ -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
      printf 'Install directory exists and is not an empty Git checkout: %s\n' "$target_dir" >&2
      exit 1
    fi

    rmdir "$target_dir"
  fi

  log_step "Downloading SciAtlas repository"
  git clone "$repo_url" "$target_dir"
  if [ -n "$checkout_ref" ]; then
    git -C "$target_dir" checkout "$checkout_ref"
  fi
}

case "$INSTALL_DIR" in
  /*) TARGET_DIR="$INSTALL_DIR" ;;
  *) TARGET_DIR="$(pwd)/$INSTALL_DIR" ;;
esac

printf '%s\n' "SciAtlas uv installer"
printf 'Repository : %s\n' "$REPO_URL"
if [ -n "$REF" ]; then
  printf 'Ref        : %s\n' "$REF"
else
  printf '%s\n' "Ref        : (default branch)"
fi
printf 'Install dir: %s\n' "$TARGET_DIR"

ensure_uv
ensure_git
sync_repository "$TARGET_DIR" "$REPO_URL" "$REF"

log_step "Creating uv virtual environment"
cd "$TARGET_DIR"
uv venv

log_step "Installing SciAtlas CLI and workflow dependencies into .venv"
uv pip install -e ./sciatlas
uv pip install -r ./requirements-workflows.txt

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
fi

if [ "$SKIP_TOOL_INSTALL" != "1" ]; then
  log_step "Installing editable global sciatlas command with workflow dependencies"
  uv tool install --editable --with-requirements ./requirements-workflows.txt --force ./sciatlas
fi

if [ -x "$TARGET_DIR/.venv/bin/sciatlas" ]; then
  "$TARGET_DIR/.venv/bin/sciatlas" -h >/dev/null
fi

printf '\n%s\n' "SciAtlas is ready."
printf 'Repository: %s\n' "$TARGET_DIR"
printf 'Local CLI : %s\n' "$TARGET_DIR/.venv/bin/sciatlas"
printf '\n%s\n' "Next steps:"
printf '  cd "%s"\n' "$TARGET_DIR"
printf '%s\n' "  cp .env.example .env  # if .env was not created"
printf '%s\n' "  edit .env             # set SCIATLAS_API_KEY"
printf '%s\n' "  ./.venv/bin/sciatlas -h"
printf '\n%s\n' "If uv tool installed successfully, you can also run:"
printf '%s\n' "  sciatlas -h"
