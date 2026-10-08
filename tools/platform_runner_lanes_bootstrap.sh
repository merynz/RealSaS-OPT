#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REALSAS_RUNNER_REPO_URL:-https://github.com/merynz/RealSaS-OPT}"
TOKEN="${REALSAS_GITHUB_RUNNER_TOKEN:-}"
TARGET_ROOT="${REALSAS_RUNNER_LANES_ROOT:-$HOME/realsas_ci_runners}"
SOURCE_DIR="${REALSAS_RUNNER_SOURCE_DIR:-}"
LABELS="${REALSAS_CI_LIGHT_LABELS:-realsas-ci-light,ci-light,cpu}"

fail() { echo "RUNNER_LANE_BOOTSTRAP_FAIL=$*" >&2; exit 2; }

[[ -n "$TOKEN" ]] || fail "REALSAS_GITHUB_RUNNER_TOKEN_REQUIRED"
command -v rsync >/dev/null || fail "RSYNC_REQUIRED"
command -v systemctl >/dev/null || fail "SYSTEMD_REQUIRED"
systemctl --user show-environment >/dev/null 2>&1 || fail "USER_SYSTEMD_UNAVAILABLE"

mkdir -p "$TARGET_ROOT" "$HOME/.config/systemd/user"
TARGET_ROOT="$(cd "$TARGET_ROOT" && pwd)"

if [[ -z "$SOURCE_DIR" ]]; then
  mapfile -t candidates < <(find "$HOME" -maxdepth 3 -type f -name config.sh -path '*runner*' -printf '%h\n' 2>/dev/null | sort -u)
  configured=()
  for candidate in "${candidates[@]}"; do
    candidate="$(cd "$candidate" && pwd)"
    case "$candidate/" in
      "$TARGET_ROOT/"*) continue ;;
    esac
    if [[ -f "$candidate/.runner" && -x "$candidate/run.sh" ]]; then
      configured+=("$candidate")
    fi
  done
  [[ ${#configured[@]} -eq 1 ]] || fail "SOURCE_RUNNER_AMBIGUOUS count=${#configured[@]} set_REALSAS_RUNNER_SOURCE_DIR"
  SOURCE_DIR="${configured[0]}"
fi

SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
case "$SOURCE_DIR/" in
  "$TARGET_ROOT/"*) fail "SOURCE_RUNNER_MUST_NOT_BE_MANAGED_LANE=$SOURCE_DIR" ;;
esac
[[ -x "$SOURCE_DIR/config.sh" && -x "$SOURCE_DIR/run.sh" ]] || fail "SOURCE_RUNNER_INVALID=$SOURCE_DIR"

install_lane() {
  local name="$1"
  local dest="$TARGET_ROOT/$name"
  local unit="realsas-actions-${name}.service"
  local unit_path="$HOME/.config/systemd/user/$unit"

  if [[ -f "$dest/.runner" ]]; then
    echo "RUNNER_LANE_ALREADY_CONFIGURED=$name dest=$dest"
  else
    mkdir -p "$dest"
    rsync -a --delete \
      --exclude '.runner' \
      --exclude '.credentials' \
      --exclude '.credentials_rsaparams' \
      --exclude '.service' \
      --exclude '_work/' \
      --exclude '_diag/' \
      "$SOURCE_DIR/" "$dest/"
    (
      cd "$dest"
      ./config.sh --unattended \
        --url "$REPO_URL" \
        --token "$TOKEN" \
        --name "$name" \
        --labels "$LABELS" \
        --work '_work' \
        --replace
    )
  fi

  cat > "$unit_path" <<EOF
[Unit]
Description=RealSaS GitHub Actions runner $name
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$dest
ExecStart=$dest/run.sh
Restart=always
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=300
UMask=0077

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable --now "$unit"
  systemctl --user is-active --quiet "$unit" || fail "RUNNER_SERVICE_NOT_ACTIVE=$unit"
  echo "RUNNER_LANE_ACTIVE=$name labels=$LABELS service=$unit"
}

install_lane realsas-ci-light-a
install_lane realsas-ci-light-b

echo "RUNNER_LANE_BOOTSTRAP_PASS source=$SOURCE_DIR target_root=$TARGET_ROOT"
echo "NOTE=New lanes intentionally do_not_have_the_legacy_realsas_label; current workflows cannot consume them until reviewed cutover."
