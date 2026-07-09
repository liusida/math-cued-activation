#!/usr/bin/env bash
set -u
set -o pipefail

REMOTE="${REMOTE:-root@154.93.109.214}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_PATH="${REMOTE_PATH:-~/math-cued-activation/outputs/imo-answerbench-text-vllm/}"
LOCAL_PATH="${LOCAL_PATH:-outputs/imo-answerbench-text-vllm/}"
EXPECTED="${EXPECTED:-400}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
NOTIFY_MESSAGE="${NOTIFY_MESSAGE:-VibeThinker-3B Generating Finished}"
PROGRESS_NOTIFY="${PROGRESS_NOTIFY:-1}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

count_files() {
  local pattern="$1"
  find "$LOCAL_PATH" -name "$pattern" 2>/dev/null | wc -l
}

notify_done() {
  if command -v telegram-notify >/dev/null 2>&1; then
    telegram-notify "$NOTIFY_MESSAGE"
  else
    echo "[$(timestamp)] telegram-notify not found; completion message: $NOTIFY_MESSAGE"
  fi
}

notify_progress() {
  local json_count="$1"
  local txt_count="$2"
  if [ "$PROGRESS_NOTIFY" != "1" ]; then
    return
  fi
  if command -v telegram-notify >/dev/null 2>&1; then
    telegram-notify "VibeThinker-3B progress: json=${json_count}/${EXPECTED}, txt=${txt_count}/${EXPECTED}"
  fi
}

mkdir -p "$LOCAL_PATH"

echo "[$(timestamp)] syncing vLLM outputs until complete"
echo "remote: ${REMOTE}:${REMOTE_PATH}"
echo "local:  ${LOCAL_PATH}"
echo "expecting at least ${EXPECTED} .json and ${EXPECTED} .txt files"
echo "interval: ${INTERVAL_SECONDS}s"

while true; do
  echo "[$(timestamp)] rsync start"
  if rsync -avP -e "ssh -p ${SSH_PORT}" "${REMOTE}:${REMOTE_PATH}" "$LOCAL_PATH"; then
    json_count="$(count_files '*.json')"
    txt_count="$(count_files '*.txt')"
    echo "[$(timestamp)] counts: json=${json_count}/${EXPECTED} txt=${txt_count}/${EXPECTED}"
    notify_progress "$json_count" "$txt_count"

    if [ "$json_count" -ge "$EXPECTED" ] && [ "$txt_count" -ge "$EXPECTED" ]; then
      echo "[$(timestamp)] complete"
      notify_done
      exit 0
    fi
  else
    status=$?
    echo "[$(timestamp)] rsync failed with exit code ${status}; retrying after sleep"
  fi

  sleep "$INTERVAL_SECONDS"
done
