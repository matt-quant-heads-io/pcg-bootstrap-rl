#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################

ENV_CFGS=(
  "configs/envs/sokoban-v0.yaml"
  "configs/envs/zelda-v0.yaml"
  "configs/envs/ddave-v0.yaml"
)

# Algorithms with their config paths (ALG:::PATH)
ALGOS=(
  "PPO:::configs/algos/ppo.yaml"
  # "SAC:::configs/algos/sac.yaml"
  # "TD3:::configs/algos/td3.yaml"
)

MODEL="CNNPolicy"
LAUNCH_LOG_ROOT="/home/ubuntu/pcg-bootstrap-rl/launch_logs"
MAX_PROCS="${MAX_PROCS:-3}"
PY="${PY:-python}"

# Optional features
DRY_RUN="${DRY_RUN:-0}"              # only print commands
GPU_RR="${GPU_RR:-0}"                # round-robin GPUs across jobs
HEARTBEAT_SEC="${HEARTBEAT_SEC:-30}" # heartbeat to progress.log
FAIL_FAST="${FAIL_FAST:-1}"          # 1 = exit if any job fails; 0 = continue

########################################
# HELPERS
########################################
have_cmd() { command -v "$1" >/dev/null 2>&1; }
timestamp() { date +"%Y%m%d_%H%M%S"; }
iso_now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
fmt_secs() { local s="$1"; printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }
list_gpus() {
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
  else
    echo ""
  fi
}

# clean shutdown
pids=()
cleanup() {
  trap - INT TERM EXIT
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

append_csv_row() {
  local csv="$1" header="$2" row="$3"
  local dir; dir="$(dirname "$csv")"
  mkdir -p "$dir"
  if have_cmd flock; then
    exec {fd}>>"$csv.lock"
    flock -x "$fd" bash -lc '
      csv="$1"; header="$2"; row="$3";
      if [[ ! -s "$csv" ]]; then printf "%s\n" "$header" >> "$csv"; fi
      printf "%s\n" "$row" >> "$csv"
    ' bash "$csv" "$header" "$row"
    exec {fd}>&-
  else
    if [[ ! -s "$csv" ]]; then printf "%s\n" "$header" >> "$csv"; fi
    printf "%s\n" "$row" >> "$csv"
  fi
}

run_one() {
  local job_id="$1" total_jobs="$2" ALG="$3" ALGO_CFG="$4" ENV_CFG="$5" GPU_ASSIGN="${6:-}"

  [[ -f "$ALGO_CFG" ]] || { echo "ERROR: algo_config not found: $ALGO_CFG" >&2; return 1; }
  [[ -f "$ENV_CFG"  ]] || { echo "ERROR: env_config not found: $ENV_CFG"  >&2; return 1; }

  local ENV_NAME; ENV_NAME="$(basename "$ENV_CFG" .yaml)"
  local STAMP;    STAMP="$(timestamp)"
  local LOG_DIR="${LAUNCH_LOG_ROOT}/${ENV_NAME}/${ALG}/${STAMP}"
  mkdir -p "$LOG_DIR"

  local CMD=( "$PY" main.py --algorithm "$ALG" --model "$MODEL" --env_config "$ENV_CFG" --algo_config "$ALGO_CFG" )

  local START_ISO; START_ISO="$(iso_now)"
  local START_S;   START_S="$(date +%s)"
  echo "[$job_id/$total_jobs] === [START] $ALG on $ENV_NAME @ $STAMP === (start: $START_ISO)"
  {
    echo "Timestamp: $STAMP"
    echo "Start (UTC): $START_ISO"
    [[ -n "$GPU_ASSIGN" ]] && echo "CUDA_VISIBLE_DEVICES=$GPU_ASSIGN"
    echo "Command:"; printf '%q ' "${CMD[@]}"; echo; echo
  } | tee "$LOG_DIR/args.txt" >/dev/null

  # Heartbeat
  local HB_FLAG="$LOG_DIR/.running"; : > "$HB_FLAG"
  (
    local t=0
    while [[ -f "$HB_FLAG" ]]; do
      printf "[%s] RUNNING: %s on %s | elapsed=%s sec\n" "$(iso_now)" "$ALG" "$ENV_NAME" "$t" >> "$LOG_DIR/progress.log"
      sleep "$HEARTBEAT_SEC" || break
      t=$((t + HEARTBEAT_SEC))
    done
  ) &
  local HB_PID=$!; pids+=("$HB_PID")

  local EXIT_CODE=0
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] skipping execution."
  else
    if [[ -n "$GPU_ASSIGN" ]]; then
      CUDA_VISIBLE_DEVICES="$GPU_ASSIGN" "${CMD[@]}" >"$LOG_DIR/stdout.txt" 2>"$LOG_DIR/stderr.txt" || EXIT_CODE=$?
    else
      "${CMD[@]}" >"$LOG_DIR/stdout.txt" 2>"$LOG_DIR/stderr.txt" || EXIT_CODE=$?
    fi
  fi

  rm -f "$HB_FLAG"; kill "$HB_PID" 2>/dev/null || true; wait "$HB_PID" 2>/dev/null || true

  local END_ISO; END_ISO="$(iso_now)"
  local END_S;   END_S="$(date +%s)"
  local DUR=$((END_S - START_S))
  printf "[%s] DONE: %s on %s | duration=%s (%ds) | exit=%d\n" "$END_ISO" "$ALG" "$ENV_NAME" "$(fmt_secs "$DUR")" "$DUR" "$EXIT_CODE" >> "$LOG_DIR/progress.log"
  echo "[$job_id/$total_jobs] === [DONE]  $ALG on $ENV_NAME @ $STAMP === (end: $END_ISO, dur: $(fmt_secs "$DUR"), exit: $EXIT_CODE)"

  local SUMMARY_CSV="${LAUNCH_LOG_ROOT}/summary.csv"
  local HEADER="env,algo,stamp,start_iso,end_iso,duration_sec,exit_code,log_dir"
  local ROW; ROW="$(printf "%s,%s,%s,%s,%s,%d,%d,%s" "$ENV_NAME" "$ALG" "$STAMP" "$START_ISO" "$END_ISO" "$DUR" "$EXIT_CODE" "$LOG_DIR")"
  append_csv_row "$SUMMARY_CSV" "$HEADER" "$ROW"

  return "$EXIT_CODE"
}

export -f run_one have_cmd timestamp iso_now fmt_secs append_csv_row list_gpus
export LAUNCH_LOG_ROOT MODEL PY DRY_RUN HEARTBEAT_SEC

########################################
# PRECHECK: ensure all configs exist
########################################
for ENV_CFG in "${ENV_CFGS[@]}"; do
  [[ -f "$ENV_CFG" ]] || { echo "ERROR: env_config not found: $ENV_CFG" >&2; exit 1; }
done
for entry in "${ALGOS[@]}"; do
  ALG="${entry%%:::*}"
  ALGO_CFG="${entry##*:::}"
  [[ -n "$ALG" && -n "$ALGO_CFG" ]] || { echo "ERROR: bad ALGOS entry: $entry" >&2; exit 1; }
  [[ -f "$ALGO_CFG" ]] || { echo "ERROR: algo_config for $ALG not found: $ALGO_CFG" >&2; exit 1; }
done
mkdir -p "$LAUNCH_LOG_ROOT"

########################################
# BUILD JOBS  (prepare a 4-column table)
########################################
# Table columns: idx<TAB>ALG<TAB>ALGO_CFG<TAB>ENV_CFG
JOB_TABLE_FILE="$(mktemp)"
idx=0
for ENV_CFG in "${ENV_CFGS[@]}"; do
  for entry in "${ALGOS[@]}"; do
    ALG="${entry%%:::*}"; ALGO_CFG="${entry##*:::}"
    idx=$((idx+1))
    printf "%d\t%s\t%s\t%s\n" "$idx" "$ALG" "$ALGO_CFG" "$ENV_CFG" >> "$JOB_TABLE_FILE"
  done
done
TOTAL_JOBS="$idx"

# GPU RR discovery (only used in bash fallback below)
GPU_LIST=(); GPU_COUNT=0
if [[ "$GPU_RR" == "1" ]]; then
  read -r -a GPU_LIST <<< "$(list_gpus)"
  GPU_COUNT=${#GPU_LIST[@]}
  if (( GPU_COUNT == 0 )); then
    echo "GPU_RR requested but no GPUs detected; continuing without GPU assignment."
    GPU_RR=0
  fi
fi

########################################
# DISPATCH
########################################
if have_cmd parallel; then
  echo "Using GNU parallel with $MAX_PROCS jobs..."
  # Directly map columns to run_one args: {1}=idx {2}=ALG {3}=ALGO_CFG {4}=ENV_CFG
  # Note: We’re not doing GPU RR here (parallel doesn’t know job index modulo GPUs cleanly).
  parallel -j "$MAX_PROCS" --colsep '\t' \
    run_one {1} "$TOTAL_JOBS" {2} {3} {4} \
    :::: "$JOB_TABLE_FILE"
else
  echo "GNU parallel not found. Using Bash background jobs with concurrency=$MAX_PROCS..."
  semaphore=$MAX_PROCS
  fail_count=0

  while IFS=$'\t' read -r j_idx ALG ALGO_CFG ENV_CFG; do
    # wait for a free slot
    while (( $(jobs -rp | wc -l) >= semaphore )); do sleep 0.5; done

    GPU_ASSIGN=""
    if [[ "$GPU_RR" == "1" && $GPU_COUNT -gt 0 ]]; then
      # j_idx is 1-based
      GPU_ASSIGN="${GPU_LIST[$(( (j_idx-1) % GPU_COUNT ))]}"
    fi

    bash -lc "run_one \"$j_idx\" \"$TOTAL_JOBS\" \"$ALG\" \"$ALGO_CFG\" \"$ENV_CFG\" \"$GPU_ASSIGN\"" &
    pids+=($!)
  done < "$JOB_TABLE_FILE"

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      if [[ "$FAIL_FAST" == "1" ]]; then
        echo "A job failed (pid $pid). Exiting due to FAIL_FAST=1."
        rm -f "$JOB_TABLE_FILE"
        exit 1
      else
        echo "A job failed (pid $pid). Continuing (FAIL_FAST=0)."
        fail_count=$((fail_count+1))
      fi
    fi
  done

  if (( fail_count > 0 )); then
    echo "Completed with $fail_count failed job(s)."
  fi
fi

rm -f "$JOB_TABLE_FILE"
echo "All distributed runs finished."
