#!/bin/bash
# sim_watchdog.sh — 백그라운드 Isaac Sim/학습 프로세스 감시견.
#   예상 시간(MAX_SEC)을 넘기거나, 로그가 STALL_SEC 동안 갱신 안 되면(=멈춤/행)
#   프로세스 트리를 통째로 죽인다. GPU 낭비/무한 행 방지.
#
# 사용:
#   ./sim_watchdog.sh <PID> <MAX_SEC> <LOGFILE> [STALL_SEC=300]
# 보통 감시 대상 sim 잡과 함께 백그라운드로 띄운다:
#   nohup ./isaaclab.sh -p <script> --headless > /tmp/job.log 2>&1 & PID=$!
#   nohup ./sim_watchdog.sh $PID 1800 /tmp/job.log 300 > /tmp/job.watchdog.log 2>&1 &
#
# 종료 코드: 0=정상종료 감지, 2=시간초과 kill, 3=로그 stall kill

PID="$1"; MAX="${2:-1800}"; LOG="$3"; STALL="${4:-300}"
[ -z "$PID" ] && { echo "usage: sim_watchdog.sh <PID> <MAX_SEC> <LOG> [STALL_SEC]"; exit 1; }

start=$(date +%s)

kill_tree() {
  local p="$1"
  for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
  kill -TERM "$p" 2>/dev/null
}

while kill -0 "$PID" 2>/dev/null; do
  now=$(date +%s); el=$((now - start))
  if [ "$el" -gt "$MAX" ]; then
    echo "[watchdog] KILL(timeout): ${el}s > MAX ${MAX}s — 프로세스 트리 종료"
    kill_tree "$PID"; sleep 3; kill -9 "$PID" 2>/dev/null
    exit 2
  fi
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    lm=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
    since=$((now - lm))
    if [ "$since" -gt "$STALL" ]; then
      echo "[watchdog] KILL(stall): 로그 ${since}s 무갱신 > STALL ${STALL}s — 멈춤 판단, 종료"
      kill_tree "$PID"; sleep 3; kill -9 "$PID" 2>/dev/null
      exit 3
    fi
  fi
  sleep 15
done
echo "[watchdog] PID $PID 예산 내 정상 종료 (경과 $(( $(date +%s) - start ))s)"
exit 0
