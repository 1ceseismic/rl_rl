#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
usage: scripts/prof.sh <cmd>
  build              build-prof/ (RelWithDebInfo + frame pointers)
  sweep [N...]       sps per arena count (default: 12 24 36 48 72 96)
  nsys               gpu/cpu timeline, checks for pageable memcpy
  perf               cpu sample -> perf.data, prints top symbols
  heap               heaptrack, prints alloc counts
  c2c                false sharing (needs perf_event_paranoid <= 0)

env: DUR=90 (run seconds), N=36 (arenas), GGL_ROOT=...
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build-prof"
BINARY="$BUILD_DIR/rl_rl_ggl"
OUT="$ROOT/prof-out"
DUR="${DUR:-90}"
N="${N:-36}"
WARMUP=3

for p in "$ROOT/../GigaLearnCPP-Leak" "$HOME/code/_misc_repos/GigaLearnCPP-Leak" "$HOME/code/GigaLearnCPP-Leak"; do
    [[ -f "$p/GigaLearnCPP/CMakeLists.txt" ]] && GGL_ROOT="${GGL_ROOT:-$p}" && break
done
: "${GGL_ROOT:?gigalearn source not found, set GGL_ROOT}"

check_idle() {
    local load
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" 'BEGIN{exit !(l > 4)}'; then
        echo "load is $load, benchmarks will be noise. FORCE=1 to run anyway."
        [[ "${FORCE:-0}" == "1" ]] || exit 1
    fi
}

build() {
    [[ -x "$BINARY" ]] && return 0
    echo "[prof] building $BUILD_DIR"
    BUILD_DIR="$BUILD_DIR" BUILD_TYPE=RelWithDebInfo GGL_ROOT="$GGL_ROOT" \
        CXXFLAGS="-fno-omit-frame-pointer -g" \
        TORCH_PATH="${TORCH_PATH:-$ROOT/.venv/lib/python3.12/site-packages/torch}" \
        "$ROOT/scripts/build_ggl.sh"
}

run_env() {
    local py_home py_site
    py_home=$("$ROOT/.venv/bin/python" -c 'import sys; print(sys.base_prefix)')
    py_site=$("$ROOT/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    echo "GGL_MESH_DIR=${GGL_MESH_DIR:-$ROOT/collision_meshes}" \
         "GGL_NUM_GAMES=$1" "GGL_RUN_NAME=prof-$1" "GGL_NO_METRICS=1" \
         "PYTHONHOME=$py_home" "PYTHONPATH=$py_site"
}

# median, skips warmup
stat_key() {
    grep -F "$2: " "$1" | sed 's/.*: //; s/,//g' | tail -n +$((WARMUP + 1)) \
        | sort -g | awk '{v[NR]=$1} END {if (!NR) printf "n/a"; else if (v[int((NR+1)/2)] < 10) printf "%.4f", v[int((NR+1)/2)]; else printf "%.0f", v[int((NR+1)/2)]}'
}

# blocking stdin, no spin
launch() {
    local log=$1 n=$2 i; shift 2
    rm -rf "$BUILD_DIR/checkpoints_ggl/prof-$n"
    (cd "$BUILD_DIR" && export $(run_env "$n") && exec "$@" "$BINARY" \
        < <(sleep $((DUR + 600))) >"$log" 2>&1) &
    local pid=$!
    sleep "$DUR"
    pkill -TERM -x rl_rl_ggl || true
    for i in $(seq 1 600); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    pkill -KILL -x rl_rl_ggl 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -rf "$BUILD_DIR/checkpoints_ggl/prof-$n"
}

cmd_sweep() {
    check_idle; build
    mkdir -p "$OUT"
    local list=("$@")
    [[ ${#list[@]} -eq 0 ]] && list=(12 24 36 48 72 96)
    printf '%6s %12s %12s %12s %12s\n' arenas collect/s overall/s env_step infer
    for n in "${list[@]}"; do
        local log="$OUT/sweep-$n.log"
        launch "$log" "$n"
        printf '%6s %12s %12s %12s %12s\n' "$n" \
            "$(stat_key "$log" 'Collection Steps/Second')" \
            "$(stat_key "$log" 'Overall Steps/Second')" \
            "$(stat_key "$log" 'Env Step Time')" \
            "$(stat_key "$log" 'Inference Time')"
    done
    echo "logs: $OUT/sweep-*.log"
}

cmd_nsys() {
    check_idle; build
    mkdir -p "$OUT"
    launch "$OUT/nsys.log" "$N" nsys profile -t cuda,osrt,nvtx --sample=cpu -y 20 \
        --force-overwrite=true -o "$OUT/ggl"
    nsys stats --force-export=true --report cuda_gpu_mem_time_sum,cuda_gpu_kern_sum,cuda_api_sum \
        "$OUT/ggl.nsys-rep" 2>/dev/null | head -80
}

cmd_perf() {
    check_idle; build
    mkdir -p "$OUT"
    launch "$OUT/perf.log" "$N" perf record -F 199 --call-graph fp -o "$OUT/perf.data" --
    perf report -i "$OUT/perf.data" --stdio --no-children --sort dso,symbol 2>/dev/null \
        | grep -v '^#' | grep -v '^$' | head -60
}

cmd_heap() {
    check_idle; build
    mkdir -p "$OUT"
    rm -f "$OUT"/heap*.zst
    launch "$OUT/heap.log" "$N" heaptrack --record-only -o "$OUT/heap"
    heaptrack_print "$OUT"/heap*.zst 2>/dev/null | head -80
}

cmd_c2c() {
    check_idle; build
    [[ $(cat /proc/sys/kernel/perf_event_paranoid) -le 0 ]] || {
        echo "needs: sudo sysctl kernel.perf_event_paranoid=0"; exit 1; }
    mkdir -p "$OUT"
    launch "$OUT/c2c.log" "$N" perf c2c record -o "$OUT/c2c.data" --
    perf c2c report -i "$OUT/c2c.data" --stdio 2>/dev/null | head -60
}

case "${1:-}" in
    build) build ;;
    sweep) shift; cmd_sweep "$@" ;;
    nsys) cmd_nsys ;;
    perf) cmd_perf ;;
    heap) cmd_heap ;;
    c2c) cmd_c2c ;;
    *) usage; exit 2 ;;
esac
