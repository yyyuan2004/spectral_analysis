#!/usr/bin/env bash
# Driver for the preanalysis scripts.
#
# Usage:
#   ./run_preanalysis.sh --task A1                    # one task
#   ./run_preanalysis.sh --task all                   # all tasks (A0 is a no-op)
#   ./run_preanalysis.sh --task A3 -- --epochs 80     # forward extra args to the task
#
# A0 was dropped (no apple_id mapping available in current dataset filenames);
# selecting A0 prints a notice and exits 0.
#
# Failures in one task do not stop the others (set +e).  A summary table is
# written to outputs/preanalysis/summary.md.
set -u
set +e

TASK="all"
EXTRA_ARGS=()
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2;;
        --task=*) TASK="${1#--task=}"; shift;;
        --python) PYTHON="$2"; shift 2;;
        --) shift; EXTRA_ARGS+=("$@"); break;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0;;
        *) EXTRA_ARGS+=("$1"); shift;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PA_DIR="${SCRIPT_DIR}/scripts/preanalysis"
OUT_ROOT="${PREANALYSIS_OUTPUT_ROOT:-outputs/preanalysis}"
mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/summary.md"

if [[ ! -f "${SUMMARY}" ]]; then
    {
        echo "# Preanalysis summary"
        echo ""
        echo "| task | script | started | exit | duration | log |"
        echo "|------|--------|---------|------|----------|-----|"
    } > "${SUMMARY}"
fi

declare -A TASKS_PATH=(
    [A1]="${PA_DIR}/spectral_curves_with_separability.py"
    [A2]="${PA_DIR}/bootstrap_band_stability.py"
    [A3]="${PA_DIR}/filter_tolerance_analysis.py"
    [A4]="${PA_DIR}/error_distance_to_boundary.py"
    [A5]="${PA_DIR}/single_band_ablation.py"
)
# Order chosen to match the recommended implementation/run order:
# A1 produces sufficient_stats used by A2; A4 depends only on a trained
# checkpoint; A5/A3 are the longest training-heavy passes and go last.
ORDER=(A1 A2 A4 A5 A3)

run_one() {
    local name="$1"
    local script="${TASKS_PATH[${name}]:-}"
    local log="${OUT_ROOT}/${name}.run.log"

    if [[ "${name}" == "A0" ]]; then
        echo "[A0] skipped: per-apple grouped split is not available with the current dataset naming (no apple_id mapping)." | tee -a "${log}"
        echo "| A0 | (skipped) | - | 0 | - | ${log} |" >> "${SUMMARY}"
        return 0
    fi

    if [[ -z "${script}" || ! -f "${script}" ]]; then
        echo "[${name}] ERROR: script not found (${script:-none})" | tee -a "${log}"
        echo "| ${name} | missing | - | 1 | - | ${log} |" >> "${SUMMARY}"
        return 1
    fi

    local start_iso
    start_iso="$(date -Iseconds)"
    local t0=${SECONDS}
    echo "=== [${name}] ${script} ===" | tee "${log}"
    echo "started: ${start_iso}" | tee -a "${log}"
    echo "extra args: ${EXTRA_ARGS[*]:-}" | tee -a "${log}"
    echo "" | tee -a "${log}"

    "${PYTHON}" "${script}" "${EXTRA_ARGS[@]:-}" 2>&1 | tee -a "${log}"
    local rc=${PIPESTATUS[0]}
    local dt=$((SECONDS - t0))
    echo "" | tee -a "${log}"
    echo "exit: ${rc}  duration: ${dt}s" | tee -a "${log}"
    echo "| ${name} | ${script##*/} | ${start_iso} | ${rc} | ${dt}s | ${log} |" >> "${SUMMARY}"
    return ${rc}
}

case "${TASK}" in
    all)
        run_one A0
        for t in "${ORDER[@]}"; do run_one "${t}"; done
        ;;
    A0|A1|A2|A3|A4|A5)
        run_one "${TASK}"
        ;;
    *)
        echo "unknown task: ${TASK}" >&2
        exit 2
        ;;
esac

echo ""
echo "summary appended to ${SUMMARY}"
