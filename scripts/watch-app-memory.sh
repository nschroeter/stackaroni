#!/bin/zsh
#
# Sample the memory of a manually-driven `stackaroni-app` run, until it exits.
#
#   ./scripts/watch-app-memory.sh          # then launch the app
#   ./scripts/watch-app-memory.sh out.tsv  # somewhere other than target/
#
# macOS only — `footprint` is an Apple tool. There is no Linux/Windows equivalent
# wired up here because the memory work this exists for was measured on macOS and
# a second implementation would be untested guesswork.
#
# # Why footprint, and why the JSON
#
# `phys_footprint` is what Activity Monitor calls "Memory": dirty + compressed,
# excluding clean file-backed pages. That is the honest figure for this pipeline,
# because the scratch planes are mmapped files whose clean pages inflate RSS
# without being a claim on RAM — measured 16.5 GB of RSS against 5.6 GB of
# footprint after a run, a 3x difference in what you would report.
#
# Read through `-j` (JSON), never the text output: the text form switches to whole
# gigabytes above 10 GB and *rounds*, so a process holding 10.6 GiB prints "11 GB".
# That mis-stated a real measurement by enough to change its conclusion. The JSON
# carries raw **bytes** — a shell whose text output reads "2560 KB" comes through
# here as 2507112.
#
# # Why the peak comes from the kernel
#
# `phys_footprint_peak` is a kernel-tracked high-water mark, so it cannot slip
# between samples. Do not compute the peak from the sampled column: at 1 Hz that
# missed 1.1 GB of a 10.8 GB peak on a 33-frame run, because the maximum falls
# inside a stage rather than at a boundary.
#
# See the T18 row in `docs/eval-log.md` for the measurements this produced.

set -u
OUT=${1:-target/app-memory.tsv}
JSON=$(mktemp -t stackaroni-footprint)
trap 'rm -f $JSON' EXIT
mkdir -p ${OUT:h}

if [[ $(uname) != Darwin ]] || ! whence footprint >/dev/null; then
  print -ru2 -- "needs macOS and /usr/bin/footprint"
  exit 1
fi

# Only a process started *after* this script. Latching onto an instance whose
# earlier workload was not observed yields a kernel peak with no workload attached
# to it — which is exactly how the first attempt at this went wrong.
EXISTING=" $(pgrep -x stackaroni-app | tr '\n' ' ')"
print -r -- "waiting for a NEW stackaroni-app (ignoring:${EXISTING:- none})..."
PID=""
while [[ -z $PID ]]; do
  for p in $(pgrep -x stackaroni-app); do
    [[ $EXISTING == *" $p "* ]] || { PID=$p; break }
  done
  [[ -z $PID ]] && sleep 1
done
print -r -- "attached to pid $PID — quit the app to finish"

# Raw values as reported, alongside GB for reading. Note the units differ: `ps`
# gives RSS in kilobytes, `footprint -j` gives bytes. The raw columns are the
# record; the GB ones are a convenience and the only place rounding happens.
printf 't_s\trss_kb\tfootprint_b\tpeak_b\trss_gb\tfootprint_gb\tpeak_gb\n' > $OUT

T0=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  footprint -p $PID -j $JSON >/dev/null 2>&1
  CUR=$(rg -o '"phys_footprint":(\d+)' -r '$1' $JSON 2>/dev/null | tail -1)
  PEAK=$(rg -o '"phys_footprint_peak":(\d+)' -r '$1' $JSON 2>/dev/null | tail -1)
  : ${RSS:=0} ${CUR:=0} ${PEAK:=0}
  printf '%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\n' \
    $(( $(date +%s) - T0 )) $RSS $CUR $PEAK \
    $(( RSS / 1048576.0 )) $(( CUR / 1073741824.0 )) $(( PEAK / 1073741824.0 )) >> $OUT
  sleep 1
done

PEAK_B=$(awk -F'\t' 'NR>1 && $4+0 > m {m=$4} END{print m+0}' $OUT)
print -r -- "app exited after $(( $(date +%s) - T0 ))s"
printf 'peak phys_footprint: %s bytes = %.3f GB\n' $PEAK_B $(( PEAK_B / 1073741824.0 ))
print -r -- "samples in $OUT"
