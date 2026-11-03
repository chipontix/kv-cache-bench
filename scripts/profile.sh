#!/bin/sh
# Profile of the paged kernel, f16 vs i8, with xctrace (Metal System Trace + shader timeline).
# Writes results/<label>/: env.txt, plain.csv (kvbench without the profiler), and per <type>-<len>:
# kvbench.csv (timing while profiled), shader-timeline.xml (GPU time per kernel dispatch, sampled),
# counters.xml (the GPU counters xctrace recorded), gpu-intervals.xml (all GPU work, every process);
# summary.csv from scripts/profile.py; counters.txt: which GPU counters this machine gives at all.
# The .trace bundles (~30 MB each) are deleted after export.
set -eu
cd "$(dirname "$0")/.."
label=${1:-int8-profile}
lens=${LENS:-512,4096,32768}
reps=${REPS:-30}
out=results/$label
traces=build/profile
mkdir -p "$out" "$traces"
./scripts/env.sh > "$out/env.txt"
xcrun xctrace version >> "$out/env.txt" 2>&1
cat "$out/env.txt"
# Load average every 10 s for the whole run: the profile is only as good as the machine was quiet.
(while :; do echo "$(date +%H:%M:%S) $(sysctl -n vm.loadavg)"; sleep 10; done) > "$out/load.txt" &
sampler=$!
trap 'kill $sampler' EXIT

run="./build/kvbench --pages=16 --no-cpu --no-naive --reps=$reps"
$run --types=f16,i8 --lens="$lens" | tee "$out/plain.csv"

mst="$(xcode-select -p)/../Applications/Instruments.app/Contents/Packages/GPU.instrdst/Contents/Templates/Metal System Trace.tracetemplate"
check_disk() {  # one .trace at a time, but stop before writing it if the disk is nearly full
    free_gb=$(df -g . | awk 'NR==2 {print $4}')
    [ "$free_gb" -ge 5 ] || { echo "only ${free_gb} GB free, stopping" >&2; exit 1; }
}
export_table() {  # trace schema file
    xcrun xctrace export --input "$1" --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$2\"]" > "$3"
}

# Which counters can be had: the Metal API's counter sets, then xctrace's GPU counter instrument
# with each counter profile the template option takes (Instruments shows them as counter sets).
{
    echo "MTLDevice.counterSets:"
    printf 'import Metal\nfor s in MTLCreateSystemDefaultDevice()!.counterSets ?? [] { print("  " + s.name + ": " + s.counters.map { $0.name }.joined(separator: ", ")) }\n' \
        > "$traces/sets.swift"
    swift "$traces/sets.swift"
    echo "xctrace Metal System Trace, GPU counters recorded per counterprofile option value:"
    for p in 0 1 2 3 4 5 6; do
        python3 scripts/profile.py template "$mst" "$traces/counters-$p.tracetemplate" \
            counterprofile=$p counterscounterprofile=$p counterprofileinternal=$p
        rm -rf "$traces/counters-$p.trace"
        check_disk
        xcrun xctrace record --template "$traces/counters-$p.tracetemplate" --output "$traces/counters-$p.trace" \
            --no-prompt --launch -- $run --types=i8 --lens=4096 --reps=3 > /dev/null
        export_table "$traces/counters-$p.trace" gpu-counter-info "$traces/counters-$p.xml"
        echo "  $p: $(python3 scripts/profile.py counters "$traces/counters-$p.xml")"
        rm -rf "$traces/counters-$p.trace"
    done
} > "$out/counters.txt"  # not a pipe: check_disk's exit has to stop the script
cat "$out/counters.txt"

tmpl=$traces/metal-shader-timeline.tracetemplate
python3 scripts/profile.py template "$mst" "$tmpl" shaderprofiler=1 countersshaderprofiler=1 shaderprofilerinternal=1
for n in $(echo "$lens" | tr , ' '); do  # f16 and i8 back to back at each length
    for t in f16 i8; do
        d=$out/$t-$n
        tr=$traces/$t-$n.trace
        mkdir -p "$d"
        rm -rf "$tr"
        check_disk
        xcrun xctrace record --template "$tmpl" --output "$tr" --no-prompt \
            --target-stdout "$d/kvbench.csv" --launch -- $run --types=$t --lens=$n
        export_table "$tr" metal-shader-profiler-intervals "$d/shader-timeline.xml"
        export_table "$tr" gpu-counter-info "$d/counters.xml"
        export_table "$tr" metal-gpu-intervals "$d/gpu-intervals.xml"
        rm -rf "$tr"  # a .trace is ~30 MB; the exports above are what is kept
    done
done
python3 scripts/profile.py summary "$out" | tee "$out/summary.csv"
echo "load after: $(uptime | sed 's/.*load averages*: //')" >> "$out/env.txt"
