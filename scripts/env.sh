#!/bin/sh
# Machine, toolchain and load before a run; bench.sh and compare.sh save it as env.txt.
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
echo "macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "cpu: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo ?)P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo ?)E)"
echo "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
system_profiler SPDisplaysDataType | grep -E 'Chipset Model|Total Number of Cores|Metal Support' | sed 's/^ */gpu /'
echo "compiler: $(c++ --version | head -1)"
echo "model: $(system_profiler SPHardwareDataType 2>/dev/null | sed -n 's/^ *Model Name: //p')"
echo "power: $(pmset -g batt | head -1 | sed "s/Now drawing from //; s/'//g"), $(pmset -g | grep -E '^ *powermode' | sed 's/^ *//; s/  */ /g')"
echo "load before: $(uptime | sed 's/.*load averages*: //')"
echo "docker containers running: $(docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
echo "top processes by CPU before the run (second sample of top -l 2):"
top -l 2 -o cpu -n 6 -stats command,cpu 2>/dev/null | tail -7 | sed 's/^/  /'
