#!/usr/bin/env python3
"""Helpers for scripts/profile.sh.

  profile.py template SRC DST K=V...  copy of an Instruments template with options K set to the
                                      integer V (xctrace has no flags for instrument options)
  profile.py counters FILE            names of the counters in an exported gpu-counter-info table
  profile.py summary DIR              per-kernel GPU time from DIR/<type>-<len>/*.xml -> stdout CSV
"""
import glob, os, plistlib, statistics, sys
import xml.etree.ElementTree as ET


def template(src, dst, settings):
    p = plistlib.load(open(src, "rb"))
    objs = p["$objects"]
    found = set()
    for o in list(objs):
        if isinstance(o, dict) and "NS.keys" in o:
            for i, k in enumerate(o["NS.keys"]):
                if objs[k.data] in settings:
                    objs.append(int(settings[objs[k.data]]))
                    o["NS.objects"][i] = plistlib.UID(len(objs) - 1)
                    found.add(objs[k.data])
    if found != set(settings):
        sys.exit(f"options {set(settings) - found} not found in {src}")
    plistlib.dump(p, open(dst, "wb"), fmt=plistlib.FMT_BINARY)


def rows(path):
    """Rows of an `xctrace export --xpath` table as dicts mnemonic -> (raw text, fmt).
    xctrace writes a repeated value once with id= and later as ref=; this resolves the refs."""
    root = ET.parse(path).getroot()
    cols = [c.findtext("mnemonic") for c in root.iter("col")]
    seen = {}
    out = []
    for r in root.iter("row"):
        d = {}
        for name, el in zip(cols, list(r)):
            if "ref" in el.attrib:
                d[name] = seen[el.attrib["ref"]]
            else:
                v = (el.text or "", el.attrib.get("fmt", ""))
                if "id" in el.attrib:
                    seen[el.attrib["id"]] = v
                for sub in el.iter():  # nested values (e.g. a pid inside a process) can be refs later
                    if "id" in sub.attrib and sub is not el:
                        seen[sub.attrib["id"]] = (sub.text or "", sub.attrib.get("fmt", ""))
                d[name] = v
        out.append(d)
    return out


def counter_names(path):
    return "; ".join(sorted({r["name"][1] for r in rows(path)})) or "none"


def summary(base):
    print("dtype,seq_len,kernel,samples,median_us,min_us,max_us,median_pct_of_kick,"
          "gpu_counters_recorded,other_process_gpu_ms,kvbench_gpu_ms")
    for d in sorted(glob.glob(os.path.join(base, "*-*/"))):
        dtype, n = os.path.basename(d.rstrip("/")).split("-")
        counters = counter_names(d + "counters.xml")
        busy = {}
        for r in rows(d + "gpu-intervals.xml"):
            who = "kvbench" if r["process"][1].startswith("kvbench") else "other"
            busy[who] = busy.get(who, 0) + int(r["duration"][0]) / 1e6
        by_kernel = {}
        for r in rows(d + "shader-timeline.xml"):
            k = r["name"][1].split(" (")[0]
            by_kernel.setdefault(k, []).append((int(r["duration"][0]) / 1e3, float(r["percent-of-kick"][0])))
        for k, v in sorted(by_kernel.items()):
            us = [a for a, _ in v]
            print(f"{dtype},{n},{k},{len(v)},{statistics.median(us):.1f},{min(us):.1f},{max(us):.1f},"
                  f"{statistics.median(p for _, p in v):.1f},{counters},"
                  f"{busy.get('other', 0):.1f},{busy.get('kvbench', 0):.1f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["template"]:
        template(sys.argv[2], sys.argv[3], dict(a.split("=") for a in sys.argv[4:]))
    elif sys.argv[1:2] == ["counters"]:
        print(counter_names(sys.argv[2]))
    elif sys.argv[1:2] == ["summary"]:
        summary(sys.argv[2])
    else:
        sys.exit(__doc__)
