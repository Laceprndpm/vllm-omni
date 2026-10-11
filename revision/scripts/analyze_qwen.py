# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Summarize saved before/after evidence without starting an Engine."""

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root")
    args = parser.parse_args()
    root = Path(args.evidence_root)
    arms, memory = {}, []
    for phase in ("before", "after"):
        directory = root / f"native-{phase}"
        cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
        traces = [json.loads(line) for line in (directory / "rank-0.jsonl").read_text().splitlines()]
        snapshots, missing_observations = [], []
        for case in cases:
            for rank in case["ranks"]:
                expected = {result["id"] for result in case["results"]}
                observed = {ref["request_id"] for ref in rank["refs"].values()}
                if expected - observed:
                    missing_observations.append({"case": case["case"], "rank": rank["rank"],
                                                 "ids": sorted(expected - observed)})
                private = [t for ref in rank["refs"].values() for t in ref["tensors"].values()
                           if t["request_private"]]
                memory.append({
                    "phase": phase, "case": case["case"], "rank": rank["rank"],
                    "allocated_bytes": rank["allocated"], "reserved_bytes": rank["reserved"],
                    "cache_entries": len(rank["cache"]), "batch_states": len(rank["batch_states"]),
                    "lora_entries": len(rank["lora"]), "observed_states": len(rank["refs"]),
                    "alive_states": sum(ref["state_alive"] for ref in rank["refs"].values()),
                    "observed_private_tensors": len(private),
                    "alive_private_tensors": sum(t["alive"] for t in private),
                })
            snapshots.append({"case": case["case"], "errors": case["errors"],
                              "batch_sizes": dict(collections.Counter(case["batch_sizes"]))})
        rounds = []
        for case in cases:
            if not case["case"].startswith("c32-"):
                continue
            ids = {r["id"] for r in case["results"]}
            steps = [r for r in traces if r["event"] == "step" and ids.intersection(r["scheduled"])]
            packing = [r["make_batch_seconds"] for r in steps if r["make_batch_calls"]]
            rounds.append({
                "case": case["case"], "seconds": case["seconds"],
                "completed": sum(not r["aborted"] for r in case["results"]),
                "aborted": sum(r["aborted"] for r in case["results"]),
                "batch_sizes": dict(collections.Counter(case["batch_sizes"])),
                "make_batch_calls": sum(r["make_batch_calls"] for r in steps),
                "make_batch_host_seconds_total": sum(packing),
                "make_batch_host_seconds_p50": statistics.median(packing),
            })
        arms[phase] = {
            "cases": snapshots, "c32_rounds": rounds,
            "missing_state_observations": missing_observations,
            "c32_seconds_median": statistics.median(r["seconds"] for r in rounds),
            "images": {r["id"]: r["sha256"] for c in cases for r in c["results"] if not r["aborted"]},
        }
    before, after = arms["before"]["images"], arms["after"]["images"]
    result = {"arms": arms, "outputs_compared": len(before),
              "image_ids_match": before.keys() == after.keys(),
              "changed_image_hashes": [rid for rid, digest in before.items() if after.get(rid) != digest],
              "c32_wall_time_median_change_percent": 100 * (
                  arms["after"]["c32_seconds_median"] / arms["before"]["c32_seconds_median"] - 1)}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "idle-memory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(memory[0]))
        writer.writeheader()
        writer.writerows(memory)
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
