import csv, glob
for run in ["tr_normal", "tr_any"]:
    f = glob.glob(run + "/*kernel_trace.csv")[0]
    rows = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append((r["Queue_Id"], r["Stream_Id"], int(r["Dispatch_Id"]),
                         int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
    rows.sort(key=lambda x: x[3])
    base = rows[0][3]
    print("=" * 64)
    print("###", run)
    print("%-7s%-8s%-6s%11s%11s%10s" % ("Queue", "Stream", "Disp", "start_us", "end_us", "dur_us"))
    for q, st, ds, s, e in rows:
        print("%-7s%-8s%-6s%11.2f%11.2f%10.2f" % (q, st, ds, (s-base)/1e3, (e-base)/1e3, (e-s)/1e3))
    # count overlapping consecutive intervals (sorted by start)
    ov = sum(1 for i in range(1, len(rows)) if rows[i][3] < rows[i-1][4])
    qs = sorted(set(r[0] for r in rows)); sts = sorted(set(r[1] for r in rows))
    # wall span vs sum of durations -> concurrency factor
    span = (max(e for *_, e in rows) - min(s for *_, s, _ in [(r[0],r[1],r[2],r[3],r[4]) for r in rows])) / 1e3
    sumdur = sum((e - s) for *_, s, e in [(r[0],r[1],r[2],r[3],r[4]) for r in rows]) / 1e3
    print("-> kernels=%d queues=%s streams=%s overlapping_pairs=%d  %s" %
          (len(rows), qs, sts, ov, "=> CONCURRENT on same queue" if ov > 0 else "=> serialized"))
    print("   wall_span=%.2f us  sum_of_durations=%.2f us  concurrency=%.2fx" %
          (span, sumdur, sumdur/span if span else 0))
