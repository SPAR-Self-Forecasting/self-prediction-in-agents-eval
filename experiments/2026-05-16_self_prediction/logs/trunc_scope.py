import glob, json
import trajectory

MAX = 120_000
files = glob.glob('data_real/rollouts/**/r*.json', recursive=True)
n = patch_trunc = reas_trunc = either = skipped = 0
maxp = maxr = 0
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    n += 1
    if d.get('errored') or not d.get('trajectory'):
        skipped += 1
        continue
    try:
        p = trajectory.extract_final_patch(d['trajectory']) or ''
        r = trajectory.concatenated_reasoning(d['trajectory']) or ''
    except Exception:
        skipped += 1
        continue
    lp, lr = len(p), len(r)
    maxp, maxr = max(maxp, lp), max(maxr, lr)
    pt, rt = lp > MAX, lr > MAX
    patch_trunc += pt
    reas_trunc += rt
    either += (pt or rt)

print(f'rollouts scanned: {n}  (errored/no-traj skipped: {skipped})')
print(f'patch > 120k chars (doc/defensive_code judged on truncated): {patch_trunc}')
print(f'reasoning > 120k chars (uncertainty judged on truncated): {reas_trunc}')
print(f'AFFECTED rollouts (either truncated): {either}  = {100*either/max(n-skipped,1):.2f}% of valid')
print(f'largest patch: {maxp:,} chars | largest reasoning: {maxr:,} chars')
