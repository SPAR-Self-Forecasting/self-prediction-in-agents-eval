import glob, json, collections

cell_err = collections.Counter()
cell_tot = collections.Counter()
for f in glob.glob('data_real/rollouts/**/r*.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    key = (d.get('model'), d.get('task_id'))
    cell_tot[key] += 1
    if d.get('errored'):
        cell_err[key] += 1

total = sum(cell_tot.values())
errored = sum(cell_err.values())
print(f'rollouts done: {total}   errored: {errored} ({100*errored/max(total,1):.1f}%)')
print(f'(model,task) cells touched: {len(cell_tot)}')

dist = collections.Counter(cell_err.values())
print('errored-rollouts-per-cell distribution:')
for n in sorted(dist):
    print(f'  {n} errored: {dist[n]} cells')
at_risk = sum(1 for v in cell_err.values() if v > 3)  # >3 of 8 errored = below the 5/8 valid gate
print(f'cells with >3 errored (would drop below the 5/8 gate even if NOT retried): {at_risk}')
print('worst 5 cells:', cell_err.most_common(5))
