import glob, json, collections
tot = collections.Counter(); err = collections.Counter()
for f in glob.glob('data_real/rollouts/**/r*.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    m = d.get('model', '?'); tot[m] += 1
    if d.get('errored'):
        err[m] += 1
print('total rollouts:', sum(tot.values()), ' models seen:', len(tot))
print('  total / errored  model')
for m in sorted(tot, key=lambda x: -tot[x]):
    print(f'  {tot[m]:4d} / {err[m]:3d}   {m}')
