import glob, json, os, statistics

files = glob.glob('data_real/rollouts/**/r*.json', recursive=True)
print('total rollout files:', len(files))
if not files:
    raise SystemExit

sizes = sorted(os.path.getsize(f) for f in files)
print('size bytes: min=%d  median=%d  max=%d' % (sizes[0], sizes[len(sizes)//2], sizes[-1]))
print('tiny(<800B):', sum(1 for s in sizes if s < 800))

readable = solved = nonempty_patch = errored = 0
sample_keys = None
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        errored += 1
        continue
    readable += 1
    if sample_keys is None:
        sample_keys = list(d.keys())
    if d.get('solved') or d.get('resolved'):
        solved += 1
    patch = d.get('patch') or d.get('model_patch') or d.get('prediction') or ''
    if isinstance(patch, str) and patch.strip():
        nonempty_patch += 1

print('readable:', readable, ' json_errors:', errored)
print('solved:', solved, ' (%.1f%%)' % (100*solved/max(readable,1)))
print('nonempty_patch:', nonempty_patch, ' (%.1f%%)' % (100*nonempty_patch/max(readable,1)))
print('sample keys:', sample_keys)
