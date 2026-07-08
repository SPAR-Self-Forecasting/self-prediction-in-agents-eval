import glob, json, os, collections

files = glob.glob('data_real/rollouts/**/r*.json', recursive=True)
# newest first, to see errors from the CURRENT run
files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

errs = collections.Counter()
samples = {}
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('errored'):
        e = (d.get('error') or '').strip()
        # collapse to a signature (first line, trimmed)
        sig = e.splitlines()[0][:160] if e else '<empty error>'
        errs[sig] += 1
        if sig not in samples:
            samples[sig] = (f, e[:1200])

print('=== error signatures (count) ===')
for sig, n in errs.most_common(12):
    print(f'{n:5d}  {sig}')

print('\n=== full text of top signature ===')
if errs:
    top = errs.most_common(1)[0][0]
    f, e = samples[top]
    print('file:', f)
    print(e)
