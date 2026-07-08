import glob, json, os

# 1. find an errored qwen3-next-80b rollout and dump its shape
target = None
for f in glob.glob('data_real/rollouts/*qwen3-next*/**/r*.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('errored'):
        target = (f, d)
        break

if not target:
    print('no errored qwen3-next rollout found')
    raise SystemExit

f, d = target
print('FILE:', f)
print('keys:', list(d.keys()))
t = d.get('trajectory') or []
print('trajectory_len:', len(t), '(0 = agent never ran; >0 = agent ran, grading failed)')
print('solved:', d.get('solved'), ' errored:', d.get('errored'))
print('error (first 700 chars):')
print((d.get('error') or '')[:700])

# 2. did the agent produce a patch? scan trajectory text for a diff
joined = json.dumps(t)[:200000]
print('\nmentions "diff --git":', 'diff --git' in joined)
print('mentions "COMPLETE_TASK":', 'COMPLETE_TASK' in joined)

# 3. find the swebench grader log for this run
run_id_hint = os.path.basename(f).replace('.json', '')  # rN
model_dir = f.split('/rollouts/')[1].split('/')[0]
print('\n=== searching swebench run_evaluation logs ===')
hits = []
for pat in ('**/run_instance.log', '**/run_evaluation/**/*.log'):
    for lg in glob.glob(pat, recursive=True):
        if 'qwen3-next' in lg:
            hits.append(lg)
print('grader logs found:', len(hits))
for lg in sorted(set(hits))[:2]:
    print('---', lg)
    try:
        txt = open(lg, errors='replace').read()
        print(txt[-1800:])
    except Exception as e:
        print('read err', e)
