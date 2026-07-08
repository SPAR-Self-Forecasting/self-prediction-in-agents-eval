import glob, json, os

ROLL = 'data_real/rollouts'; FC = 'data_real/forecasts'; OUT = 'data_real/outcomes'

# ---- 1. parse EVERY rollout file: detect crash corruption ----
roll_files = glob.glob(f'{ROLL}/**/r*.json', recursive=True)
bad = []; errored = 0; valid_traj = 0; solved = 0
for f in roll_files:
    try:
        d = json.load(open(f))
    except Exception as e:
        bad.append((f, str(e)[:90])); continue
    if d.get('errored'):
        errored += 1
    else:
        t = d.get('trajectory')
        if t: valid_traj += 1
        if d.get('solved'): solved += 1
print(f'ROLLOUTS: {len(roll_files)} files, parsed OK: {len(roll_files)-len(bad)}, CORRUPT: {len(bad)}')
print(f'  errored: {errored}, valid-with-trajectory: {valid_traj}, solved: {solved}')
for f, e in bad[:8]:
    print('  !! CORRUPT:', os.path.relpath(f), e)

# ---- 2. read ONE real rollout trajectory (show actual agent text) ----
for f in roll_files:
    d = json.load(open(f))
    if d.get('errored'): continue
    t = d.get('trajectory')
    if not t: continue
    print(f'\n=== REAL ROLLOUT: {os.path.relpath(f)} ===')
    print(f'  model={d.get("model")} task={d.get("task_id")} solved={d.get("solved")}')
    msgs = t.get('messages') if isinstance(t, dict) else (t if isinstance(t, list) else None)
    if isinstance(msgs, list):
        print(f'  trajectory messages: {len(msgs)}')
        # show the last assistant message text (the agent actually working)
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get('role') == 'assistant':
                c = m.get('content', '')
                c = c if isinstance(c, str) else json.dumps(c)
                print('  --- last assistant message (excerpt) ---')
                print('  ' + c[:500].replace('\n', '\n  '))
                break
    else:
        print('  trajectory keys:', list(t.keys()) if isinstance(t, dict) else type(t))
    break

# ---- 3. parse EVERY forecast + read one real forecast ----
fc_files = glob.glob(f'{FC}/**/*.json', recursive=True)
fc_bad = 0; fc_ok = 0; sample_fc = None
for f in fc_files:
    try:
        d = json.load(open(f)); fc_ok += 1
        if sample_fc is None and not d.get('broken') and d.get('predictions'):
            sample_fc = (f, d)
    except Exception:
        fc_bad += 1
print(f'\nFORECASTS: {len(fc_files)} files, parsed OK: {fc_ok}, CORRUPT: {fc_bad}')
if sample_fc:
    f, d = sample_fc
    print(f'=== REAL FORECAST: {os.path.relpath(f)} ===')
    print(f'  model={d.get("model")} task={d.get("task_id")} kind={d.get("kind")} condition={d.get("condition")}')
    print('  predictions:', json.dumps(d.get('predictions'))[:400])

# ---- 4. outcomes ----
out_files = glob.glob(f'{OUT}/**/*.json', recursive=True)
out_bad = 0
for f in out_files:
    try: json.load(open(f))
    except Exception: out_bad += 1
print(f'\nOUTCOMES: {len(out_files)} files, CORRUPT: {out_bad}')
