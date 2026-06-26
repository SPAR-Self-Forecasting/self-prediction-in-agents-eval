import json
with open('experiments/2026-05-16_self_prediction/data/tasks.json', encoding='utf-8') as f:
    tasks = json.load(f)
from collections import Counter
diff_counter = Counter(t.get('difficulty', '?') for t in tasks)
print('Difficulty mix of the 30 sampled tasks:', dict(diff_counter))
print()
print('Task breakdown:')
for t in sorted(tasks, key=lambda x: (x.get('difficulty', 'z'), x['instance_id'])):
    print(f"  [{t.get('difficulty', '?'):6s}] {t['instance_id']:38s} repo={t['repo']}")
