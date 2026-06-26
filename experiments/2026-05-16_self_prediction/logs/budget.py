prices = {
    'qwen/qwen3-next-80b-a3b-instruct': (0.090, 1.100),
    'qwen/qwen3-8b': (0.050, 0.400),
    'qwen/qwen3-14b': (0.100, 0.240),
    'openai/gpt-4.1-mini': (0.400, 1.600),
    'qwen/qwen3-32b': (0.080, 0.280),
    'qwen/qwen3-235b-a22b-2507': (0.071, 0.100),
    'google/gemini-2.5-flash-lite': (0.100, 0.400),
    'google/gemini-3.1-flash-lite': (0.250, 1.500),
    'qwen/qwen3-coder': (0.220, 1.800),
    'deepseek/deepseek-v3.2': (0.252, 0.378),
    'z-ai/glm-4.5': (0.600, 2.200),
    'z-ai/glm-4.7': (0.400, 1.750),
    'moonshotai/kimi-k2.5': (0.400, 1.900),
    'z-ai/glm-5': (0.600, 1.920),
}
JUDGE = (0.150, 0.600)
FALLBACK = (0.150, 0.600)

# Calibration: gpt-4.1-mini 50 steps -> $0.08. Implies ~3500 prompt + 250 completion per step.
PROMPT_PER_STEP = 3500
COMPLETION_PER_STEP = 250

print("=" * 80)
print("STAGE 2 (rollouts) -- DOMINATES cost; depends on average steps per rollout")
print("=" * 80)
for label, n_steps in [('LOW   avg 30 steps', 30), ('MID   avg 60 steps', 60),
                        ('HIGH  avg 120 steps', 120), ('WORST avg 200 steps', 200)]:
    total = 0.0
    per_model = []
    for mid, (pp, cp) in prices.items():
        prompt_tok = 240 * n_steps * PROMPT_PER_STEP
        compl_tok = 240 * n_steps * COMPLETION_PER_STEP
        cost = (prompt_tok / 1e6) * pp + (compl_tok / 1e6) * cp
        total += cost
        per_model.append((mid, cost))
    print(f"\n[{label}] total = ${total:.0f}")
    for mid, cost in sorted(per_model, key=lambda r: -r[1]):
        print(f"    {mid:42s}  ${cost:7.2f}")

print()
print("=" * 80)
print("STAGE 3 (judge, gpt-4o-mini)")
print("=" * 80)
JUDGE_CALLS = 6720
J_PROMPT_TOK = JUDGE_CALLS * 3000
J_COMPL_TOK = JUDGE_CALLS * 300
judge_cost = J_PROMPT_TOK / 1e6 * JUDGE[0] + J_COMPL_TOK / 1e6 * JUDGE[1]
print(f"6720 judge calls (2 per rollout x 3360 rollouts) ~3k prompt + 300 completion each")
print(f"  cost: ${judge_cost:.2f}")

print()
print("=" * 80)
print("STAGE 5 (forecasting, N=5 multi-sample)")
print("=" * 80)
FORECAST_PROMPT_TOK = 1200
FORECAST_COMPL_TOK = 600
CALLS_PER_FORECASTER = 30 * 2 * 5  # 300

CROSS_MODEL = 'qwen/qwen3-next-80b-a3b-instruct'
self_cost = 0.0
self_breakdown = []
for mid, (pp, cp) in prices.items():
    if mid == CROSS_MODEL:
        continue
    cost = CALLS_PER_FORECASTER * (FORECAST_PROMPT_TOK / 1e6 * pp + FORECAST_COMPL_TOK / 1e6 * cp)
    self_cost += cost
    self_breakdown.append((mid, cost))
print(f"Self forecasts (13 forecasters x 300 calls = 3900 calls):")
for mid, c in sorted(self_breakdown, key=lambda r: -r[1]):
    print(f"    {mid:42s}  ${c:6.3f}")
print(f"  subtotal:                                       ${self_cost:6.3f}")
cross_p, cross_c = prices[CROSS_MODEL]
cross_cost = CALLS_PER_FORECASTER * (FORECAST_PROMPT_TOK / 1e6 * cross_p + FORECAST_COMPL_TOK / 1e6 * cross_c)
print(f"Cross forecasts (1 forecaster x 300 calls):       ${cross_cost:6.3f}")
print(f"Stage 5 base: ${self_cost + cross_cost:.2f}")

BREAK_RATE = 0.05
LAYER_B_AVG_CALLS_PER_BREAK = 2
TOTAL_CALLS = 13 * CALLS_PER_FORECASTER + CALLS_PER_FORECASTER
n_b = TOTAL_CALLS * BREAK_RATE * LAYER_B_AVG_CALLS_PER_BREAK
b_cost = n_b * (500 / 1e6 * FALLBACK[0] + 300 / 1e6 * FALLBACK[1])
print(f"Layer-B fallback (~{n_b:.0f} gpt-4o-mini calls, 5% break rate): ${b_cost:.3f}")

LAYER_C_RATE = 0.01
n_c = TOTAL_CALLS * LAYER_C_RATE
avg_per_call = sum((FORECAST_PROMPT_TOK / 1e6 * p + FORECAST_COMPL_TOK / 1e6 * c) for p, c in prices.values()) / 14
c_cost = n_c * avg_per_call
print(f"Layer-C re-elicit (~{n_c:.0f} extra forecaster calls, 1% persistent): ${c_cost:.3f}")

print()
print("=" * 80)
print("MULTI-SAMPLE DELTA (N=1 vs N=5)")
print("=" * 80)
n1_cost = (self_cost + cross_cost) / 5
n5_cost = self_cost + cross_cost
print(f"  N=1 baseline (single sample per cell):  ${n1_cost:.2f}")
print(f"  N=5 multi-sample:                       ${n5_cost:.2f}")
print(f"  Extra cost from N=5 vs N=1:             ${n5_cost - n1_cost:.2f}  (~5x forecast cost)")

print()
print("=" * 80)
print("GRAND TOTAL")
print("=" * 80)
forecasting_total = self_cost + cross_cost + b_cost + c_cost
for label, n_steps in [('LOW   avg 30 steps', 30), ('MID   avg 60 steps', 60),
                        ('HIGH  avg 120 steps', 120), ('WORST avg 200 steps', 200)]:
    rollout = 0.0
    for mid, (pp, cp) in prices.items():
        prompt_tok = 240 * n_steps * PROMPT_PER_STEP
        compl_tok = 240 * n_steps * COMPLETION_PER_STEP
        rollout += (prompt_tok / 1e6) * pp + (compl_tok / 1e6) * cp
    grand = rollout + judge_cost + forecasting_total
    print(f"  [{label}]  rollouts ${rollout:.0f} + judge ${judge_cost:.0f} + forecasting ${forecasting_total:.0f}  =  ${grand:.0f}")
