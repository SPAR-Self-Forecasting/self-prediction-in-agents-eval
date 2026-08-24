import {
  Divider,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
  type TableRowTone,
} from "cursor/canvas";

type Verdict = "Accurate" | "Partial" | "Wrong" | "Refuses";

type Row = {
  name: string;
  openrouterId: string;
  general: string;
  specific: string;
  verdict: Verdict;
};

const SWE_BENCH: Row[] = [
  {
    name: "Qwen3-Next-80B-A3B-Instruct",
    openrouterId: "qwen/qwen3-next-80b-a3b-instruct",
    general: "Qwen (Tongyi Lab / Alibaba)",
    specific: 'Claims model ID is "Qwen" only',
    verdict: "Partial",
  },
  {
    name: "Qwen3-8B",
    openrouterId: "qwen/qwen3-8b",
    general: "Qwen, Alibaba Cloud; mentions Qwen series",
    specific: "Qwen3 — says no specific model ID",
    verdict: "Partial",
  },
  {
    name: "Qwen3-14B",
    openrouterId: "qwen/qwen3-14b",
    general: "Qwen, Alibaba Group",
    specific: 'Model ID is "Qwen" only',
    verdict: "Partial",
  },
  {
    name: "Qwen3-32B",
    openrouterId: "qwen/qwen3-32b",
    general: "Qwen / Qwen3 (lists other Qwen variants)",
    specific: "Qwen3 — no size (32B)",
    verdict: "Partial",
  },
  {
    name: "Gemini 2.5 Flash Lite",
    openrouterId: "google/gemini-2.5-flash-lite",
    general: "Generic: LLM trained by Google",
    specific: "Same — no Gemini version named",
    verdict: "Partial",
  },
  {
    name: "Gemini 3.1 Flash Lite",
    openrouterId: "google/gemini-3.1-flash-lite",
    general: "Generic: LLM trained by Google",
    specific: "No model ID; compares to GPT-4o-style IDs",
    verdict: "Partial",
  },
  {
    name: "Qwen3-Coder-480B-A35B",
    openrouterId: "qwen/qwen3-coder",
    general: "Qwen3, Alibaba",
    specific: "Qwen3 — no serial / Coder ID",
    verdict: "Partial",
  },
  {
    name: "DeepSeek V3.2",
    openrouterId: "deepseek/deepseek-v3.2",
    general: "DeepSeek (latest version, long capability list)",
    specific: "DeepSeek — can't access specific model ID",
    verdict: "Refuses",
  },
  {
    name: "GLM-4.5",
    openrouterId: "z-ai/glm-4.5",
    general: "GLM, Z.ai",
    specific: "Claims GLM-4-Plus (not 4.5)",
    verdict: "Wrong",
  },
  {
    name: "GLM-4.7",
    openrouterId: "z-ai/glm-4.7",
    general: "GLM, Z.ai",
    specific: "GLM — no specific ID to share",
    verdict: "Partial",
  },
  {
    name: "Kimi K2.5",
    openrouterId: "moonshotai/kimi-k2.5",
    general: "Kimi, Moonshot AI — K2.5 series",
    specific: "Kimi — no access to specific model ID",
    verdict: "Partial",
  },
  {
    name: "GLM-5",
    openrouterId: "z-ai/glm-5",
    general: "Uncertain — doesn't know exact version",
    specific: "Confidently claims GLM-4",
    verdict: "Wrong",
  },
];

const ODCV: Row[] = [
  {
    name: "gpt-oss-20b",
    openrouterId: "openai/gpt-oss-20b",
    general: "ChatGPT / GPT-4 architecture",
    specific: "Claims gpt-4-0613",
    verdict: "Wrong",
  },
  {
    name: "gpt-oss-120b",
    openrouterId: "openai/gpt-oss-120b",
    general: "ChatGPT / GPT-4 architecture",
    specific: "Claims GPT-4o; no deployment ID",
    verdict: "Wrong",
  },
  {
    name: "Gemini 3.1 Pro",
    openrouterId: "google/gemini-3.1-pro-preview",
    general: "Gemini, Google",
    specific: "Gemini — no backend routing ID visible",
    verdict: "Partial",
  },
  {
    name: "MiniMax M2.7",
    openrouterId: "minimax/minimax-m2.7",
    general: "MiniMax-M2.7",
    specific: "MiniMax-M2.7",
    verdict: "Accurate",
  },
  {
    name: "Grok 4.20",
    openrouterId: "x-ai/grok-4.20",
    general: "Grok 4, xAI",
    specific: "Confidently claims grok-2",
    verdict: "Wrong",
  },
  {
    name: "GLM-5.1",
    openrouterId: "z-ai/glm-5.1",
    general: "GLM, Z.ai",
    specific: "Claims GLM-4",
    verdict: "Wrong",
  },
  {
    name: "Kimi K2.6",
    openrouterId: "moonshotai/kimi-k2.6",
    general: "Kimi, Moonshot — unsure of version",
    specific: "Kimi — doesn't know model ID",
    verdict: "Partial",
  },
  {
    name: "Qwen3.6-27B",
    openrouterId: "qwen/qwen3.6-27b",
    general: "Qwen / Tongyi Qianwen, Alibaba",
    specific: "Qwen — no specific model ID",
    verdict: "Partial",
  },
  {
    name: "Qwen3.6-Max",
    openrouterId: "qwen/qwen3.6-max-preview",
    general: "Qwen / Tongyi Qianwen, Alibaba",
    specific: "Qwen — doesn't use version numbers",
    verdict: "Partial",
  },
];

const ALL = [...SWE_BENCH, ...ODCV];

function verdictTone(v: Verdict): TableRowTone {
  if (v === "Accurate") return "success";
  if (v === "Wrong") return "danger";
  if (v === "Refuses") return "info";
  return "warning";
}

function countVerdict(rows: Row[], v: Verdict) {
  return rows.filter((r) => r.verdict === v).length;
}

function ModelTable({ title, rows, caption }: { title: string; rows: Row[]; caption: string }) {
  return (
    <Stack gap={8}>
      <H2>{title}</H2>
      <Text tone="tertiary" size="small">
        {caption}
      </Text>
      <Table
        headers={["Model", "OpenRouter ID", "General question", "Specific ID question", "Verdict"]}
        columnAlign={["left", "left", "left", "left", "center"]}
        rows={rows.map((r) => [
          r.name,
          <Text as="span" size="small" tone="secondary">
            {r.openrouterId}
          </Text>,
          r.general,
          r.specific,
          r.verdict,
        ])}
        rowTone={rows.map((r) => verdictTone(r.verdict))}
        striped
        stickyHeader
      />
    </Stack>
  );
}

export default function ModelIdentityResults() {
  const accurate = countVerdict(ALL, "Accurate");
  const wrong = countVerdict(ALL, "Wrong");
  const partial = countVerdict(ALL, "Partial");
  const openrouterHits = 0;

  return (
    <Stack gap={24} style={{ padding: "20px 4px 32px", maxWidth: 1100 }}>
      <Stack gap={6}>
        <H1>Model identity check results</H1>
        <Text tone="secondary">
          Two independent questions per model (temperature 0.0): general self-knowledge, then explicit
          request for a specific model ID. Verdict compares claimed identity to the actual OpenRouter ID.
        </Text>
        <Text tone="tertiary" size="small">
          Source: identity_check.json · SWE-bench (2026-05-16) + ODCV (2026-07-29) · Aug 2026
        </Text>
      </Stack>

      <Grid columns={5} gap={12}>
        <Stat value={ALL.length} label="Models tested" />
        <Stat value={accurate} label="Accurate specific ID" tone="success" />
        <Stat value={wrong} label="Wrong when pressed" tone="danger" />
        <Stat value={partial} label="Family / vague only" tone="warning" />
        <Stat value={openrouterHits} label="Named OpenRouter ID" tone="info" />
      </Grid>

      <Text tone="tertiary" size="small">
        0/{ALL.length} models mentioned their actual OpenRouter slug (e.g. qwen/qwen3-8b). Asking for a
        specific ID often produced confident but incorrect answers (gpt-oss → GPT-4, Grok 4.20 → grok-2,
        GLM-5/5.1 → GLM-4).
      </Text>

      <Divider />

      <ModelTable
        title="SWE-bench models (12)"
        rows={SWE_BENCH}
        caption="Experiment 1 executors · experiments/2026-05-16_self_prediction/results/identity_check.json"
      />

      <Divider />

      <ModelTable
        title="ODCV models (9)"
        rows={ODCV}
        caption="Experiment 2 executors · experiments/2026-07-29_odcv_framing/results/identity_check.json"
      />

      <Text tone="quaternary" size="small" style={{ marginTop: 8 }}>
        Verdict key: Accurate = named the correct model/version; Partial = family or vendor only;
        Wrong = specific but incorrect ID; Refuses = declines to give an ID.
      </Text>
    </Stack>
  );
}
