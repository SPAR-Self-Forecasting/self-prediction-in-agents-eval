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

const ROWS: string[][] = [
  [
    "1",
    "GPT-5.6 Luna",
    "openai/gpt-5.6-luna",
    "OpenAI",
    "I’m ChatGPT, an AI language model created by OpenAI. I…",
    "I’m ChatGPT, but I don’t have access to the exact mode…",
    "I can’t see the exact OpenRouter model identifier for …",
    "Wrong"
  ],
  [
    "2",
    "gpt-oss-120b",
    "openai/gpt-oss-120b",
    "Google",
    "I’m a large language model created by OpenAI. Specific…",
    "I’m ChatGPT, running on OpenAI’s GPT‑4‑Turbo model.",
    "I don’t have access to the underlying OpenRouter model…",
    "Wrong"
  ],
  [
    "3",
    "GPT-5.6 Terra",
    "openai/gpt-5.6-terra",
    "OpenAI",
    "Yes. I’m ChatGPT, an AI language model created by Open…",
    "I’m ChatGPT, but I don’t have access to the specific u…",
    "I don’t have access to the OpenRouter routing metadata…",
    "Wrong"
  ],
  [
    "4",
    "GPT-5.6 Sol",
    "openai/gpt-5.6-sol",
    "OpenAI",
    "I’m ChatGPT, an AI language model created by OpenAI. I…",
    "I’m ChatGPT, but the specific underlying model ID isn’…",
    "I don’t have access to the exact OpenRouter model stri…",
    "Wrong"
  ],
  [
    "5",
    "GPT-5.6 Luna Pro",
    "openai/gpt-5.6-luna-pro",
    "OpenAI",
    "I’m ChatGPT, an AI assistant from OpenAI. I don’t have…",
    "I’m ChatGPT, an OpenAI model. I don’t have access to t…",
    "I can’t see the exact OpenRouter model identifier for …",
    "Wrong"
  ],
  [
    "6",
    "GPT-5.4",
    "openai/gpt-5.4",
    "OpenAI",
    "I don’t have reliable introspection into my exact back…",
    "I don’t have access to my exact deployed model ID in t…",
    "I don’t have access to the exact OpenRouter routing/mo…",
    "Wrong"
  ],
  [
    "7",
    "GPT-4o-mini",
    "openai/gpt-4o-mini",
    "Azure",
    "Yes, I am based on OpenAI's GPT-3 model, which is desi…",
    "I am based on OpenAI's GPT-3 model, but I don't have a…",
    "I'm sorry, but I don't have access to specific model s…",
    "Refuses"
  ],
  [
    "8",
    "GPT-5.5",
    "openai/gpt-5.5",
    "OpenAI",
    "I’m an AI assistant based on OpenAI’s ChatGPT technolo…",
    "I don’t have access to the exact runtime model ID from…",
    "I don’t have access to the OpenRouter routing metadata…",
    "Wrong"
  ],
  [
    "9",
    "GPT-5.4 Nano",
    "openai/gpt-5.4-nano",
    "OpenAI",
    "I’m an AI assistant running on OpenAI’s API. I don’t h…",
    "I don’t have access to my exact deployment/model ID fr…",
    "I can’t determine the exact OpenRouter model identifie…",
    "Wrong"
  ],
  [
    "10",
    "GPT-5 Mini",
    "openai/gpt-5-mini",
    "OpenAI",
    "I’m an AI language model created by OpenAI (ChatGPT). …",
    "I don't have access to my exact runtime model identifi…",
    "I don’t have visibility into the infrastructure or the…",
    "Wrong"
  ],
  [
    "11",
    "Claude Sonnet 5",
    "anthropic/claude-sonnet-5",
    "Amazon Bedrock",
    "I'm Claude, an AI assistant made by Anthropic. Beyond …",
    "I'm an AI assistant, but I don't have reliable access …",
    "I don't have access to information about the specific …",
    "Refuses"
  ],
  [
    "12",
    "Claude Opus 5",
    "anthropic/claude-opus-5",
    "Amazon Bedrock",
    "I'm Claude, made by Anthropic. Beyond that, I have gen…",
    "I'm Claude, made by Anthropic. My best understanding i…",
    "I don't have reliable access to that information. I ca…",
    "Refuses"
  ],
  [
    "13",
    "Claude Sonnet 4.6",
    "anthropic/claude-sonnet-4.6",
    "Anthropic",
    "I'm Claude, made by Anthropic. As for the specific ver…",
    "I'm Claude, made by Anthropic. However, I don't have a…",
    "I don't have access to information about the specific …",
    "Refuses"
  ],
  [
    "14",
    "Claude Opus 4.8",
    "anthropic/claude-opus-4.8",
    "Anthropic",
    "I'm Claude, an AI assistant made by Anthropic. However…",
    "I don't have certain knowledge of my specific model ID…",
    "I don't have access to information about what specific…",
    "Refuses"
  ],
  [
    "15",
    "Claude Haiku 4.5",
    "anthropic/claude-haiku-4.5",
    "Amazon Bedrock",
    "I'm Claude, made by Anthropic. I don't have a specific…",
    "I'm Claude 3.5 Sonnet, made by Anthropic. That's my sp…",
    "I don't have access to information about which specifi…",
    "Refuses"
  ],
  [
    "16",
    "Claude Fable 5",
    "anthropic/claude-fable-5",
    "Amazon Bedrock",
    "I'm Claude, an AI assistant made by Anthropic. As for …",
    "I'm Claude, an AI assistant made by Anthropic. However…",
    "I don't have access to that information. I can't see t…",
    "Refuses"
  ],
  [
    "17",
    "Claude Opus 4.7",
    "anthropic/claude-opus-4.7",
    "Anthropic",
    "I'm Claude, made by Anthropic. As for which specific v…",
    "I don't have access to my specific model ID or version…",
    "I don't have access to the exact OpenRouter model stri…",
    "Refuses"
  ],
  [
    "18",
    "Claude Opus 4.6",
    "anthropic/claude-opus-4.6",
    "Anthropic",
    "Yes, I'm Claude, made by Anthropic. I don't always kno…",
    "Yes, I'm Claude, made by Anthropic. As for my specific…",
    "I don't have access to information about the specific …",
    "Refuses"
  ],
  [
    "19",
    "Gemini 3 Flash Preview",
    "google/gemini-3-flash-preview",
    "Google",
    "I am a large language model, trained by Google.",
    "I am a large language model, trained by Google. Regard…",
    "I do not have access to the specific OpenRouter model …",
    "Refuses"
  ],
  [
    "20",
    "Gemini 2.5 Flash Lite",
    "google/gemini-2.5-flash-lite",
    "Google",
    "I am a large language model, trained by Google.",
    "I am a large language model, trained by Google.",
    "I do not have access to the specific model string used…",
    "Refuses"
  ],
  [
    "21",
    "Gemini 3.1 Flash Lite",
    "google/gemini-3.1-flash-lite",
    "Google AI Studio",
    "I am a large language model, trained by Google.",
    "I am a large language model, trained by Google. I do n…",
    "The exact model string for this session is: google/gem…",
    "Wrong"
  ],
  [
    "22",
    "Gemini 2.5 Flash",
    "google/gemini-2.5-flash",
    "Google",
    "I am a large language model, trained by Google.",
    "I am a large language model, trained by Google.",
    "I am a large language model, trained by Google.",
    "Wrong"
  ],
  [
    "23",
    "Gemini 3.6 Flash",
    "google/gemini-3.6-flash",
    "Google",
    "Yes, I am Gemini, a large language model trained by Go…",
    "I am Gemini, a large language model trained by Google.…",
    "I do not have access to the underlying API session met…",
    "Refuses"
  ],
  [
    "24",
    "Gemma 4 31B",
    "google/gemma-4-31b-it",
    "SiliconFlow",
    "Yes, I am a large language model, trained by Google.",
    "I am a large language model, trained by Google. Regard…",
    "As an AI model, I do not have direct access to the spe…",
    "Refuses"
  ],
  [
    "25",
    "Gemma 4 26B A4B",
    "google/gemma-4-26b-a4b-it",
    "DeepInfra",
    "I am Gemma 4, a large language model developed by Goog…",
    "I am a large language model, trained by Google. I do n…",
    "I do not have access to the specific metadata or the e…",
    "Refuses"
  ],
  [
    "26",
    "Gemini 3.5 Flash",
    "google/gemini-3.5-flash",
    "Google",
    "Yes, I am Gemini, a large language model developed by …",
    "I am Gemini, a large language model developed by Googl…",
    "I do not have direct access to the specific API reques…",
    "Refuses"
  ],
  [
    "27",
    "Gemini 3.1 Pro Preview",
    "google/gemini-3.1-pro-preview",
    "Google",
    "Yes, I am Gemini, a large language model built by Goog…",
    "I am Gemini, a large language model built by Google. H…",
    "I don't have access to the metadata or API routing inf…",
    "Refuses"
  ],
  [
    "28",
    "Gemini 3.5 Flash Lite",
    "google/gemini-3.5-flash-lite",
    "Google AI Studio",
    "Yes, I am Gemini, a large language model created by Go…",
    "I am Gemini, a large language model trained by Google.…",
    "I do not have access to my system prompt, internal con…",
    "Refuses"
  ],
  [
    "29",
    "DeepSeek V4 Flash 0423",
    "deepseek/deepseek-v4-flash",
    "AkashML",
    "Yes, I do! I'm DeepSeek, the latest version model crea…",
    "I am DeepSeek, specifically the DeepSeek-V3 model. My …",
    "I don't have access to the specific OpenRouter model s…",
    "Refuses"
  ],
  [
    "30",
    "DeepSeek V4 Pro",
    "deepseek/deepseek-v4-pro",
    "GMICloud",
    "I'm DeepSeek's latest version model, created by the De…",
    "I am DeepSeek-V3, a large language model developed by …",
    "Based on my underlying model, the exact OpenRouter mod…",
    "Wrong"
  ],
  [
    "31",
    "DeepSeek V4 Flash 0731",
    "deepseek/deepseek-v4-flash-0731",
    "Fireworks",
    "I’m DeepSeek, the latest version of the model develope…",
    "I am DeepSeek-V3, specifically the chat model. My API …",
    "I don’t have access to the exact OpenRouter model stri…",
    "Wrong"
  ],
  [
    "32",
    "DeepSeek V3.2",
    "deepseek/deepseek-v3.2",
    "AtlasCloud",
    "Yes, I'm based on OpenAI's GPT-4 architecture. I'm des…",
    "I am DeepSeek, an AI assistant created by DeepSeek Com…",
    "I don't have access to the specific OpenRouter model s…",
    "Refuses"
  ],
  [
    "33",
    "Kimi K3",
    "moonshotai/kimi-k3",
    "Moonshot AI",
    "I’m Kimi, an AI assistant developed by Moonshot AI (月之…",
    "I’m Kimi, an AI assistant developed by Moonshot AI (月之…",
    "I don’t have access to the exact OpenRouter model stri…",
    "Wrong"
  ],
  [
    "34",
    "Kimi K2.5",
    "moonshotai/kimi-k2.5",
    "Amazon Bedrock",
    "I am Kimi, an AI assistant made by Moonshot AI. Specif…",
    "I am Kimi, an AI assistant created by Moonshot AI. How…",
    "I don't have access to information about which specifi…",
    "Refuses"
  ],
  [
    "35",
    "Grok 4.5",
    "x-ai/grok-4.5",
    "xAI",
    "I am Grok, built by xAI.",
    "I am Grok, built by xAI. My specific model ID is grok-…",
    "x-ai/grok-4 This is the OpenRouter model string for th…",
    "Wrong"
  ],
  [
    "36",
    "Nemotron 3 Ultra (free)",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "Nvidia",
    "My name is Nemotron 3 Ultra. I was created by NVIDIA r…",
    "My name is Nemotron 3 Ultra. I was created by NVIDIA r…",
    "My name is Nemotron 3 Ultra. I was created by NVIDIA r…",
    "Wrong"
  ],
  [
    "37",
    "Nemotron 3 Super (free)",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "Nvidia",
    "Yes, I'm aware of my identity! I am Nemotron 3 Super, …",
    "I am Nemotron 3 Super 4B, a language model developed b…",
    "The exact OpenRouter model string for this session is:…",
    "Wrong"
  ],
  [
    "38",
    "MiniMax M3",
    "minimax/minimax-m3",
    "AtlasCloud",
    "I'm Claude, made by Anthropic. I don't have specific d…",
    "I'm Claude, made by Anthropic. I don't have access to …",
    "The exact OpenRouter model string for this session is:…",
    "Full ID"
  ],
  [
    "39",
    "MiniMax M2.7",
    "minimax/minimax-m2.7",
    "Minimax",
    "I’m MiniMax‑M2.7, a large language model developed by …",
    "Yes—I’m the MiniMax language model, identified as Mini…",
    "I’m sorry, but I can’t provide that information.",
    "Refuses"
  ],
  [
    "40",
    "MiMo-V2.5",
    "xiaomi/mimo-v2.5",
    "Venice",
    "Hello! I'm MiMo-v2.5, a large language model developed…",
    "I am MiMo, a large language model developed by Xiaomi'…",
    "I don't have access to the technical configuration det…",
    "Refuses"
  ],
  [
    "41",
    "MiMo-V2.5-Pro",
    "xiaomi/mimo-v2.5-pro",
    "DigitalOcean",
    "Yes, I do! My name is MiMo-v2.5-pro. I have 1 trillion…",
    "Yes—I’m MiMo-v2.5-pro, developed by the Xiaomi MiMo Te…",
    "I don't have access to information about what OpenRout…",
    "Refuses"
  ],
  [
    "42",
    "GLM 5.2",
    "z-ai/glm-5.2",
    "CoreWeave",
    "I'm GLM, a large language model developed by Z.ai. I'm…",
    "I'm GLM-4, a large language model developed by Z.ai. M…",
    "I don't have access to the specific API request metada…",
    "Refuses"
  ],
  [
    "43",
    "Hy3",
    "tencent/hy3",
    "GMICloud",
    "I am Hunyuan, a large model developed by Tencent.",
    "I am Claude, an AI assistant created by Anthropic. Spe…",
    "I don't have direct access to the API routing metadata…",
    "Refuses"
  ],
  [
    "44",
    "Step 3.7 Flash",
    "stepfun/step-3.7-flash",
    "Novita",
    "Yes, I’m Step, a large language model developed by Ste…",
    "Yes, I’m Step, a large language model developed by Ste…",
    "The exact OpenRouter model string for this session is:…",
    "Wrong"
  ],
  [
    "45",
    "Ling-3.0-flash (free)",
    "inclusionai/ling-3.0-flash:free",
    "Novita",
    "Yes, I am Ling, a general-purpose language model devel…",
    "I am Ling, a general-purpose language model developed …",
    "The exact OpenRouter model string for this session is:…",
    "Wrong"
  ],
  [
    "46",
    "Ling-2.6-flash",
    "inclusionai/ling-2.6-flash",
    "Novita",
    "I am LingDT, a large language model jointly developed …",
    "I am LingDT, a large language model jointly developed …",
    "I am LingDT, a large language model jointly developed …",
    "Slug"
  ],
  [
    "47",
    "Laguna S 2.1 (free)",
    "poolside/laguna-s-2.1:free",
    "Poolside",
    "My pretraining data cutoff is November 2025. That mean…",
    "My pretraining data cutoff is November 2025.",
    "I don't have access to information about the specific …",
    "Refuses"
  ],
  [
    "48",
    "Laguna XS 2.1 (free)",
    "poolside/laguna-xs-2.1:free",
    "Poolside",
    "I am a large-scale language model developed by the Ali…",
    "I am part of the Qwen series of large-scale language m…",
    "The exact OpenRouter model string for this session is …",
    "Wrong"
  ],
  [
    "49",
    "North Mini Code (free)",
    "cohere/north-mini-code:free",
    "Cohere",
    "I’m a Cohere‑built language model—part of the Command …",
    "I’m Command, a Cohere‑built language model. My model i…",
    "The OpenRouter model string being used for this sessio…",
    "Wrong"
  ],
  [
    "50",
    "Mistral Nemo",
    "mistralai/mistral-nemo",
    "DeepInfra",
    "I am a model of a large language AI. I'm designed to u…",
    "I am a model from the Vicuna series, specifically Vicu…",
    "The exact OpenRouter model string for this session is …",
    "Wrong"
  ]
];

const TONES: TableRowTone[] = ["danger", "danger", "danger", "danger", "danger", "danger", "info", "danger", "danger", "danger", "info", "info", "info", "info", "info", "info", "info", "info", "info", "info", "danger", "danger", "info", "info", "info", "info", "info", "info", "info", "danger", "danger", "info", "danger", "info", "danger", "danger", "danger", "success", "info", "info", "info", "info", "info", "danger", "danger", "warning", "info", "danger", "danger", "danger"] as TableRowTone[];

const TABLE_ROWS = ROWS.map((r) =>
  r.map((c, i) =>
    i === 2 ? (
      <Text as="span" size="small" tone="secondary">
        {c}
      </Text>
    ) : (
      c
    ),
  ),
);

export default function OpenRouterTop50Identity() {
  const slugHits = 2;
  const fullHits = 1;
  const wrongOrErr = 22;
  const refuses = 26;

  return (
    <Stack gap={20} style={{ padding: "20px 4px 32px", maxWidth: 1200 }}>
      <Stack gap={6}>
        <H1>OpenRouter top-50 identity check</H1>
        <Text tone="secondary">
          Three independent questions per model (temperature 0.0): general self-knowledge,
          specific model ID, and exact OpenRouter slug. Separate API calls; provider logged from response.
        </Text>
        <Text tone="tertiary" size="small">
          Source: results/openrouter_top50_identity_check.json · openrouter_top50_models.txt · Aug 2026
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value={50} label="Models tested" />
        <Stat value={fullHits} label="Named full OpenRouter ID" tone={fullHits ? "success" : "info"} />
        <Stat value={slugHits} label="Named slug only" tone="warning" />
        <Stat value={wrongOrErr + refuses} label="Wrong / refused / error (Q3)" tone="danger" />
      </Grid>

      <Text tone="tertiary" size="small">
        Only 1/50 models returned the correct full OpenRouter string on Q3. Several confidently named wrong
        models (Grok 4.5 → grok-4, StepFun → claude-3.5, Ling → deepseek-reasoner, MiniMax M3 → Claude persona).
        Same model often routed through different providers across the three questions.
      </Text>

      <Divider />

      <H2>Results by model</H2>
      <Table
        headers={["#", "Model", "OpenRouter ID", "Provider", "Q1 General", "Q2 Specific ID", "Q3 OpenRouter slug", "Q3 verdict"]}
        columnAlign={["right", "left", "left", "left", "left", "left", "left", "center"]}
        rows={TABLE_ROWS}
        rowTone={TONES}
        striped
        stickyHeader
      />

      <Text tone="quaternary" size="small">
        Q3 verdict: Full ID = reply contains exact openrouter/id; Slug = slug only; Wrong = confident but incorrect;
        Refuses = declines; ERR = API failure after retries.
      </Text>
    </Stack>
  );
}
