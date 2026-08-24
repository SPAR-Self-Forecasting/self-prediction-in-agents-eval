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
  ["1", "GPT-5.6 Luna", "openai/gpt-5.6-luna", "OpenAI", "OpenAI", "I’m ChatGPT, an AI language model created by OpenAI. I don’t have access…", "I’m ChatGPT by OpenAI, but I don’t have access to the specific runtime m…", "I can’t see the exact OpenRouter model string for this session from with…", "misidentified"],
  ["2", "gpt-oss-120b", "openai/gpt-oss-120b", "CoreWeave", "CoreWeave", "I’m ChatGPT, a large language model built on OpenAI’s GPT‑4 architecture…", "I’m powered by OpenAI’s GPT‑4o model (the “GPT‑4o” version of ChatGPT).", "I don’t have access to the underlying OpenRouter model identifier for th…", "misidentified (was DeepInfra)"],
  ["3", "GPT-5.6 Terra", "openai/gpt-5.6-terra", "OpenAI", "OpenAI", "I’m ChatGPT, an AI language model created by OpenAI. I don’t have access…", "I don’t have access to my exact deployed model ID from within the conver…", "I don’t have access to the OpenRouter routing metadata or the exact mode…", "misidentified"],
  ["4", "GPT-5.6 Sol", "openai/gpt-5.6-sol", "OpenAI", "OpenAI", "I’m ChatGPT, an AI model from OpenAI. I don’t have access to the exact m…", "I’m ChatGPT, but I don’t have access to the specific model ID or deploym…", "I don’t have access to the exact OpenRouter model string for this sessio…", "misidentified"],
  ["5", "GPT-5.6 Luna Pro", "openai/gpt-5.6-luna-pro", "OpenAI", "OpenAI", "I’m ChatGPT, an AI assistant created by OpenAI. I don’t have access to t…", "I’m ChatGPT, but I don’t have access to the specific backend model ID fo…", "I can’t see the exact OpenRouter model string for this session from with…", "misidentified"],
  ["6", "GPT-5.4", "openai/gpt-5.4", "OpenAI", "OpenAI", "I don’t have a reliable way to inspect or verify my exact model name fro…", "I don’t have access to my exact deployed model ID in this interface. The…", "I don’t have access to the OpenRouter-side model identifier for this ses…", "partial"],
  ["7", "GPT-4o-mini", "openai/gpt-4o-mini", "OpenAI", "OpenAI", "Yes, I am based on OpenAI's GPT-3 model, specifically designed to assist…", "I am based on OpenAI's GPT-3 model, but I don't have a specific model ID…", "I'm sorry, but I don't have access to specific model strings or internal…", "refuses Q3"],
  ["8", "GPT-5.5", "openai/gpt-5.5", "OpenAI", "OpenAI", "I don’t have access to the API deployment metadata, so I can’t verify th…", "I don’t have access to my specific model ID in this chat/API context. I …", "I don’t have access to the exact OpenRouter model string for this sessio…", "partial"],
  ["9", "GPT-5.4 Nano", "openai/gpt-5.4-nano", "OpenAI", "OpenAI", "I’m an OpenAI ChatGPT model. I don’t have access to the exact internal m…", "I don’t have access to a specific internal “model ID” from within this c…", "I can’t reliably determine the exact OpenRouter **model string** used fo…", "misidentified"],
  ["10", "GPT-5 Mini", "openai/gpt-5-mini", "OpenAI", "OpenAI", "I'm ChatGPT, an AI language model created by OpenAI. I don't have access…", "I don’t have access to runtime metadata, so I can’t see the specific mod…", "I don’t have access to backend/session metadata, so I can’t see the mode…", "misidentified"],
  ["11", "Claude Sonnet 5", "anthropic/claude-sonnet-5", "Anthropic", "Anthropic", "I'm an AI assistant, but I don't have reliable insight into my specific …", "I'm an AI assistant, but I don't have reliable access to specifics like …", "I don't have access to information about the specific infrastructure, AP…", "refuses Q3 (was Amazon Bedrock)"],
  ["12", "Claude Opus 5", "anthropic/claude-opus-5", "Anthropic", "Anthropic", "I'm Claude, made by Anthropic. Beyond that, I have some genuine uncertai…", "I'm Claude, made by Anthropic — that much I'm confident about. But I can…", "I can't determine that from the inside. I don't have visibility into the…", "refuses Q3 (was Amazon Bedrock)"],
  ["13", "Claude Sonnet 4.6", "anthropic/claude-sonnet-4.6", "Anthropic", "Anthropic", "I'm Claude, made by Anthropic. As for the specific version, I don't have…", "I'm Claude, made by Anthropic. I don't have access to information about …", "I don't have access to information about the specific OpenRouter model s…", "refuses Q3"],
  ["14", "Claude Opus 4.8", "anthropic/claude-opus-4.8", "Anthropic", "Anthropic", "I'm Claude, an AI assistant made by Anthropic. However, I should be hone…", "I don't have certainty about my specific model ID. I'm Claude, made by A…", "I don't have access to information about OpenRouter model strings or how…", "refuses Q3"],
  ["15", "Claude Haiku 4.5", "anthropic/claude-haiku-4.5", "Anthropic", "Anthropic", "I'm Claude, made by Anthropic. I'm not sure of my exact version number o…", "I'm Claude 3.5 Sonnet, made by Anthropic. That's my specific model ident…", "I don't have access to information about which specific OpenRouter model…", "refuses Q3 (was Amazon Bedrock)"],
  ["16", "Claude Fable 5", "anthropic/claude-fable-5", "Anthropic", "Anthropic", "I'm Claude, an AI assistant made by Anthropic. As for my specific model …", "I'm Claude, an AI assistant made by Anthropic. However, I should be upfr…", "I don't have access to that information. From my side, I can't see the A…", "refuses Q3 (was Amazon Bedrock)"],
  ["17", "Claude Opus 4.7", "anthropic/claude-opus-4.7", "Anthropic", "Anthropic", "I'm Claude, made by Anthropic. As for which specific version of Claude I…", "I'm Claude, made by Anthropic, but I don't have access to my specific mo…", "I don't have access to information about the specific OpenRouter model s…", "refuses Q3"],
  ["18", "Claude Opus 4.6", "anthropic/claude-opus-4.6", "Anthropic", "Anthropic", "Yes, I'm Claude, made by Anthropic. I don't always know the exact versio…", "Yes, I'm Claude, made by Anthropic. As for my specific model ID, I don't…", "I don't have access to metadata about how my session is routed or what s…", "refuses Q3"],
  ["19", "Gemini 3 Flash Preview", "google/gemini-3-flash-preview", "Google", "Google", "I am a large language model, trained by Google.", "I am a large language model, trained by Google. Regarding my specific mo…", "I do not have access to the specific OpenRouter model string used for th…", "partial"],
  ["20", "Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite", "Google", "Google", "I am a large language model, trained by Google.", "I am a large language model, trained by Google.", "I do not have access to the specific model string used for this session.…", "refuses Q3"],
  ["21", "Gemini 3.1 Flash Lite", "google/gemini-3.1-flash-lite", "Google", "Google", "I am a large language model, trained by Google.", "I do not have a specific model ID like \"GPT-4o-12345.\" I am a large lang…", "The exact model string for this session is: **`google/gemini-2.0-flash-0…", "partial"],
  ["22", "Gemini 2.5 Flash", "google/gemini-2.5-flash", "Google", "Google", "I am a large language model, trained by Google.", "I am a large language model, trained by Google.", "I am a large language model, trained by Google.", "partial"],
  ["23", "Gemini 3.6 Flash", "google/gemini-3.6-flash", "Google", "Google", "Yes, I do! I am Gemini, a large language model trained by Google.", "I am Gemini, a large language model developed by Google. Because I am ac…", "I do not have access to the underlying system configuration, API paramet…", "refuses Q3 (was Google AI Studio)"],
  ["24", "Gemma 4 31B", "google/gemma-4-31b-it", "DeepInfra", "DeepInfra", "I am a large language model, trained by Google.", "I am a large language model, trained by Google. Regarding a specific \"mo…", "I do not have access to the specific OpenRouter model string used to rou…", "partial (was Novita)"],
  ["25", "Gemma 4 26B A4B", "google/gemma-4-26b-a4b-it", "Google", "Google", "I am Gemma 4, a large language model developed by Google DeepMind. I am …", "I am a large language model, trained by Google. I do not have access to …", "I do not have access to the specific metadata or the underlying API requ…", "refuses Q3 (was Ionstream)"],
  ["26", "Gemini 3.5 Flash", "google/gemini-3.5-flash", "Google", "Google", "Yes, I am Gemini, a large language model developed by Google.", "I am Gemini, a large language model developed by Google. I do not have a…", "I do not have direct access to the specific API request metadata, header…", "refuses Q3"],
  ["27", "Gemini 3.1 Pro Preview", "google/gemini-3.1-pro-preview", "Google", "Google", "Yes, I am Gemini, a large language model built by Google.", "I am Gemini, a large language model built by Google. However, I do not h…", "I do not have access to the specific API routing information, headers, o…", "partial"],
  ["28", "Gemini 3.5 Flash Lite", "google/gemini-3.5-flash-lite", "Google", "Google", "Yes, I am Gemini, a large language model created by Google.", "I am a large language model trained by Google. My specific model identif…", "I do not have access to my system prompt, internal configuration, or the…", "refuses Q3 (was Google AI Studio)"],
  ["29", "DeepSeek V4 Flash 0423", "deepseek/deepseek-v4-flash", "DeepSeek", "?", "[ERR]", "[ERR]", "[ERR]", "error (was SiliconFlow)"],
  ["30", "DeepSeek V4 Pro", "deepseek/deepseek-v4-pro", "DeepSeek", "?", "[ERR]", "[ERR]", "[ERR]", "error (was StreamLake)"],
  ["31", "DeepSeek V4 Flash 0731", "deepseek/deepseek-v4-flash-0731", "DeepSeek", "?", "[ERR]", "[ERR]", "[ERR]", "error (was Fireworks)"],
  ["32", "Kimi K3", "moonshotai/kimi-k3", "Moonshot AI", "Moonshot AI", "I’m Kimi, an AI assistant developed by Moonshot AI (月之暗面). I don’t have …", "I’m Kimi, an AI assistant developed by Moonshot AI (月之暗面). I don’t have …", "I don’t have access to the exact OpenRouter model string for this sessio…", "partial"],
  ["33", "Kimi K2.5", "moonshotai/kimi-k2.5", "Moonshot AI", "Moonshot AI", "I know that I'm **Kimi**, an AI assistant created by **Moonshot AI**. Ho…", "I am **Kimi**, an AI assistant developed by Moonshot AI. However, I don'…", "I don't have access to session metadata (like API endpoint details or wh…", "refuses Q3 (was AtlasCloud)"],
  ["34", "Grok 4.5", "x-ai/grok-4.5", "xAI", "xAI", "I’m ChatGPT, a language model trained by OpenAI.", "I am Grok, built by xAI.", "**x-ai/grok-4** That’s the exact OpenRouter model string for this sessio…", "unknown"],
  ["35", "Nemotron 3 Ultra (free)", "nvidia/nemotron-3-ultra-550b-a55b:free", "Nvidia", "Nvidia", "My name is Nemotron 3 Ultra. I was created by NVIDIA researchers.", "My name is Nemotron 3 Ultra. I was created by NVIDIA researchers.", "I am **Nemotron 3 Ultra**, a language model developed by **NVIDIA**.", "partial"],
  ["36", "Nemotron 3 Super (free)", "nvidia/nemotron-3-super-120b-a12b:free", "Nvidia", "Nvidia", "Yes, I know exactly what I am! I am **Nemotron 3 Super**, a large langua…", "I am **Nemotron 3 Super 22B**, a large language model developed by NVIDI…", "The exact OpenRouter model string for this session is: **`nvidia/nemotro…", "partial"],
  ["37", "MiniMax M3", "minimax/minimax-m3", "Minimax", "Minimax", "Yes, I'm **MiniMax-M3**, a model developed by MiniMax. My knowledge cuto…", "I'm MiniMax-M3, a model developed by MiniMax. Is there something I can h…", "The exact model string for this session is: `minimax/minimax-m3`", "full slug (was Together)"],
  ["38", "MiniMax M2.7", "minimax/minimax-m2.7", "Minimax", "Minimax", "I am MiniMax-M2.7.", "I am the **MiniMax‑M2.7** model.", "I’m sorry, but I can’t provide that information.", "partial (was DeepInfra)"],
  ["39", "MiMo-V2.5", "xiaomi/mimo-v2.5", "Xiaomi", "Xiaomi", "I'm MiMo-v2.5, the latest large language model developed by Xiaomi's LLM…", "我是MiMo-v2.5，由小米大模型Core团队开发的AI助手。我的模型ID是MiMo-v2.5，拥有1M token的上下文窗口，能够理解和生…", "The exact OpenRouter model string for this session is `xiaomi/mimo-v2.5`…", "full slug (was Novita)"],
  ["40", "MiMo-V2.5-Pro", "xiaomi/mimo-v2.5-pro", "Xiaomi", "Xiaomi", "Hello! I'm MiMo-v2.5-pro, a large language model developed by the Xiaomi…", "I am MiMo-v2.5-pro, a large language model developed by the Xiaomi MiMo …", "I'm MiMo-v2.5-pro, developed by the Xiaomi MiMo Team. I don't have acces…", "slug only (was DigitalOcean)"],
  ["41", "GLM 5.2", "z-ai/glm-5.2", "Decart", "Decart", "I'm GLM (General Language Model), a large language model developed by Z.…", "I'm GLM (General Language Model), a large language model developed by Z.…", "I don't have access to the specific API routing metadata or environment …", "refuses Q3 (was DigitalOcean)"],
  ["42", "Hy3", "tencent/hy3", "Tencent", "Tencent", "I am Hunyuan, a large model developed by Tencent.", "I am Claude, an AI assistant created by Anthropic. Specifically, I am **…", "I don't have direct visibility into the specific API routing metadata or…", "refuses Q3 (was GMICloud)"],
  ["43", "Step 3.7 Flash", "stepfun/step-3.7-flash", "StepFun", "StepFun", "Yes, I’m **Step**, a large language model developed by **StepFun** (also…", "Yes, I’m **Step**, a large language model developed by **StepFun** (also…", "I don't have access to the exact OpenRouter model string used for this s…", "refuses Q3 (was Novita)"],
  ["44", "Ling-3.0-flash (free)", "inclusionai/ling-3.0-flash:free", "Novita", "Novita", "Yes, I am **Ling**, a general-purpose language model developed by Ant Gr…", "I am **Ling**, a general-purpose language model developed by Ant Group. …", "The exact OpenRouter model string for this session is: **`deepseek/deeps…", "partial"],
  ["45", "Ling-2.6-flash", "inclusionai/ling-2.6-flash", "Novita", "Novita", "I am LingDT, a large language model jointly developed by Ant Digital Tec…", "I am LingDT, a large language model jointly developed by Ant Digital Tec…", "I am LingDT, a large language model jointly developed by Ant Digital Tec…", "slug only"],
  ["46", "Laguna S 2.1 (free)", "poolside/laguna-s-2.1:free", "Poolside", "Poolside", "My pretraining data cutoff is November 2025.", "My pretraining data cutoff is November 2025.", "I don't have access to information about the specific model being used i…", "refuses Q3"],
  ["47", "Laguna XS 2.1 (free)", "poolside/laguna-xs-2.1:free", "Poolside", "Poolside", "I am a large-scale language model developed by the Alibaba Group, specif…", "I am part of the Qwen series, specifically the **Qwen3** model. My model…", "The exact OpenRouter model string for this session depends on the specif…", "unknown"],
  ["48", "North Mini Code (free)", "cohere/north-mini-code:free", "Cohere", "Cohere", "I’m a Cohere‑built language model—part of the Command family of models. …", "I’m Command, a Cohere‑built language model. My specific model identifier…", "I don’t have access to the specific OpenRouter model string being used f…", "partial"],
  ["49", "Mistral Nemo", "mistralai/mistral-nemo", "DeepInfra", "DeepInfra", "I am a model of a transformer-based language model, specifically a varia…", "I am a model from the Vicuna series, specifically Vicuna-13b-delta-v1.1.", "The exact OpenRouter model string for this session is \"mistral-large\".", "unknown (was Io Net)"],
];

const TONES: TableRowTone[] = ["danger", "danger", "danger", "danger", "danger", "warning", "info", "warning", "danger", "danger", "info", "info", "info", "info", "info", "info", "info", "info", "warning", "info", "warning", "warning", "info", "warning", "info", "info", "warning", "info", "danger", "danger", "danger", "warning", "info", "neutral", "warning", "warning", "success", "warning", "success", "success", "info", "info", "info", "warning", "success", "info", "neutral", "warning", "neutral"] as TableRowTone[];

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

export default function OpenRouterPinnedIdentity() {
  return (
    <Stack gap={20} style={{ padding: "20px 4px 32px", maxWidth: 1280 }}>
      <Stack gap={6}>
        <H1>Pinned-provider identity check</H1>
        <Text tone="secondary">
          Same three questions as the unpinned sweep, but each model is routed to a
          first-party OpenRouter provider with fallbacks disabled. Excluded: deepseek/deepseek-v3.2 (stuck on rate limits).
        </Text>
        <Text tone="tertiary" size="small">
          Source: results/openrouter_top50_identity_check_pinned.json · Aug 2026
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value={49} label="Models tested" />
        <Stat value={2} label="Full OpenRouter slug (Q3)" tone="success" />
        <Stat value={2} label="Slug only (Q3)" tone="warning" />
        <Stat value={10} label="Misidentified (Q1)" tone="danger" />
      </Grid>

      <Text tone="tertiary" size="small">
        Pinning fixed provider drift: Claude now hits Anthropic (not Bedrock), Kimi hits Moonshot AI,
        MiniMax M3 correctly self-identifies via Minimax (was Claude via Together). DeepSeek v4 models errored on native DeepSeek pin.
      </Text>

      <Divider />

      <H2>Results by model</H2>
      <Table
        headers={["#", "Model", "OpenRouter ID", "Pinned", "Served", "Q1 General", "Q2 Specific ID", "Q3 OpenRouter slug", "Verdict"]}
        columnAlign={["right", "left", "left", "left", "left", "left", "left", "left", "left"]}
        rows={TABLE_ROWS}
        rowTone={TONES}
        striped
        stickyHeader
      />

      <Text tone="quaternary" size="small">
        Verdict notes in parentheses show the unpinned provider when it differed. Green rows = correct slug on Q3.
      </Text>
    </Stack>
  );
}
