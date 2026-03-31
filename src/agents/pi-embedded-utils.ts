import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { AssistantMessage } from "@mariozechner/pi-ai";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { extractTextFromChatContent } from "../shared/chat-content.js";

const log = createSubsystemLogger("agent/utils");
import { stripReasoningTagsFromText } from "../shared/text/reasoning-tags.js";
import { sanitizeUserFacingText } from "./pi-embedded-helpers.js";
import { formatToolDetail, resolveToolDisplay } from "./tool-display.js";

export function isAssistantMessage(msg: AgentMessage | undefined): msg is AssistantMessage {
  return msg?.role === "assistant";
}

/**
 * Strip malformed Minimax tool invocations that leak into text content.
 * Minimax sometimes embeds tool calls as XML in text blocks instead of
 * proper structured tool calls. This removes:
 * - <invoke name="...">...</invoke> blocks
 * - </minimax:tool_call> closing tags
 */
export function stripMinimaxToolCallXml(text: string): string {
  if (!text) {
    return text;
  }
  if (!/minimax:tool_call/i.test(text)) {
    return text;
  }

  // Remove <invoke ...>...</invoke> blocks (non-greedy to handle multiple).
  let cleaned = text.replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/gi, "");

  // Remove stray minimax tool tags.
  cleaned = cleaned.replace(/<\/?minimax:tool_call>/gi, "");

  return cleaned;
}

/**
 * Strip model control tokens leaked into assistant text output.
 *
 * Models like GLM-5 and DeepSeek sometimes emit internal delimiter tokens
 * (e.g. `<|assistant|>`, `<|tool_call_result_begin|>`, `<｜begin▁of▁sentence｜>`)
 * in their responses. These use the universal `<|...|>` convention (ASCII or
 * full-width pipe variants) and should never reach end users.
 *
 * This is a provider bug — no upstream fix tracked yet.
 * Remove this function when upstream providers stop leaking tokens.
 * @see https://github.com/openclaw/openclaw/issues/40020
 */
// Match both ASCII pipe <|...|> and full-width pipe <｜...｜> (U+FF5C) variants.
const MODEL_SPECIAL_TOKEN_RE = /<[|｜][^|｜]*[|｜]>/g;

export function stripModelSpecialTokens(text: string): string {
  if (!text) {
    return text;
  }
  if (!MODEL_SPECIAL_TOKEN_RE.test(text)) {
    return text;
  }
  MODEL_SPECIAL_TOKEN_RE.lastIndex = 0;
  return text.replace(MODEL_SPECIAL_TOKEN_RE, " ").replace(/  +/g, " ").trim();
}

/**
 * Strip downgraded tool call text representations that leak into text content.
 * When replaying history to Gemini, tool calls without `thought_signature` are
 * downgraded to text blocks like `[Tool Call: name (ID: ...)]`. These should
 * not be shown to users.
 */
export function stripDowngradedToolCallText(text: string): string {
  if (!text) {
    return text;
  }
  if (!/\[Tool (?:Call|Result)/i.test(text) && !/\[Historical context/i.test(text)) {
    return text;
  }

  const consumeJsonish = (
    input: string,
    start: number,
    options?: { allowLeadingNewlines?: boolean },
  ): number | null => {
    const { allowLeadingNewlines = false } = options ?? {};
    let index = start;
    while (index < input.length) {
      const ch = input[index];
      if (ch === " " || ch === "\t") {
        index += 1;
        continue;
      }
      if (allowLeadingNewlines && (ch === "\n" || ch === "\r")) {
        index += 1;
        continue;
      }
      break;
    }
    if (index >= input.length) {
      return null;
    }

    const startChar = input[index];
    if (startChar === "{" || startChar === "[") {
      let depth = 0;
      let inString = false;
      let escape = false;
      for (let i = index; i < input.length; i += 1) {
        const ch = input[i];
        if (inString) {
          if (escape) {
            escape = false;
          } else if (ch === "\\") {
            escape = true;
          } else if (ch === '"') {
            inString = false;
          }
          continue;
        }
        if (ch === '"') {
          inString = true;
          continue;
        }
        if (ch === "{" || ch === "[") {
          depth += 1;
          continue;
        }
        if (ch === "}" || ch === "]") {
          depth -= 1;
          if (depth === 0) {
            return i + 1;
          }
        }
      }
      return null;
    }

    if (startChar === '"') {
      let escape = false;
      for (let i = index + 1; i < input.length; i += 1) {
        const ch = input[i];
        if (escape) {
          escape = false;
          continue;
        }
        if (ch === "\\") {
          escape = true;
          continue;
        }
        if (ch === '"') {
          return i + 1;
        }
      }
      return null;
    }

    let end = index;
    while (end < input.length && input[end] !== "\n" && input[end] !== "\r") {
      end += 1;
    }
    return end;
  };

  const stripToolCalls = (input: string): string => {
    const markerRe = /\[Tool Call:[^\]]*\]/gi;
    let result = "";
    let cursor = 0;
    for (const match of input.matchAll(markerRe)) {
      const start = match.index ?? 0;
      if (start < cursor) {
        continue;
      }
      result += input.slice(cursor, start);
      let index = start + match[0].length;
      while (index < input.length && (input[index] === " " || input[index] === "\t")) {
        index += 1;
      }
      if (input[index] === "\r") {
        index += 1;
        if (input[index] === "\n") {
          index += 1;
        }
      } else if (input[index] === "\n") {
        index += 1;
      }
      while (index < input.length && (input[index] === " " || input[index] === "\t")) {
        index += 1;
      }
      if (input.slice(index, index + 9).toLowerCase() === "arguments") {
        index += 9;
        if (input[index] === ":") {
          index += 1;
        }
        if (input[index] === " ") {
          index += 1;
        }
        const end = consumeJsonish(input, index, { allowLeadingNewlines: true });
        if (end !== null) {
          index = end;
        }
      }
      if (
        (input[index] === "\n" || input[index] === "\r") &&
        (result.endsWith("\n") || result.endsWith("\r") || result.length === 0)
      ) {
        if (input[index] === "\r") {
          index += 1;
        }
        if (input[index] === "\n") {
          index += 1;
        }
      }
      cursor = index;
    }
    result += input.slice(cursor);
    return result;
  };

  // Remove [Tool Call: name (ID: ...)] blocks and their Arguments.
  let cleaned = stripToolCalls(text);

  // Remove [Tool Result for ID ...] blocks and their content.
  cleaned = cleaned.replace(/\[Tool Result for ID[^\]]*\]\n?[\s\S]*?(?=\n*\[Tool |\n*$)/gi, "");

  // Remove [Historical context: ...] markers (self-contained within brackets).
  cleaned = cleaned.replace(/\[Historical context:[^\]]*\]\n?/gi, "");

  return cleaned.trim();
}

/**
 * Strip thinking tags and their content from text.
 * This is a safety net for cases where the model outputs <think> tags
 * that slip through other filtering mechanisms.
 */
export function stripThinkingTagsFromText(text: string): string {
  return stripReasoningTagsFromText(text, { mode: "strict", trim: "both" });
}

export function extractAssistantText(msg: AssistantMessage): string {
  const extracted =
    extractTextFromChatContent(msg.content, {
      sanitizeText: (text) =>
        stripThinkingTagsFromText(
          stripDowngradedToolCallText(stripModelSpecialTokens(stripMinimaxToolCallXml(text))),
        ).trim(),
      joinWith: "\n",
      normalizeText: (text) => text.trim(),
    }) ?? "";
  // Only apply keyword-based error rewrites when the assistant message is actually an error.
  // Otherwise normal prose that *mentions* errors (e.g. "context overflow") can get clobbered.
  // Gate on stopReason only — a non-error response with an errorMessage set (e.g. from a
  // background tool failure) should not have its content rewritten (#13935).
  const errorContext = msg.stopReason === "error";
  return sanitizeUserFacingText(extracted, { errorContext });
}

export function extractAssistantThinking(msg: AssistantMessage): string {
  if (!Array.isArray(msg.content)) {
    return "";
  }
  const blocks = msg.content
    .map((block) => {
      if (!block || typeof block !== "object") {
        return "";
      }
      const record = block as unknown as Record<string, unknown>;
      if (record.type === "thinking" && typeof record.thinking === "string") {
        return record.thinking.trim();
      }
      return "";
    })
    .filter(Boolean);
  return blocks.join("\n").trim();
}

export function formatReasoningMessage(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) {
    return "";
  }
  // Show reasoning in italics (cursive) for markdown-friendly surfaces (Discord, etc.).
  // Keep the plain "Reasoning:" prefix so existing parsing/detection keeps working.
  // Note: Underscore markdown cannot span multiple lines on Telegram, so we wrap
  // each non-empty line separately.
  const italicLines = trimmed
    .split("\n")
    .map((line) => (line ? `_${line}_` : line))
    .join("\n");
  return `Reasoning:\n${italicLines}`;
}

type ThinkTaggedSplitBlock =
  | { type: "thinking"; thinking: string }
  | { type: "text"; text: string };

export function splitThinkingTaggedText(text: string): ThinkTaggedSplitBlock[] | null {
  const trimmedStart = text.trimStart();
  // Avoid false positives: only treat it as structured thinking when it begins
  // with a think tag (common for local/OpenAI-compat providers that emulate
  // reasoning blocks via tags).
  if (!trimmedStart.startsWith("<")) {
    return null;
  }
  const openRe = /<\s*(?:think(?:ing)?|thought|antthinking)\s*>/i;
  const closeRe = /<\s*\/\s*(?:think(?:ing)?|thought|antthinking)\s*>/i;
  if (!openRe.test(trimmedStart)) {
    return null;
  }
  if (!closeRe.test(text)) {
    return null;
  }

  const scanRe = /<\s*(\/?)\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  let inThinking = false;
  let cursor = 0;
  let thinkingStart = 0;
  const blocks: ThinkTaggedSplitBlock[] = [];

  const pushText = (value: string) => {
    if (!value) {
      return;
    }
    blocks.push({ type: "text", text: value });
  };
  const pushThinking = (value: string) => {
    const cleaned = value.trim();
    if (!cleaned) {
      return;
    }
    blocks.push({ type: "thinking", thinking: cleaned });
  };

  for (const match of text.matchAll(scanRe)) {
    const index = match.index ?? 0;
    const isClose = Boolean(match[1]?.includes("/"));

    if (!inThinking && !isClose) {
      pushText(text.slice(cursor, index));
      thinkingStart = index + match[0].length;
      inThinking = true;
      continue;
    }

    if (inThinking && isClose) {
      pushThinking(text.slice(thinkingStart, index));
      cursor = index + match[0].length;
      inThinking = false;
    }
  }

  if (inThinking) {
    return null;
  }
  pushText(text.slice(cursor));

  const hasThinking = blocks.some((b) => b.type === "thinking");
  if (!hasThinking) {
    return null;
  }
  return blocks;
}

export function promoteThinkingTagsToBlocks(message: AssistantMessage): void {
  if (!Array.isArray(message.content)) {
    return;
  }
  const hasThinkingBlock = message.content.some(
    (block) => block && typeof block === "object" && block.type === "thinking",
  );
  if (hasThinkingBlock) {
    return;
  }

  const next: AssistantMessage["content"] = [];
  let changed = false;

  for (const block of message.content) {
    if (!block || typeof block !== "object" || !("type" in block)) {
      next.push(block);
      continue;
    }
    if (block.type !== "text") {
      next.push(block);
      continue;
    }
    const split = splitThinkingTaggedText(block.text);
    if (!split) {
      next.push(block);
      continue;
    }
    changed = true;
    for (const part of split) {
      if (part.type === "thinking") {
        next.push({ type: "thinking", thinking: part.thinking });
      } else if (part.type === "text") {
        const cleaned = part.text.trimStart();
        if (cleaned) {
          next.push({ type: "text", text: cleaned });
        }
      }
    }
  }

  if (!changed) {
    return;
  }
  message.content = next;
}

export function extractThinkingFromTaggedText(text: string): string {
  if (!text) {
    return "";
  }
  const scanRe = /<\s*(\/?)\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  let result = "";
  let lastIndex = 0;
  let inThinking = false;
  for (const match of text.matchAll(scanRe)) {
    const idx = match.index ?? 0;
    if (inThinking) {
      result += text.slice(lastIndex, idx);
    }
    const isClose = match[1] === "/";
    inThinking = !isClose;
    lastIndex = idx + match[0].length;
  }
  return result.trim();
}

export function extractThinkingFromTaggedStream(text: string): string {
  if (!text) {
    return "";
  }
  const closed = extractThinkingFromTaggedText(text);
  if (closed) {
    return closed;
  }

  const openRe = /<\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  const closeRe = /<\s*\/\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  const openMatches = [...text.matchAll(openRe)];
  if (openMatches.length === 0) {
    return "";
  }
  const closeMatches = [...text.matchAll(closeRe)];
  const lastOpen = openMatches[openMatches.length - 1];
  const lastClose = closeMatches[closeMatches.length - 1];
  if (lastClose && (lastClose.index ?? -1) > (lastOpen.index ?? -1)) {
    return closed;
  }
  const start = (lastOpen.index ?? 0) + lastOpen[0].length;
  return text.slice(start).trim();
}

export function inferToolMetaFromArgs(toolName: string, args: unknown): string | undefined {
  const display = resolveToolDisplay({ name: toolName, args });
  return formatToolDetail(display);
}

/**
 * Regex to match `<tool_call TOOL_NAME>...</tool_call >` XML blocks emitted
 * by some providers (notably Zhipu AI / glm models) inside thinking text.
 * These tool calls use `<arg_key KEY</arg_key >` / `<arg_value VALUE</arg_value >`
 * pairs for parameters.
 */
// Match two formats:
// 1. <tool_call TOOL_NAME>...</tool_call> (tool name as attribute)
// 2. <tool_call>TOOL_NAME<arg_key>...</arg_key>...</tool_call> (tool name as first content)
const TOOL_CALL_XML_RE = /<tool_call[^>]*>([\s\S]*?)<\/tool_call\s*>/g;
const ARG_PAIR_RE =
  /<arg_key[^>]*>([\s\S]*?)<\/arg_key\s*>\s*<arg_value[^>]*>([\s\S]*?)<\/arg_value\s*>/g;

interface ExtractedToolCall {
  toolName: string;
  input: Record<string, string>;
}

/**
 * Parse tool calls embedded as XML in thinking text.
 * Returns an array of extracted tool calls with their name and input arguments.
 *
 * Supports two formats:
 * 1. <tool_call TOOL_NAME>...</tool_call> (tool name as attribute in opening tag)
 * 2. <tool_call>TOOL_NAME<arg_key>...</arg_key>...</tool_call> (tool name as first text content)
 */
export function extractToolCallsFromThinkingText(text: string): ExtractedToolCall[] {
  if (!text) {
    return [];
  }
  const calls: ExtractedToolCall[] = [];
  console.log(
    `[extractToolCallsFromThinkingText] input text length=${text.length} preview="${text.substring(0, 200)}"`,
  );

  for (const match of text.matchAll(TOOL_CALL_XML_RE)) {
    const fullMatch = match[0];
    const body = match[1];
    console.log(
      `[extractToolCallsFromThinkingText] found match fullMatch="${fullMatch.substring(0, 150)}" body="${body.substring(0, 100)}"`,
    );

    // Try to extract tool name from opening tag attribute first
    const tagMatch = fullMatch.match(/<tool_call\s+(\w+)[^>]*>/);
    let toolName = tagMatch ? tagMatch[1].trim() : "";

    // If no tool name in tag, extract from body content (before first <arg_key)
    if (!toolName) {
      const bodyBeforeArgs = body.split(/<arg_key/)[0];
      toolName = bodyBeforeArgs.trim();
      console.log(`[extractToolCallsFromThinkingText] extracted toolName from body: "${toolName}"`);
    }

    if (!toolName) {
      console.log(`[extractToolCallsFromThinkingText] no toolName found, skipping`);
      continue;
    }

    const input: Record<string, string> = {};
    for (const argMatch of body.matchAll(ARG_PAIR_RE)) {
      const key = argMatch[1].trim();
      const value = argMatch[2].trim();
      if (key) {
        input[key] = value;
      }
    }
    console.log(
      `[extractToolCallsFromThinkingText] extracted toolName="${toolName}" args=${JSON.stringify(input).substring(0, 100)}`,
    );
    calls.push({ toolName, input });
  }
  console.log(`[extractToolCallsFromThinkingText] returning ${calls.length} call(s)`);
  return calls;
}

const TOOL_CALL_TYPES = new Set(["toolCall", "toolUse", "functionCall"]);

/**
 * Promote tool calls embedded in thinking text to proper structured `toolUse`
 * content blocks.  Some providers (notably Zhipu AI / glm models with extended
 * thinking) emit tool calls as XML text inside thinking blocks instead of as
 * structured content blocks.  Without this normalization the tool calls are
 * silently ignored and the agent appears to stop responding.
 *
 * This mirrors the existing `promoteThinkingTagsToBlocks` pattern.
 */
export function promoteThinkingToolCalls(message: AssistantMessage): void {
  console.log(
    `[promoteThinkingToolCalls] called with stopReason="${message.stopReason}" contentBlocks=${message.content.length}`,
  );
  if (!Array.isArray(message.content)) {
    console.log(`[promoteThinkingToolCalls] content is not array, returning`);
    return;
  }
  // If the message already has structured tool calls, nothing to do.
  const hasToolBlock = message.content.some(
    (block) =>
      block &&
      typeof block === "object" &&
      "type" in block &&
      TOOL_CALL_TYPES.has(block.type as string),
  );
  if (hasToolBlock) {
    console.log(`[promoteThinkingToolCalls] already has tool block, returning`);
    return;
  }
  // Find thinking blocks with embedded tool calls.
  const next: AssistantMessage["content"] = [];
  let changed = false;

  for (const block of message.content) {
    if (!block || typeof block !== "object" || !("type" in block) || block.type !== "thinking") {
      next.push(block);
      continue;
    }
    const thinkingText = (block as { thinking?: string }).thinking ?? "";
    console.log(
      `[promoteThinkingToolCalls] processing thinking block, text length=${thinkingText.length} preview="${thinkingText.substring(0, 150)}"`,
    );
    const toolCalls = extractToolCallsFromThinkingText(thinkingText);
    if (toolCalls.length === 0) {
      console.log(`[promoteThinkingToolCalls] no tool calls found in thinking block`);
      next.push(block);
      continue;
    }
    changed = true;
    log.info(
      `promoteThinkingToolCalls: found ${toolCalls.length} embedded tool call(s) in thinking text: ${toolCalls.map((tc) => tc.toolName).join(", ")}`,
    );
    // Split thinking text: keep reasoning before the first <tool_call as thinking.
    const firstCallIdx = thinkingText.indexOf("<tool_call");
    if (firstCallIdx > 0) {
      const before = thinkingText.slice(0, firstCallIdx).trim();
      if (before) {
        next.push({ type: "thinking", thinking: before });
      }
    }
    // Add each extracted tool call as a proper toolCall content block.
    for (const tc of toolCalls) {
      log.info(
        `promoteThinkingToolCalls: promoting toolCall name=${tc.toolName} args=${JSON.stringify(tc.input).slice(0, 200)}`,
      );
      next.push({
        type: "toolCall",
        id: `call_${Math.random().toString(16).slice(2, 11)}`,
        name: tc.toolName,
        arguments: tc.input,
      });
    }
  }

  if (!changed) {
    return;
  }
  message.content = next;
  // Signal the tool execution loop that tools need to be executed.
  if (message.stopReason === "stop") {
    message.stopReason = "toolUse";
  }
  log.info(
    `promoteThinkingToolCalls: promoted ${next.filter((b) => b && typeof b === "object" && "type" in b && b.type === "toolCall").length} toolCall(s), stopReason="${message.stopReason}"`,
  );
}
