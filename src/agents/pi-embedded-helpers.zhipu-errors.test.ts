import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import {
  classifyFailoverReason,
  isFailoverAssistantError,
  isOverloadedErrorMessage,
  isRateLimitErrorMessage,
  isTimeoutErrorMessage,
} from "./pi-embedded-helpers.js";

// ─── Helpers: simulate assistant messages from the session ───

function makeAssistantError(opts: { errorMessage: string; stopReason?: string }): AgentMessage {
  return {
    role: "assistant",
    content: [],
    stopReason: opts.stopReason ?? "error",
    errorMessage: opts.errorMessage,
    usage: { input: 0, output: 0, totalTokens: 0 },
  } as unknown as AgentMessage;
}

/**
 * Simulate the retry decision logic from run.ts.
 * Returns the classification result and whether retry should fire.
 *
 * In run.ts the flow is:
 *   1. isFailoverAssistantError(msg)  → must be true for failover path
 *   2. classifyFailoverReason(msg.errorMessage) → reason
 *   3. isRetryableError check: rate_limit | timeout | overloaded
 */
function simulateRetryDecision(msg: AgentMessage): {
  isFailover: boolean;
  reason: ReturnType<typeof classifyFailoverReason>;
  isRetryable: boolean;
} {
  const isFailover = isFailoverAssistantError(msg);
  const reason = classifyFailoverReason((msg as { errorMessage?: string }).errorMessage ?? "");
  const isRetryable = reason === "rate_limit" || reason === "timeout" || reason === "overloaded";
  return { isFailover, reason, isRetryable };
}

// ─── Tests ───

describe("Zhipu AI error detection", () => {
  // ──── Raw pattern matching ────

  describe("rate limit errors (code 1302)", () => {
    it("detects LLM error 1302 plain text", () => {
      const msg =
        "LLM error 1302: 您的账户已达到速率限制，请您控制请求频率 (request_id: 20260320173545bdd82da5bd37400f)";
      expect(isRateLimitErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("rate_limit");
    });

    it("detects Chinese rate limit text", () => {
      expect(isRateLimitErrorMessage("您的账户已达到速率限制，请您控制请求频率")).toBe(true);
    });

    it("detects request frequency control message", () => {
      expect(isRateLimitErrorMessage("请您控制请求频率")).toBe(true);
    });

    it("detects JSON format with code 1302 (production format)", () => {
      const msg =
        '{"error":{"code":"1302","message":"您的账户已达到速率限制，请您控制请求频率"},"request_id":"20260331155218c06708d0bc0445f5"}';
      expect(isRateLimitErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("rate_limit");
    });

    it("detects JSON format with code 1302 without Chinese text", () => {
      const msg =
        '{"error":{"code":"1302","message":"Rate limit exceeded"},"request_id":"test123"}';
      expect(isRateLimitErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("rate_limit");
    });
  });

  describe("overloaded errors (code 1305)", () => {
    it("detects LLM error 1305 plain text", () => {
      const msg =
        "LLM error 1305: 该模型当前访问量过大，请稍后再试 (request_id: 20260320173545bdd82da5bd37400f)";
      expect(isOverloadedErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("overloaded");
    });

    it("detects JSON format with code 1305", () => {
      const msg =
        '{"error":{"code":"1305","message":"该模型当前访问量过大，请稍后再试"},"request_id":"20260331155218c06708d0bc0445f5"}';
      expect(isOverloadedErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("overloaded");
    });

    it("detects JSON format with code 1305 without Chinese text", () => {
      const msg =
        '{"error":{"code":"1305","message":"Model overloaded, please retry"},"request_id":"test123"}';
      expect(isOverloadedErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("overloaded");
    });

    it("detects Chinese overloaded text", () => {
      const msg = "该模型当前访问量过大，请稍后再试";
      expect(isOverloadedErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("overloaded");
    });
  });

  describe("network and timeout errors", () => {
    it("detects LLM error 1234 network error", () => {
      const msg =
        "LLM error 1234: 网络错误，错误id：20260320031740b46136f53d4c4087，请联系客服。 (request_id: 20260320031740b46136f53d4c4087)";
      expect(isTimeoutErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("timeout");
    });

    it("detects LLM error api_error Internal Network Failure", () => {
      const msg =
        "LLM error api_error: Internal Network Failure (request_id: 20260319180423e2bde364d1044614)";
      expect(isTimeoutErrorMessage(msg)).toBe(true);
      expect(classifyFailoverReason(msg)).toBe("timeout");
    });

    it("detects Chinese network error", () => {
      expect(isTimeoutErrorMessage("网络错误，请稍后重试")).toBe(true);
    });

    it("detects Internal Network Failure", () => {
      expect(isTimeoutErrorMessage("Internal Network Failure")).toBe(true);
    });

    it("detects Request was aborted as timeout", () => {
      expect(isTimeoutErrorMessage("Request was aborted.")).toBe(true);
      expect(classifyFailoverReason("Request was aborted.")).toBe("timeout");
    });
  });

  // ──── Full flow simulation ────
  // These tests simulate the exact path run.ts takes when processing an error.

  describe("full failover flow simulation", () => {
    describe("rate limit 1302 — the exact production error", () => {
      it("triggers retry for JSON 1302 error from Zhipu AI", () => {
        // This is the EXACT error from the user's production report
        const errorMessage =
          '{"error":{"code":"1302","message":"您的账户已达到速率限制，请您控制请求频率"},"request_id":"2026040122102272e021f476204f96"}';
        const assistant = makeAssistantError({ errorMessage });
        const decision = simulateRetryDecision(assistant);

        expect(decision.isFailover).toBe(true);
        expect(decision.reason).toBe("rate_limit");
        expect(decision.isRetryable).toBe(true);
      });

      it("triggers retry for LLM error 1302 wrapped format", () => {
        const errorMessage =
          "LLM error 1302: 您的账户已达到速率限制，请您控制请求频率 (request_id: abc)";
        const assistant = makeAssistantError({ errorMessage });
        const decision = simulateRetryDecision(assistant);

        expect(decision.isFailover).toBe(true);
        expect(decision.reason).toBe("rate_limit");
        expect(decision.isRetryable).toBe(true);
      });
    });

    describe("overloaded 1305 flow", () => {
      it("triggers retry for JSON 1305 error", () => {
        const errorMessage =
          '{"error":{"code":"1305","message":"该模型当前访问量过大，请稍后再试"},"request_id":"test456"}';
        const assistant = makeAssistantError({ errorMessage });
        const decision = simulateRetryDecision(assistant);

        expect(decision.isFailover).toBe(true);
        expect(decision.reason).toBe("overloaded");
        expect(decision.isRetryable).toBe(true);
      });
    });

    describe("timeout/abort flow", () => {
      it("triggers retry for 'Request was aborted.'", () => {
        const errorMessage = "Request was aborted.";
        const assistant = makeAssistantError({ errorMessage });
        const decision = simulateRetryDecision(assistant);

        expect(decision.isFailover).toBe(true);
        expect(decision.reason).toBe("timeout");
        expect(decision.isRetryable).toBe(true);
      });

      it("triggers retry for network error 1234", () => {
        const errorMessage = "LLM error 1234: 网络错误 (request_id: xyz789)";
        const assistant = makeAssistantError({ errorMessage });
        const decision = simulateRetryDecision(assistant);

        expect(decision.isFailover).toBe(true);
        expect(decision.reason).toBe("timeout");
        expect(decision.isRetryable).toBe(true);
      });
    });

    describe("non-retryable errors must NOT trigger retry", () => {
      it("does not retry image dimension errors", () => {
        const errorMessage = "Image dimensions too large: 2048x2048 exceeds maximum of 1024x1024";
        const decision = simulateRetryDecision(makeAssistantError({ errorMessage }));
        expect(decision.isRetryable).toBe(false);
        expect(decision.reason).toBeNull();
      });

      it("does not retry image size errors", () => {
        const errorMessage = "Image file size 25MB exceeds maximum of 20MB";
        const decision = simulateRetryDecision(makeAssistantError({ errorMessage }));
        expect(decision.isRetryable).toBe(false);
      });

      it("does not retry unknown errors", () => {
        const errorMessage = "Something unexpected happened";
        const decision = simulateRetryDecision(makeAssistantError({ errorMessage }));
        expect(decision.isRetryable).toBe(false);
        expect(decision.reason).toBeNull();
      });
    });

    describe("assistant message stopReason validation", () => {
      it("does not failover when stopReason is not 'error'", () => {
        const assistant = makeAssistantError({
          errorMessage: "rate limit exceeded",
          stopReason: "stop",
        });
        expect(isFailoverAssistantError(assistant)).toBe(false);
      });

      it("does not failover for undefined message", () => {
        expect(isFailoverAssistantError(undefined)).toBe(false);
      });

      it("does not failover for non-failover error message", () => {
        const assistant = makeAssistantError({
          errorMessage: "some random error",
        });
        expect(isFailoverAssistantError(assistant)).toBe(false);
      });
    });
  });
});
