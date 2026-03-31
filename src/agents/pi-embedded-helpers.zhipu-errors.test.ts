import { describe, expect, it } from "vitest";
import {
  classifyFailoverReason,
  isRateLimitErrorMessage,
  isTimeoutErrorMessage,
} from "./pi-embedded-helpers.js";

describe("Zhipu AI error detection", () => {
  describe("rate limit errors", () => {
    it("should detect LLM error 1302 rate limit", () => {
      const message =
        "LLM error 1302: 您的账户已达到速率限制，请您控制请求频率 (request_id: 20260320173545bdd82da5bd37400f)";
      expect(isRateLimitErrorMessage(message)).toBe(true);
      expect(classifyFailoverReason(message)).toBe("rate_limit");
    });

    it("should detect rate limit in Chinese", () => {
      const message = "您的账户已达到速率限制，请您控制请求频率";
      expect(isRateLimitErrorMessage(message)).toBe(true);
      expect(classifyFailoverReason(message)).toBe("rate_limit");
    });

    it("should detect request frequency control message", () => {
      const message = "请您控制请求频率";
      expect(isRateLimitErrorMessage(message)).toBe(true);
    });

    it("should detect JSON format rate limit error with code 1302", () => {
      const message =
        '{"error":{"code":"1302","message":"您的账户已达到速率限制，请您控制请求频率"},"request_id":"20260331155218c06708d0bc0445f5"}';
      expect(isRateLimitErrorMessage(message)).toBe(true);
      expect(classifyFailoverReason(message)).toBe("rate_limit");
    });

    it("should detect JSON format with code 1302 even without Chinese text", () => {
      const message =
        '{"error":{"code":"1302","message":"Rate limit exceeded"},"request_id":"test123"}';
      expect(isRateLimitErrorMessage(message)).toBe(true);
      expect(classifyFailoverReason(message)).toBe("rate_limit");
    });
  });

  describe("network and timeout errors", () => {
    it("should detect LLM error 1234 network error", () => {
      const message =
        "LLM error 1234: 网络错误，错误id：20260320031740b46136f53d4c4087，请联系客服。 (request_id: 20260320031740b46136f53d4c4087)";
      expect(isTimeoutErrorMessage(message)).toBe(true);
      expect(classifyFailoverReason(message)).toBe("timeout");
    });

    it("should detect LLM error api_error Internal Network Failure", () => {
      const message =
        "LLM error api_error: Internal Network Failure (request_id: 20260319180423e2bde364d1044614)";
      expect(isTimeoutErrorMessage(message)).toBe(true);
      expect(classifyFailoverReason(message)).toBe("timeout");
    });

    it("should detect network error in Chinese", () => {
      const message = "网络错误，请稍后重试";
      expect(isTimeoutErrorMessage(message)).toBe(true);
    });

    it("should detect internal network failure", () => {
      const message = "Internal Network Failure";
      expect(isTimeoutErrorMessage(message)).toBe(true);
    });
  });
});
