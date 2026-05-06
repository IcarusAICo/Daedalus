import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import type { ToolCall } from "../store/types.js";

interface ToolCallListProps {
  calls: ToolCall[];
  maxVisible?: number;
}

export function ToolCallList({
  calls,
  maxVisible = 15,
}: ToolCallListProps): React.ReactElement | null {
  if (calls.length === 0) return null;

  const visible = calls.slice(-maxVisible);
  const hidden = calls.length - visible.length;

  return (
    <Box flexDirection="column">
      {hidden > 0 && (
        <Text dimColor>
          ··· {hidden} earlier call{hidden !== 1 ? "s" : ""} hidden
        </Text>
      )}
      {visible.map((tc) => (
        <Box key={tc.id}>
          {tc.status === "running" ? (
            <Text color="cyan">
              <Spinner type="dots" />{" "}
            </Text>
          ) : tc.status === "success" ? (
            <Text color="green">{"▸ "}</Text>
          ) : (
            <Text color="red">{"✗ "}</Text>
          )}
          <Text color="cyan" bold>
            {tc.name}
          </Text>
          {hasArgs(tc.args) && (
            <Text dimColor>
              {" "}
              {formatArgs(tc.args)}
            </Text>
          )}
          {tc.status === "success" && tc.result && (
            <Text color="gray">
              {" → "}
              {formatResult(tc.result)}
            </Text>
          )}
          {tc.status === "error" && tc.result && (
            <Text color="red">
              {" → "}
              {truncate(tc.result, 60)}
            </Text>
          )}
          {tc.completedAt && tc.startedAt && (
            <Text dimColor>
              {" "}
              ({formatDuration(tc.completedAt, tc.startedAt)})
            </Text>
          )}
        </Box>
      ))}
    </Box>
  );
}

function hasArgs(args: Record<string, unknown>): boolean {
  return Object.keys(args).length > 0;
}

function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  const parts = entries.slice(0, 3).map(([k, v]) => {
    if (typeof v === "string") {
      return `${k}="${truncate(v, 25)}"`;
    }
    if (typeof v === "number" || typeof v === "boolean") {
      return `${k}=${v}`;
    }
    return `${k}=…`;
  });
  if (entries.length > 3) parts.push("…");
  return parts.join(" ");
}

function formatResult(result: string): string {
  const clean = result.replace(/\s+/g, " ").trim();
  return truncate(clean, 50);
}

function formatDuration(end: string, start: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}
