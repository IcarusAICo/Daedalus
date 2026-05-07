import React, { useState, useEffect, useRef } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import Spinner from "ink-spinner";
import type { ChatMessage } from "../store/types.js";
import { useAgentStore } from "../store/agent-store.js";
import { existsSync } from "node:fs";

export function ChatFeed(): React.ReactElement {
  const chatMessages = useAgentStore((s) => s.chatMessages);
  const pendingConfirm = useAgentStore((s) => s.pendingConfirm);
  const { stdout } = useStdout();
  const terminalHeight = stdout?.rows ?? 40;
  const viewportHeight = Math.max(5, terminalHeight - 8);
  const maxVisible = viewportHeight * 3;

  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [scrollOffset, setScrollOffset] = useState(0);
  const [pinToBottom, setPinToBottom] = useState(true);
  const prevCountRef = useRef(chatMessages.length);

  // Auto-scroll to bottom when new messages arrive (if pinned)
  useEffect(() => {
    if (chatMessages.length > prevCountRef.current && pinToBottom) {
      setScrollOffset(0);
    }
    prevCountRef.current = chatMessages.length;
  }, [chatMessages.length, pinToBottom]);

  useInput((input, key) => {
    if (pendingConfirm) return;

    // Page-Up: scroll back through history
    if (key.pageUp || (key.upArrow && key.shift)) {
      const maxOffset = Math.max(0, chatMessages.length - maxVisible);
      setScrollOffset((prev) => {
        const next = Math.min(prev + viewportHeight, maxOffset);
        if (next > 0) setPinToBottom(false);
        return next;
      });
      return;
    }

    // Page-Down: scroll forward toward latest
    if (key.pageDown || (key.downArrow && key.shift)) {
      setScrollOffset((prev) => {
        const next = Math.max(0, prev - viewportHeight);
        if (next === 0) setPinToBottom(true);
        return next;
      });
      return;
    }

    // 'g' or Home: jump to beginning
    if (input === "g" && !key.shift) {
      const maxOffset = Math.max(0, chatMessages.length - maxVisible);
      setScrollOffset(maxOffset);
      setPinToBottom(false);
      return;
    }

    // 'G' or End: jump to end
    if (input === "G" || key.end) {
      setScrollOffset(0);
      setPinToBottom(true);
      return;
    }

    if (key.upArrow) {
      if (selectedIdx < 0) {
        const expandableIndices = chatMessages
          .map((m, i) => (m.kind === "tool_call" ? i : -1))
          .filter((i) => i >= 0);
        if (expandableIndices.length > 0) {
          setSelectedIdx(expandableIndices[expandableIndices.length - 1]);
        }
      } else {
        const expandableIndices = chatMessages
          .map((m, i) => (m.kind === "tool_call" ? i : -1))
          .filter((i) => i >= 0);
        const curPos = expandableIndices.indexOf(selectedIdx);
        if (curPos > 0) {
          const nextIdx = expandableIndices[curPos - 1];
          setSelectedIdx(nextIdx);
          // Scroll up if selected item is above the visible window
          const currentEnd = chatMessages.length - scrollOffset;
          const currentStart = Math.max(0, currentEnd - maxVisible);
          if (nextIdx < currentStart) {
            setScrollOffset(chatMessages.length - nextIdx - maxVisible);
            setPinToBottom(false);
          }
        } else {
          // Already at the top expandable item — scroll up if there's more
          const maxOffset = Math.max(0, chatMessages.length - maxVisible);
          setScrollOffset((prev) => {
            const next = Math.min(prev + viewportHeight, maxOffset);
            if (next > 0) setPinToBottom(false);
            return next;
          });
        }
      }
    } else if (key.downArrow) {
      if (selectedIdx >= 0) {
        const expandableIndices = chatMessages
          .map((m, i) => (m.kind === "tool_call" ? i : -1))
          .filter((i) => i >= 0);
        const curPos = expandableIndices.indexOf(selectedIdx);
        if (curPos < expandableIndices.length - 1) {
          const nextIdx = expandableIndices[curPos + 1];
          setSelectedIdx(nextIdx);
          // Scroll down if selected item is below the visible window
          const currentEnd = chatMessages.length - scrollOffset;
          const currentStart = Math.max(0, currentEnd - maxVisible);
          if (nextIdx >= currentEnd) {
            const newOffset = Math.max(0, chatMessages.length - nextIdx - 1);
            setScrollOffset(newOffset);
            if (newOffset === 0) setPinToBottom(true);
          }
        } else {
          setSelectedIdx(-1);
          // Snap to bottom when deselecting past the last item
          setScrollOffset(0);
          setPinToBottom(true);
        }
      } else {
        // No selection but pressing down — scroll down if not at bottom
        if (scrollOffset > 0) {
          setScrollOffset((prev) => {
            const next = Math.max(0, prev - viewportHeight);
            if (next === 0) setPinToBottom(true);
            return next;
          });
        }
      }
    } else if ((key.return || input === " ") && selectedIdx >= 0) {
      const msg = chatMessages[selectedIdx];
      if (msg && msg.kind === "tool_call") {
        setExpandedIds((prev) => {
          const next = new Set(prev);
          if (next.has(msg.id)) next.delete(msg.id);
          else next.add(msg.id);
          return next;
        });
      }
    } else if (key.escape) {
      setSelectedIdx(-1);
    }
  });

  // Compute the visible window based on scroll offset
  const endIdx = chatMessages.length - scrollOffset;
  const startIdx = Math.max(0, endIdx - maxVisible);
  const visible = chatMessages.slice(startIdx, endIdx);
  const hiddenAbove = startIdx;
  const hiddenBelow = scrollOffset;

  return (
    <Box flexDirection="column" flexGrow={1} overflow="hidden">
      {hiddenAbove > 0 && (
        <Box>
          <Text dimColor>  ··· {hiddenAbove} earlier event{hiddenAbove !== 1 ? "s" : ""} (PgUp/Shift+↑ to scroll, g for top)</Text>
        </Box>
      )}
      {visible.map((msg, i) => {
        const globalIdx = startIdx + i;
        return (
          <ChatMessageRow
            key={msg.id}
            msg={msg}
            expanded={expandedIds.has(msg.id)}
            selected={globalIdx === selectedIdx}
          />
        );
      })}
      {hiddenBelow > 0 && (
        <Box>
          <Text dimColor>  ··· {hiddenBelow} newer event{hiddenBelow !== 1 ? "s" : ""} (PgDn/Shift+↓ to scroll, G for bottom)</Text>
        </Box>
      )}
    </Box>
  );
}

interface RowProps {
  msg: ChatMessage;
  expanded: boolean;
  selected: boolean;
}

function ChatMessageRow({ msg, expanded, selected }: RowProps): React.ReactElement | null {
  switch (msg.kind) {
    case "thinking":
      return <ThinkingRow msg={msg} />;
    case "tool_call":
      return <ToolCallRow msg={msg} expanded={expanded} selected={selected} />;
    case "phase":
      return <PhaseRow msg={msg} />;
    case "learner_feedback":
      return <LearnerFeedbackRow msg={msg} />;
    case "error":
      return (
        <Box>
          <Text color="red">{"  ✗ "}{msg.text}</Text>
        </Box>
      );
    case "status":
      return (
        <Box paddingLeft={2}>
          <Text wrap="wrap">{msg.text}</Text>
        </Box>
      );
    default:
      return null;
  }
}

function ThinkingRow({ msg }: { msg: ChatMessage }): React.ReactElement | null {
  const text = (msg.text || "").trimEnd();
  if (!text) return null;

  return (
    <Box flexDirection="column" paddingLeft={2}>
      <Text wrap="wrap">{text}</Text>
    </Box>
  );
}

function ToolCallRow({ msg, expanded, selected }: { msg: ChatMessage; expanded: boolean; selected: boolean }): React.ReactElement {
  const isRunning = msg.toolStatus === "running";
  const isError = msg.toolStatus === "error";
  const caret = expanded ? "▾" : "▸";
  const selIndicator = selected ? "›" : " ";
  const [imageText, setImageText] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);

  useEffect(() => {
    if (expanded && msg.toolImagePath && existsSync(msg.toolImagePath)) {
      setImageLoading(true);
      import("terminal-image").then((ti) =>
        ti.default.file(msg.toolImagePath!, { width: 60, height: 20 }).then((text: string) => {
          if (text && text.trim().length > 0) {
            setImageText(text);
          } else {
            setImageText(null);
          }
          setImageLoading(false);
        })
      ).catch(() => {
        setImageText(null);
        setImageLoading(false);
      });
    } else if (!expanded) {
      setImageText(null);
      setImageLoading(false);
    }
  }, [expanded, msg.toolImagePath]);

  return (
    <Box flexDirection="column">
      <Box>
        <Text color={selected ? "yellow" : undefined}>{selIndicator} </Text>
        {isRunning ? (
          <Text color="cyan"><Spinner type="dots" />{" "}</Text>
        ) : isError ? (
          <Text color="red">{"✗ "}</Text>
        ) : (
          <Text color="green">{caret} </Text>
        )}
        <Text color="cyan" bold>{msg.toolName}</Text>
        {!expanded && msg.toolArgs && Object.keys(msg.toolArgs).length > 0 && (
          <Text dimColor>{" "}{formatArgs(msg.toolArgs)}</Text>
        )}
        {!expanded && !isRunning && msg.toolResult && (
          <Text dimColor>{" → "}{truncate(msg.toolResult.replace(/\n/g, " "), 40)}</Text>
        )}
        {!expanded && msg.toolImagePath && (
          <Text dimColor>{" 🖼"}</Text>
        )}
      </Box>
      {expanded && (
        <Box flexDirection="column" paddingLeft={4}>
          {msg.toolArgs && Object.keys(msg.toolArgs).length > 0 && (
            <Box>
              <Text dimColor>Args: </Text>
              <Text>{JSON.stringify(msg.toolArgs, null, 2).slice(0, 200)}</Text>
            </Box>
          )}
          {msg.toolResult && (
            <Box>
              <Text dimColor>Result: </Text>
              <Text wrap="wrap">{msg.toolResult.slice(0, 300)}</Text>
            </Box>
          )}
          {msg.toolImagePath && (
            <Box flexDirection="column" marginTop={1}>
              <Text dimColor>Screenshot: <Text color="blue">{msg.toolImagePath}</Text></Text>
              {imageLoading && (
                <Box>
                  <Text color="cyan"><Spinner type="dots" /></Text>
                  <Text dimColor> loading image...</Text>
                </Box>
              )}
              {imageText && <Text>{imageText}</Text>}
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

function PhaseRow({ msg }: { msg: ChatMessage }): React.ReactElement | null {
  if (!msg.phase) return null;

  if (msg.phaseStatus === "pending") return null;

  // Sentinels from thinking_clear use "running" status but are just separators
  if (msg.phaseStatus === "running" && msg.id.startsWith("sep-")) return null;

  if (msg.phaseStatus === "running") {
    return (
      <Box paddingLeft={2}>
        <Text color="cyan"><Spinner type="dots" />{" "}</Text>
        <Text color="cyan" bold>{msg.phase}</Text>
        {msg.phaseSummary && (
          <Text dimColor>{" - "}{msg.phaseSummary}</Text>
        )}
      </Box>
    );
  }

  const statusIcon =
    msg.phaseStatus === "complete" ? "✓" :
    msg.phaseStatus === "failed" ? "✗" :
    msg.phaseStatus === "skipped" ? "—" : "○";

  const color =
    msg.phaseStatus === "complete" ? "green" :
    msg.phaseStatus === "failed" ? "red" : "gray";

  return (
    <Box paddingLeft={2}>
      <Text color={color} bold>{statusIcon} {msg.phase}</Text>
      {msg.phaseSummary && (
        <Text dimColor>{" - "}{msg.phaseSummary}</Text>
      )}
    </Box>
  );
}

function LearnerFeedbackRow({ msg }: { msg: ChatMessage }): React.ReactElement | null {
  let data: Record<string, unknown>;
  try {
    data = JSON.parse(msg.text || "{}");
  } catch {
    return null;
  }

  const summary = data.summary as string | undefined;
  const failurePoint = data.failure_point as string | undefined;
  const suggestions = (data.suggestions as Array<Record<string, unknown>>) ?? [];
  const newSkills = (data.new_skill_candidates as Array<Record<string, unknown>>) ?? [];

  return (
    <Box flexDirection="column" paddingLeft={2} marginY={1}>
      <Text color="magenta" bold>{"Learner Diagnosis"}</Text>
      {summary && (
        <Box paddingLeft={2}>
          <Text wrap="wrap">{summary}</Text>
        </Box>
      )}
      {failurePoint && (
        <Box paddingLeft={2}>
          <Text color="red" bold>Failure: </Text>
          <Text wrap="wrap">{failurePoint}</Text>
        </Box>
      )}
      {suggestions.length > 0 && (
        <Box flexDirection="column" paddingLeft={2}>
          <Text dimColor bold>Suggestions:</Text>
          {suggestions.slice(0, 5).map((s, i) => (
            <Box key={i} paddingLeft={1}>
              <Text dimColor>• </Text>
              <Text color="yellow">[{s.category as string}] </Text>
              <Text wrap="wrap">{s.description as string}</Text>
            </Box>
          ))}
        </Box>
      )}
      {newSkills.length > 0 && (
        <Box flexDirection="column" paddingLeft={2}>
          <Text dimColor bold>Proposed skills:</Text>
          {newSkills.map((s, i) => (
            <Box key={i} paddingLeft={1}>
              <Text dimColor>• </Text>
              <Text color="cyan">{s.proposed_id as string}</Text>
              <Text dimColor>{" — "}{s.description as string}</Text>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}

function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  const parts = entries.slice(0, 3).map(([k, v]) => {
    if (typeof v === "string") return `${k}="${truncate(v, 20)}"`;
    if (typeof v === "number" || typeof v === "boolean") return `${k}=${v}`;
    return `${k}=…`;
  });
  if (entries.length > 3) parts.push("…");
  return parts.join(" ");
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}
