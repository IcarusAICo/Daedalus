import React, { useState, useEffect, useMemo, useRef } from "react";
import { Box, Text, useInput } from "ink";
import Spinner from "ink-spinner";
import { existsSync } from "node:fs";
import type { ChatMessage } from "../store/types.js";
import { useAgentStore } from "../store/agent-store.js";
import { ScrollList, type ScrollListItem, deriveFirstVisible, clampFirstVisible } from "./scroll-list.js";
import { useMeasuredBox, padEnd, visualWidth, wrapLines } from "../utils/term.js";

const MAX_THINKING_LINES = 6;
const MAX_STATUS_LINES = 3;
const MAX_RESULT_LINES = 8;
const MAX_ARG_LINES = 4;
const MAX_FEEDBACK_LINES = 14;
const IMAGE_HEIGHT = 22;

export function ChatFeed(): React.ReactElement {
  const chatMessages = useAgentStore((s) => s.chatMessages);
  const pendingConfirm = useAgentStore((s) => s.pendingConfirm);
  const currentPhase = useAgentStore((s) => s.currentPhase);
  const phaseStatus = useAgentStore((s) => s.phases[s.currentPhase]?.status);

  // Measure available space within the parent layout (ChatFeed sits next to
  // a variable-height GoalDisplay / spinner row that appears during the
  // planner/evaluator phases).
  const { ref, size } = useMeasuredBox({ rows: 20, cols: 80 }, [
    chatMessages.length,
    !!pendingConfirm,
    currentPhase,
    phaseStatus,
  ]);

  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [firstVisible, setFirstVisible] = useState<number>(0);
  const [pinToBottom, setPinToBottom] = useState(true);
  const prevCountRef = useRef(chatMessages.length);

  const cols = Math.max(20, size.cols);
  const rows = Math.max(3, size.rows);

  const items: ScrollListItem[] = useMemo(
    () => chatMessages.map((msg, i) => buildItem(msg, i, cols, expandedIds, selectedIdx)),
    [chatMessages, cols, expandedIds, selectedIdx]
  );
  // Heights for scroll math. Zero means the row is skipped entirely (e.g.
  // empty thinking text or pending phase rows).
  const heights = useMemo(
    () => items.map((it) => {
      const h = it.height ?? 1;
      return h === 0 ? 0 : Math.max(1, h);
    }),
    [items]
  );

  // Reserve 2 rows in ScrollList for top/bottom indicators.
  const contentRows = Math.max(0, rows - 2);

  // When new messages arrive and we're pinned to bottom, jump first-visible
  // to the end so the latest message stays in view.
  useEffect(() => {
    if (chatMessages.length > prevCountRef.current && pinToBottom) {
      const lastIdx = chatMessages.length - 1;
      const fv = deriveFirstVisible(heights, lastIdx, contentRows);
      setFirstVisible(fv);
    }
    prevCountRef.current = chatMessages.length;
  }, [chatMessages.length, pinToBottom, heights, contentRows]);

  // Re-clamp on viewport resize so rows aren't left dangling.
  useEffect(() => {
    setFirstVisible((fv) => clampFirstVisible(fv, heights, contentRows));
  }, [heights, contentRows]);

  useInput((input, key) => {
    if (pendingConfirm) return;

    const expandableIndices = chatMessages
      .map((m, i) => (m.kind === "tool_call" ? i : -1))
      .filter((i) => i >= 0);

    if (key.pageUp || (key.upArrow && key.shift)) {
      setFirstVisible((fv) => Math.max(0, fv - contentRows));
      setPinToBottom(false);
      return;
    }
    if (key.pageDown || (key.downArrow && key.shift)) {
      const maxFv = computeMaxFirstVisible(heights, contentRows);
      setFirstVisible((fv) => {
        const next = Math.min(maxFv, fv + contentRows);
        if (next >= maxFv) setPinToBottom(true);
        return next;
      });
      return;
    }
    if (input === "g" && !key.shift) {
      setFirstVisible(0);
      setPinToBottom(false);
      return;
    }
    if (input === "G" || key.end) {
      const maxFv = computeMaxFirstVisible(heights, contentRows);
      setFirstVisible(maxFv);
      setPinToBottom(true);
      return;
    }

    if (key.upArrow) {
      if (expandableIndices.length === 0) return;
      let nextIdx: number;
      if (selectedIdx < 0) {
        nextIdx = expandableIndices[expandableIndices.length - 1];
      } else {
        const curPos = expandableIndices.indexOf(selectedIdx);
        if (curPos > 0) nextIdx = expandableIndices[curPos - 1];
        else nextIdx = expandableIndices[0];
      }
      setSelectedIdx(nextIdx);
      setFirstVisible((fv) => {
        const desired = deriveFirstVisible(heights, nextIdx, contentRows);
        return Math.min(fv, desired);
      });
      setPinToBottom(false);
      return;
    }
    if (key.downArrow) {
      if (selectedIdx < 0) {
        const maxFv = computeMaxFirstVisible(heights, contentRows);
        setFirstVisible((fv) => {
          const next = Math.min(maxFv, fv + 1);
          if (next >= maxFv) setPinToBottom(true);
          return next;
        });
        return;
      }
      const curPos = expandableIndices.indexOf(selectedIdx);
      if (curPos >= 0 && curPos < expandableIndices.length - 1) {
        const nextIdx = expandableIndices[curPos + 1];
        setSelectedIdx(nextIdx);
        setFirstVisible((fv) => {
          const desired = deriveFirstVisible(heights, nextIdx, contentRows);
          return Math.max(fv, desired);
        });
      } else {
        setSelectedIdx(-1);
        const maxFv = computeMaxFirstVisible(heights, contentRows);
        setFirstVisible(maxFv);
        setPinToBottom(true);
      }
      return;
    }
    if ((key.return || input === " ") && selectedIdx >= 0) {
      const msg = chatMessages[selectedIdx];
      if (msg && msg.kind === "tool_call") {
        setExpandedIds((prev) => {
          const next = new Set(prev);
          if (next.has(msg.id)) next.delete(msg.id);
          else next.add(msg.id);
          return next;
        });
        // After expansion the row is much taller; re-anchor scroll so the
        // selected row stays visible from the top of the viewport.
        setFirstVisible(selectedIdx);
        setPinToBottom(false);
      }
      return;
    }
    if (key.escape) {
      setSelectedIdx(-1);
      return;
    }
  });

  const safeFirstVisible = clampFirstVisible(firstVisible, heights, contentRows);

  return (
    <Box ref={ref} flexDirection="column" flexGrow={1} flexShrink={1} overflow="hidden" width="100%">
      <ScrollList
        items={items}
        selectedIndex={selectedIdx}
        firstVisible={safeFirstVisible}
        viewportRows={rows}
        viewportCols={cols}
      />
    </Box>
  );
}

function computeMaxFirstVisible(heights: number[], contentRows: number): number {
  if (heights.length === 0 || contentRows <= 0) return 0;
  // The largest valid firstVisible such that the items from firstVisible
  // through the end still fill `contentRows` (so the bottom indicator points
  // to no further content).
  let used = 0;
  for (let i = heights.length - 1; i >= 0; i--) {
    if (used + heights[i] > contentRows) return i + 1;
    used += heights[i];
  }
  return 0;
}

function buildItem(
  msg: ChatMessage,
  idx: number,
  cols: number,
  expandedIds: Set<string>,
  selectedIdx: number
): ScrollListItem {
  const selected = idx === selectedIdx;
  const expanded = expandedIds.has(msg.id);

  switch (msg.kind) {
    case "thinking": {
      const lines = wrapLines((msg.text || "").trimEnd(), Math.max(1, cols - 2), MAX_THINKING_LINES);
      if (lines.length === 0) {
        return { key: msg.id, height: 0, render: () => null };
      }
      return {
        key: msg.id,
        height: lines.length,
        render: ({ cols: c }) => <PaddedLines lines={lines} cols={c} indent={2} />,
      };
    }
    case "status": {
      const lines = wrapLines(msg.text || "", Math.max(1, cols - 2), MAX_STATUS_LINES);
      if (lines.length === 0) {
        return { key: msg.id, height: 1, render: ({ cols: c }) => <BlankRow cols={c} /> };
      }
      return {
        key: msg.id,
        height: lines.length,
        render: ({ cols: c }) => <PaddedLines lines={lines} cols={c} indent={2} />,
      };
    }
    case "error": {
      const text = "  \u2717 " + (msg.text || "");
      const lines = wrapLines(text, Math.max(1, cols), 2);
      return {
        key: msg.id,
        height: Math.max(1, lines.length),
        render: ({ cols: c }) => <ColoredLines lines={lines} cols={c} color="red" />,
      };
    }
    case "phase": {
      if (msg.phaseStatus === "pending") {
        return { key: msg.id, height: 0, render: () => null };
      }
      if (msg.phaseStatus === "running" && msg.id.startsWith("sep-")) {
        return { key: msg.id, height: 0, render: () => null };
      }
      return {
        key: msg.id,
        height: 1,
        render: ({ cols: c }) => <PhaseRow msg={msg} cols={c} />,
      };
    }
    case "tool_call": {
      return buildToolCallItem(msg, cols, expanded, selected);
    }
    case "learner_feedback": {
      return buildFeedbackItem(msg, cols);
    }
    default:
      return { key: msg.id, height: 0, render: () => null };
  }
}

function buildToolCallItem(
  msg: ChatMessage,
  cols: number,
  expanded: boolean,
  selected: boolean
): ScrollListItem {
  const headerHeight = 1;
  let bodyHeight = 0;
  let argLines: string[] = [];
  let resultLines: string[] = [];
  let imagePresent = false;
  let imagePathLine = "";
  if (expanded) {
    if (msg.toolArgs && Object.keys(msg.toolArgs).length > 0) {
      const argText = "Args: " + JSON.stringify(msg.toolArgs, null, 2).slice(0, 600);
      argLines = wrapLines(argText, Math.max(1, cols - 4), MAX_ARG_LINES);
      bodyHeight += argLines.length;
    }
    if (msg.toolResult) {
      const resText = "Result: " + msg.toolResult.slice(0, 800);
      resultLines = wrapLines(resText, Math.max(1, cols - 4), MAX_RESULT_LINES);
      bodyHeight += resultLines.length;
    }
    if (msg.toolImagePath && existsSync(msg.toolImagePath)) {
      imagePresent = true;
      imagePathLine = "Screenshot: " + msg.toolImagePath;
      bodyHeight += 1; // "Screenshot: ..." line
      bodyHeight += IMAGE_HEIGHT;
    }
  }

  const height = headerHeight + bodyHeight;
  return {
    key: msg.id,
    height,
    render: ({ cols: c }) => (
      <ToolCallRow
        msg={msg}
        cols={c}
        expanded={expanded}
        selected={selected}
        argLines={argLines}
        resultLines={resultLines}
        imagePresent={imagePresent}
        imagePathLine={imagePathLine}
      />
    ),
  };
}

function buildFeedbackItem(msg: ChatMessage, cols: number): ScrollListItem {
  let data: Record<string, unknown> | null = null;
  try {
    data = JSON.parse(msg.text || "{}") as Record<string, unknown>;
  } catch {
    return { key: msg.id, height: 0, render: () => null };
  }

  const summary = (data.summary as string) || "";
  const failurePoint = (data.failure_point as string) || "";
  const suggestions = (data.suggestions as Array<Record<string, unknown>>) ?? [];
  const newSkills = (data.new_skill_candidates as Array<Record<string, unknown>>) ?? [];

  const innerCols = Math.max(1, cols - 4);
  const lines: { text: string; color?: string; bold?: boolean; dim?: boolean }[] = [];
  lines.push({ text: "Learner Diagnosis", color: "magenta", bold: true });
  if (summary) {
    for (const l of wrapLines(summary, innerCols, 3)) lines.push({ text: l });
  }
  if (failurePoint) {
    for (const l of wrapLines("Failure: " + failurePoint, innerCols, 2)) {
      lines.push({ text: l, color: "red" });
    }
  }
  if (suggestions.length > 0) {
    lines.push({ text: "Suggestions:", dim: true, bold: true });
    for (const s of suggestions.slice(0, 3)) {
      const cat = (s.category as string) || "";
      const desc = (s.description as string) || "";
      for (const l of wrapLines(`\u2022 [${cat}] ${desc}`, innerCols, 2)) {
        lines.push({ text: l });
      }
    }
  }
  if (newSkills.length > 0) {
    lines.push({ text: "Proposed skills:", dim: true, bold: true });
    for (const s of newSkills.slice(0, 3)) {
      const id = (s.proposed_id as string) || "";
      const desc = (s.description as string) || "";
      for (const l of wrapLines(`\u2022 ${id} \u2014 ${desc}`, innerCols, 2)) {
        lines.push({ text: l });
      }
    }
  }

  const capped = lines.slice(0, MAX_FEEDBACK_LINES);
  return {
    key: msg.id,
    height: capped.length,
    render: ({ cols: c }) => <FeedbackRows lines={capped} cols={c} />,
  };
}

function PaddedLines({
  lines,
  cols,
  indent,
}: {
  lines: string[];
  cols: number;
  indent: number;
}): React.ReactElement {
  const prefix = " ".repeat(Math.max(0, indent));
  return (
    <>
      {lines.map((line, i) => (
        <Text key={i}>{padEnd(prefix + line, cols)}</Text>
      ))}
    </>
  );
}

function ColoredLines({
  lines,
  cols,
  color,
}: {
  lines: string[];
  cols: number;
  color: string;
}): React.ReactElement {
  return (
    <>
      {lines.map((line, i) => (
        <Box key={i} width={cols} height={1} flexShrink={0}>
          <Text color={color}>{line}</Text>
          <Text>{" ".repeat(Math.max(0, cols - visualWidth(line)))}</Text>
        </Box>
      ))}
    </>
  );
}

function BlankRow({ cols }: { cols: number }): React.ReactElement {
  return <Text>{" ".repeat(cols)}</Text>;
}

function PhaseRow({ msg, cols }: { msg: ChatMessage; cols: number }): React.ReactElement {
  if (msg.phaseStatus === "running") {
    const summary = msg.phaseSummary ? ` - ${msg.phaseSummary}` : "";
    const text = `  ${msg.phase}${summary}`;
    return (
      <Box width={cols} height={1} flexShrink={0}>
        <Text color="cyan"><Spinner type="dots" /></Text>
        <Text color="cyan" bold>{` ${msg.phase}`}</Text>
        {summary && <Text dimColor>{summary}</Text>}
        <Text>{" ".repeat(Math.max(0, cols - visualWidth(text) - 1))}</Text>
      </Box>
    );
  }
  const statusIcon =
    msg.phaseStatus === "complete" ? "\u2713" :
    msg.phaseStatus === "failed" ? "\u2717" :
    msg.phaseStatus === "skipped" ? "\u2014" : "\u25CB";
  const color =
    msg.phaseStatus === "complete" ? "green" :
    msg.phaseStatus === "failed" ? "red" : "gray";
  const summaryText = msg.phaseSummary ? ` - ${msg.phaseSummary}` : "";
  const fullText = `  ${statusIcon} ${msg.phase}${summaryText}`;
  return (
    <Box width={cols} height={1} flexShrink={0}>
      <Text color={color} bold>{`  ${statusIcon} ${msg.phase}`}</Text>
      {summaryText && <Text dimColor>{summaryText}</Text>}
      <Text>{" ".repeat(Math.max(0, cols - visualWidth(fullText)))}</Text>
    </Box>
  );
}

interface ToolCallRowProps {
  msg: ChatMessage;
  cols: number;
  expanded: boolean;
  selected: boolean;
  argLines: string[];
  resultLines: string[];
  imagePresent: boolean;
  imagePathLine: string;
}

function ToolCallRow({
  msg,
  cols,
  expanded,
  selected,
  argLines,
  resultLines,
  imagePresent,
  imagePathLine,
}: ToolCallRowProps): React.ReactElement {
  const isRunning = msg.toolStatus === "running";
  const isError = msg.toolStatus === "error";
  const caret = expanded ? "\u25BE" : "\u25B8";
  const selIndicator = selected ? "\u203A" : " ";

  // Header line
  const argSummary = !expanded && msg.toolArgs && Object.keys(msg.toolArgs).length > 0
    ? " " + formatArgs(msg.toolArgs)
    : "";
  const resultSummary = !expanded && !isRunning && msg.toolResult
    ? " \u2192 " + truncateInline(msg.toolResult.replace(/\n/g, " "), 40)
    : "";
  const imageMarker = !expanded && msg.toolImagePath ? " \u{1F5BC}" : "";

  const headerVisible = `${selIndicator} ${caret} ${msg.toolName ?? ""}${argSummary}${resultSummary}${imageMarker}`;
  const headerTrailing = " ".repeat(Math.max(0, cols - visualWidth(headerVisible)));

  // Image rendering: load text once per expand
  const [imageText, setImageText] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  useEffect(() => {
    if (!imagePresent || !msg.toolImagePath) {
      setImageText(null);
      setImageLoading(false);
      return;
    }
    setImageLoading(true);
    let cancelled = false;
    import("terminal-image")
      .then((ti) =>
        ti.default
          .file(msg.toolImagePath!, { width: Math.min(60, Math.max(20, cols - 8)), height: IMAGE_HEIGHT })
          .then((text: string) => {
            if (cancelled) return;
            setImageText(text && text.trim().length > 0 ? text : null);
            setImageLoading(false);
          })
      )
      .catch(() => {
        if (cancelled) return;
        setImageText(null);
        setImageLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [imagePresent, msg.toolImagePath, cols]);

  return (
    <>
      <Box width={cols} height={1} flexShrink={0}>
        <Text color={selected ? "yellow" : undefined}>{`${selIndicator} `}</Text>
        {isRunning ? (
          <Text color="cyan"><Spinner type="dots" />{" "}</Text>
        ) : isError ? (
          <Text color="red">{"\u2717 "}</Text>
        ) : (
          <Text color="green">{`${caret} `}</Text>
        )}
        <Text color="cyan" bold>{msg.toolName ?? ""}</Text>
        {argSummary && <Text dimColor>{argSummary}</Text>}
        {resultSummary && <Text dimColor>{resultSummary}</Text>}
        {imageMarker && <Text dimColor>{imageMarker}</Text>}
        <Text>{headerTrailing}</Text>
      </Box>
      {expanded &&
        argLines.map((line, i) => (
          <Box key={`arg-${i}`} width={cols} height={1} flexShrink={0}>
            <Text dimColor>{padEnd("    " + line, cols)}</Text>
          </Box>
        ))}
      {expanded &&
        resultLines.map((line, i) => (
          <Box key={`res-${i}`} width={cols} height={1} flexShrink={0}>
            <Text>{padEnd("    " + line, cols)}</Text>
          </Box>
        ))}
      {expanded && imagePresent && (
        <>
          <Box width={cols} height={1} flexShrink={0}>
            <Text dimColor>{padEnd("    " + imagePathLine, cols)}</Text>
          </Box>
          <ImageBlock
            cols={cols}
            height={IMAGE_HEIGHT}
            text={imageText}
            loading={imageLoading}
          />
        </>
      )}
    </>
  );
}

function ImageBlock({
  cols,
  height,
  text,
  loading,
}: {
  cols: number;
  height: number;
  text: string | null;
  loading: boolean;
}): React.ReactElement {
  // Always render exactly `height` rows so the layout never shifts.
  const lines = text ? text.split(/\r?\n/) : [];
  const out: React.ReactElement[] = [];
  for (let i = 0; i < height; i++) {
    if (i === 0 && loading) {
      out.push(
        <Box key={`img-${i}`} width={cols} height={1} flexShrink={0}>
          <Text color="cyan"><Spinner type="dots" /></Text>
          <Text dimColor>{padEnd(" loading image...", cols - 1)}</Text>
        </Box>
      );
      continue;
    }
    const raw = lines[i] ?? "";
    out.push(
      <Box key={`img-${i}`} width={cols} height={1} flexShrink={0}>
        <Text>{raw}</Text>
        <Text>{" ".repeat(Math.max(0, cols - visualWidth(raw)))}</Text>
      </Box>
    );
  }
  return <>{out}</>;
}

function FeedbackRows({
  lines,
  cols,
}: {
  lines: { text: string; color?: string; bold?: boolean; dim?: boolean }[];
  cols: number;
}): React.ReactElement {
  return (
    <>
      {lines.map((line, i) => {
        const indented = "  " + line.text;
        const trailing = " ".repeat(Math.max(0, cols - visualWidth(indented)));
        return (
          <Box key={i} width={cols} height={1} flexShrink={0}>
            <Text color={line.color} bold={line.bold} dimColor={line.dim}>
              {indented}
            </Text>
            <Text>{trailing}</Text>
          </Box>
        );
      })}
    </>
  );
}

function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  const parts = entries.slice(0, 3).map(([k, v]) => {
    if (typeof v === "string") return `${k}="${truncateInline(v, 20)}"`;
    if (typeof v === "number" || typeof v === "boolean") return `${k}=${v}`;
    return `${k}=\u2026`;
  });
  if (entries.length > 3) parts.push("\u2026");
  return parts.join(" ");
}

function truncateInline(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "\u2026";
}
