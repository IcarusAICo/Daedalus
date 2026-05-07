import React, { useState, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import { listTraces, type TraceMeta } from "../bridge/trace-loader.js";
import { ScrollList, type ScrollListItem } from "./scroll-list.js";
import { padEnd, visualWidth } from "../utils/term.js";

interface TraceBrowserProps {
  tracesDir: string;
  onSelect: (taskId: string) => void;
  onExit: () => void;
  viewportRows: number;
  viewportCols: number;
}

export function TraceBrowser({
  tracesDir,
  onSelect,
  onExit,
  viewportRows,
  viewportCols,
}: TraceBrowserProps): React.ReactElement {
  const [traces, setTraces] = useState<TraceMeta[]>([]);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const metas = listTraces(tracesDir);
    setTraces(metas);
    setLoading(false);
  }, [tracesDir]);

  useInput((_input, key) => {
    if (key.escape) {
      onExit();
    } else if (key.upArrow) {
      setSelected((s) => Math.max(0, s - 1));
    } else if (key.downArrow) {
      setSelected((s) => Math.min(Math.max(0, traces.length - 1), s + 1));
    } else if (key.return && traces.length > 0) {
      const t = traces[selected];
      onSelect(t.run_id || t.task_id || "");
    }
  });

  if (loading) {
    return (
      <Box flexDirection="column" width={viewportCols} height={viewportRows} flexShrink={0}>
        <Text dimColor>{padEnd("Loading traces...", viewportCols)}</Text>
      </Box>
    );
  }

  if (traces.length === 0) {
    return (
      <Box flexDirection="column" width={viewportCols} height={viewportRows} flexShrink={0}>
        <Text bold>{padEnd("Previous Runs", viewportCols)}</Text>
        <Text dimColor>{padEnd(`No traces found in ${tracesDir}`, viewportCols)}</Text>
        <Text>{padEnd("", viewportCols)}</Text>
        <Text dimColor>{padEnd("Esc to go back", viewportCols)}</Text>
      </Box>
    );
  }

  // Reserve 1 row for header and 1 row for footer help text. The remaining
  // rows are owned by ScrollList (which itself reserves 2 indicator rows).
  const HEADER_ROWS = 1;
  const FOOTER_ROWS = 1;
  const listRows = Math.max(0, viewportRows - HEADER_ROWS - FOOTER_ROWS);

  const items: ScrollListItem[] = traces.map((trace, i) => ({
    key: trace.run_id || trace.task_id || `t-${i}`,
    height: 1,
    render: ({ selected: isSel, cols }) => (
      <TraceRow trace={trace} selected={isSel} cols={cols} />
    ),
  }));

  return (
    <Box flexDirection="column" width={viewportCols} height={viewportRows} flexShrink={0}>
      <Text bold>
        {padEnd(`Previous Runs  (${traces.length} trace${traces.length === 1 ? "" : "s"})`, viewportCols)}
      </Text>
      <ScrollList
        items={items}
        selectedIndex={selected}
        viewportRows={listRows}
        viewportCols={viewportCols}
      />
      <Text dimColor>{padEnd("\u2191\u2193 navigate, Enter to view, Esc to go back", viewportCols)}</Text>
    </Box>
  );
}

function TraceRow({
  trace,
  selected,
  cols,
}: {
  trace: TraceMeta;
  selected: boolean;
  cols: number;
}): React.ReactElement {
  const date = new Date(trace.started);
  const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const timeStr = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  const duration = trace.finished
    ? Math.round((new Date(trace.finished).getTime() - date.getTime()) / 1000)
    : 0;
  const durationStr = duration > 60 ? `${Math.floor(duration / 60)}m${duration % 60}s` : `${duration}s`;
  const statusColor = trace.status === "success" ? "green" : trace.status === "failed" ? "red" : "yellow";
  const statusIcon = trace.status === "success" ? "\u2713" : trace.status === "failed" ? "\u2717" : "\u25CB";

  const cursor = selected ? "\u276F " : "  ";
  const stamp = `${dateStr} ${timeStr} `;
  const tail = ` (${durationStr}${trace.events != null ? `, ${trace.events} events` : ""})`;
  const title = trace.goal || trace.task_name || trace.run_id || trace.task_id || "(untitled)";

  // Reserve space for cursor + status + stamp + tail; truncate title to fit.
  const fixed = visualWidth(cursor) + 2 /* status icon + space */ + visualWidth(stamp) + visualWidth(tail);
  const titleBudget = Math.max(0, cols - fixed);
  const titleClipped = clipPlain(title, titleBudget);

  const consumed = fixed + visualWidth(titleClipped);
  const trailing = " ".repeat(Math.max(0, cols - consumed));

  return (
    <Box width={cols} height={1} flexShrink={0}>
      <Text color={selected ? "cyan" : undefined}>{cursor}</Text>
      <Text color={statusColor}>{statusIcon} </Text>
      <Text dimColor>{stamp}</Text>
      <Text bold={selected}>{titleClipped}</Text>
      <Text dimColor>{tail}</Text>
      <Text>{trailing}</Text>
    </Box>
  );
}

function clipPlain(s: string, cols: number): string {
  if (cols <= 0) return "";
  const chars = [...s];
  if (chars.length <= cols) return s;
  if (cols === 1) return chars[0];
  return chars.slice(0, cols - 1).join("") + "\u2026";
}
