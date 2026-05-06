import React, { useState, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import { listTraces, type TraceMeta } from "../bridge/trace-loader.js";

interface TraceBrowserProps {
  tracesDir: string;
  onSelect: (taskId: string) => void;
  onExit: () => void;
}

export function TraceBrowser({ tracesDir, onSelect, onExit }: TraceBrowserProps): React.ReactElement {
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
      setSelected((s) => Math.min(traces.length - 1, s + 1));
    } else if (key.return && traces.length > 0) {
      onSelect(traces[selected].task_id);
    }
  });

  if (loading) {
    return (
      <Box>
        <Text dimColor>Loading traces...</Text>
      </Box>
    );
  }

  if (traces.length === 0) {
    return (
      <Box flexDirection="column">
        <Text bold>Previous Runs</Text>
        <Text dimColor>No traces found in {tracesDir}</Text>
        <Box marginTop={1}>
          <Text dimColor>Esc to go back</Text>
        </Box>
      </Box>
    );
  }

  const visibleCount = 20;
  const startIdx = Math.max(0, selected - visibleCount + 5);
  const visible = traces.slice(startIdx, startIdx + visibleCount);

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold>Previous Runs</Text>
        <Text dimColor>{" "}({traces.length} traces)</Text>
      </Box>

      {visible.map((trace, i) => {
        const globalIdx = startIdx + i;
        const isSelected = globalIdx === selected;
        const date = new Date(trace.started);
        const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
        const timeStr = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
        const duration = trace.finished
          ? Math.round((new Date(trace.finished).getTime() - date.getTime()) / 1000)
          : 0;
        const durationStr = duration > 60 ? `${Math.floor(duration / 60)}m${duration % 60}s` : `${duration}s`;
        const statusColor = trace.status === "success" ? "green" : trace.status === "failed" ? "red" : "yellow";
        const statusIcon = trace.status === "success" ? "✓" : trace.status === "failed" ? "✗" : "○";

        return (
          <Box key={trace.task_id}>
            <Text color={isSelected ? "cyan" : undefined}>
              {isSelected ? "❯ " : "  "}
            </Text>
            <Text color={statusColor}>{statusIcon} </Text>
            <Text dimColor>{dateStr} {timeStr} </Text>
            <Text bold={isSelected}>{trace.task_name || trace.task_id}</Text>
            <Text dimColor>{" "}({durationStr}, {trace.events} events)</Text>
          </Box>
        );
      })}

      <Box marginTop={1}>
        <Text dimColor>↑↓ navigate, Enter to view, Esc to go back</Text>
      </Box>
    </Box>
  );
}
