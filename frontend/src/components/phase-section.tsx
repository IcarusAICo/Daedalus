import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import type { PhaseStatus } from "../store/types.js";

interface PhaseSectionProps {
  title: string;
  status: PhaseStatus;
  progress?: { current: number; total: number };
  summary?: string;
  children?: React.ReactNode;
}

const STATUS_ICONS: Record<PhaseStatus, string> = {
  pending: "○",
  running: "●",
  complete: "✓",
  failed: "✗",
  skipped: "—",
};

const STATUS_COLORS: Record<PhaseStatus, string> = {
  pending: "gray",
  running: "cyan",
  complete: "green",
  failed: "red",
  skipped: "gray",
};

export function PhaseSection({
  title,
  status,
  progress,
  summary,
  children,
}: PhaseSectionProps): React.ReactElement | null {
  if (status === "pending") return null;

  const color = STATUS_COLORS[status];
  const icon = STATUS_ICONS[status];
  const progressStr = progress ? ` ${progress.current}/${progress.total}` : "";
  const isExpanded = status === "running";

  return (
    <Box flexDirection="column" marginY={0} paddingX={1}>
      <Box>
        <Text color={color} bold>
          {status === "running" ? (
            <Spinner type="dots" />
          ) : (
            icon
          )}
        </Text>
        <Text bold color={color}>
          {" "}
          {title}
        </Text>
        {progressStr && <Text dimColor>{progressStr}</Text>}
        {status === "running" && summary && (
          <Text dimColor> — {summary}</Text>
        )}
        {status === "complete" && summary && (
          <Text dimColor> — {summary}</Text>
        )}
        {status === "failed" && summary && (
          <Text color="red"> — {summary}</Text>
        )}
        {status === "skipped" && (
          <Text dimColor> (skipped)</Text>
        )}
      </Box>

      {isExpanded && children && (
        <Box
          flexDirection="column"
          paddingLeft={2}
          borderStyle="single"
          borderColor="gray"
          borderLeft={true}
          borderRight={false}
          borderTop={false}
          borderBottom={false}
          marginLeft={1}
        >
          {children}
        </Box>
      )}

      {!isExpanded && status === "complete" && children && (
        <Box
          flexDirection="column"
          paddingLeft={2}
          marginLeft={1}
        >
          {children}
        </Box>
      )}
    </Box>
  );
}
