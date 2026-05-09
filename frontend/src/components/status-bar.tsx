import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { useAgentStore } from "../store/agent-store.js";
import { useElapsedTime } from "../hooks/use-elapsed-time.js";

const PHASE_LABELS: Record<string, string> = {
  idle: "Idle",
  explorer: "Explorer",
  strategy: "Strategy",
  planner: "Planner",
  resetter: "Resetter",
  executor: "Executor",
  evaluator: "Evaluator",
  learner: "Learner",
};

const PHASE_SUMMARIES: Record<string, string> = {
  planner: "generating program...",
  evaluator: "checking success criteria...",
  explorer: "exploring environment...",
  executor: "executing plan...",
  learner: "analyzing trace...",
  resetter: "resetting environment...",
  strategy: "analyzing goal...",
};

function ContextGauge({ used, max }: { used: number; max: number }): React.ReactElement {
  const pct = Math.min(100, Math.round((used / max) * 100));
  const color = pct >= 85 ? "red" : pct >= 60 ? "yellow" : "green";
  const filled = Math.round(pct / 10);
  const bar = "●".repeat(filled) + "○".repeat(10 - filled);
  return (
    <Text>
      <Text color={color}>{bar}</Text>
      <Text dimColor> ctx </Text>
      <Text color={color} bold>{pct}%</Text>
    </Text>
  );
}

interface StatusBarProps {
  isReplay?: boolean;
}

export function StatusBar({ isReplay }: StatusBarProps): React.ReactElement {
  const currentPhase = useAgentStore((s) => s.currentPhase);
  const phases = useAgentStore((s) => s.phases);
  const attempt = useAgentStore((s) => s.attempt);
  const error = useAgentStore((s) => s.error);
  const verdict = useAgentStore((s) => s.verdict);
  const contextUsage = useAgentStore((s) => s.contextUsage);
  const startedAt = useAgentStore((s) => s.startedAt);
  const elapsed = useElapsedTime(startedAt);

  const phase = phases[currentPhase];
  const isRunning = phase?.status === "running";
  const progressStr = phase?.progress
    ? ` ${phase.progress.current}/${phase.progress.total}`
    : "";

  const attemptStr = attempt > 0 ? `  Attempt ${attempt + 1}` : "";
  const summaryText = phase?.summary || PHASE_SUMMARIES[currentPhase] || "";

  const showVerdictInstead = verdict && error;

  return (
    <Box
      borderStyle="single"
      borderColor="gray"
      paddingX={1}
      justifyContent="space-between"
      width="100%"
      height={3}
      flexShrink={0}
    >
      <Box>
        {showVerdictInstead ? (
          <Text color={verdict.achieved ? "green" : "red"} bold>
            {verdict.achieved ? "✓" : "✗"} {verdict.summary || (verdict.achieved ? "Goal achieved" : "Goal not achieved")}
          </Text>
        ) : error ? (
          <Text color="red" bold>
            Error: {error}
          </Text>
        ) : (
          <Text>
            {isReplay && <Text color="magenta" bold>[Replay] </Text>}
            {isRunning && <><Text color="cyan"><Spinner type="dots" /></Text><Text> </Text></>}
            <Text color="yellow" bold>
              {PHASE_LABELS[currentPhase] ?? currentPhase}
            </Text>
            <Text dimColor>{progressStr}</Text>
            {summaryText && <Text dimColor>{" — "}{summaryText}</Text>}
            <Text dimColor>{attemptStr}</Text>
          </Text>
        )}
      </Box>
      <Box gap={2}>
        {contextUsage && (currentPhase === "explorer" || currentPhase === "learner") && (
          <ContextGauge used={contextUsage.used} max={contextUsage.max} />
        )}
        {currentPhase === "executor" && (
          <Text dimColor>Esc: learn</Text>
        )}
        {startedAt && <Text bold>{elapsed}</Text>}
        <Text dimColor>Ctrl+C: exit</Text>
      </Box>
    </Box>
  );
}
