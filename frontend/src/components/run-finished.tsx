import React from "react";
import { Box, Text } from "ink";
import { useAgentStore } from "../store/agent-store.js";
import { useElapsedTime } from "../hooks/use-elapsed-time.js";
import { wrapLines, padEnd } from "../utils/term.js";

const CHECKMARK = "\u2713";
const CROSS = "\u2717";

export function RunFinished({ onReturn }: { onReturn: () => void }): React.ReactElement {
  const goal = useAgentStore((s) => s.goal);
  const verdict = useAgentStore((s) => s.verdict);
  const error = useAgentStore((s) => s.error);
  const startedAt = useAgentStore((s) => s.startedAt);
  const attempt = useAgentStore((s) => s.attempt);
  const runMode = useAgentStore((s) => s.runMode);
  const elapsed = useElapsedTime(startedAt);

  const achieved = verdict?.achieved ?? false;
  const hasVerdict = !!verdict;

  const borderColor = achieved ? "green" : "red";
  const statusIcon = achieved ? CHECKMARK : CROSS;
  const statusText = achieved ? "Goal Achieved" : "Goal Not Achieved";
  const statusColor = achieved ? "green" : "red";

  // Inner width of the bordered box (border takes 2 cols, paddingX=2 takes 4).
  const boxWidth = 72;
  const innerWidth = boxWidth - 6;

  const goalLines = goal ? wrapLines(goal, innerWidth - 6, 4) : [];

  const metaLine = [
    runMode ? `Mode: ${runMode}` : "",
    `Duration: ${elapsed}`,
    attempt > 0 ? `Attempts: ${attempt + 1}` : "",
  ]
    .filter(Boolean)
    .join("    ");

  const criteriaResults = hasVerdict && verdict.results ? verdict.results : [];
  const passedCount = criteriaResults.filter((r) => r.passed).length;
  const totalCount = criteriaResults.length;

  const summaryLines = hasVerdict && verdict.summary
    ? wrapLines(verdict.summary, innerWidth, 3)
    : [];

  const errorLines = !hasVerdict && error
    ? wrapLines(error, innerWidth - 2, 3)
    : [];

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} justifyContent="center" alignItems="center">
      <Box
        flexDirection="column"
        borderStyle="round"
        borderColor={borderColor}
        paddingX={2}
        paddingY={1}
        width={boxWidth}
      >
        {/* Status header */}
        <Box justifyContent="center">
          <Text color={statusColor} bold>
            {statusIcon} {statusText}
          </Text>
        </Box>
        <Text>{" "}</Text>

        {/* Goal */}
        {goalLines.length > 0 && (
          <>
            {goalLines.map((line, i) => (
              <Text key={`goal-${i}`}>
                {i === 0 ? <Text dimColor>{"Goal: "}</Text> : <Text>{"      "}</Text>}
                <Text>{line}</Text>
              </Text>
            ))}
          </>
        )}

        {/* Run info on a single line */}
        <Text>
          {metaLine}
        </Text>
        <Text>{" "}</Text>

        {/* Criteria results */}
        {criteriaResults.length > 0 && (
          <>
            <Text dimColor bold>{"Criteria:"}</Text>
            {criteriaResults.map((r, i) => (
              <Text key={`crit-${i}`} color={r.passed ? "green" : "red"}>
                {"  "}{r.passed ? CHECKMARK : CROSS}{" "}
                <Text color={r.passed ? undefined : "red"}>
                  {r.description.length > innerWidth - 4
                    ? r.description.slice(0, innerWidth - 5) + "\u2026"
                    : r.description}
                </Text>
              </Text>
            ))}
            <Text>{" "}</Text>
            <Text dimColor>
              {passedCount}/{totalCount} criteria passed
              {hasVerdict && !achieved ? " (mode-all). Goal NOT achieved." : achieved ? ". Goal achieved!" : ""}
            </Text>
          </>
        )}

        {/* Error (when no verdict) */}
        {errorLines.length > 0 && (
          <>
            <Text color="red" bold>{"Error:"}</Text>
            {errorLines.map((line, i) => (
              <Text key={`err-${i}`} color="red">{"  "}{line}</Text>
            ))}
          </>
        )}

        {/* Verdict summary */}
        {summaryLines.length > 0 && (
          <>
            {summaryLines.map((line, i) => (
              <Text key={`sum-${i}`} dimColor>{line}</Text>
            ))}
          </>
        )}
      </Box>

      {/* Return prompt */}
      <Text>{" "}</Text>
      <Text dimColor>
        {"Press "}<Text color="cyan" bold>Enter</Text>{" to return to start  \u2022  "}<Text color="cyan" bold>Esc</Text>{" to exit"}
      </Text>
    </Box>
  );
}
