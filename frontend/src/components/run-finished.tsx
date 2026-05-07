import React from "react";
import { Box, Text } from "ink";
import { useAgentStore } from "../store/agent-store.js";
import { useElapsedTime } from "../hooks/use-elapsed-time.js";

const CHECKMARK = "✓";
const CROSS = "✗";

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

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} justifyContent="center">
      <Box
        flexDirection="column"
        borderStyle="round"
        borderColor={borderColor}
        paddingX={2}
        paddingY={1}
        alignSelf="center"
        width={72}
      >
        {/* Status header */}
        <Box justifyContent="center" marginBottom={1}>
          <Text color={statusColor} bold>
            {statusIcon} {statusText}
          </Text>
        </Box>

        {/* Goal */}
        {goal && (
          <Box marginBottom={1}>
            <Text dimColor>Goal: </Text>
            <Text wrap="wrap">{goal}</Text>
          </Box>
        )}

        {/* Run info */}
        <Box gap={3} marginBottom={1}>
          {runMode && (
            <Box>
              <Text dimColor>Mode: </Text>
              <Text color="cyan">{runMode}</Text>
            </Box>
          )}
          <Box>
            <Text dimColor>Duration: </Text>
            <Text bold>{elapsed}</Text>
          </Box>
          {attempt > 0 && (
            <Box>
              <Text dimColor>Attempts: </Text>
              <Text>{attempt + 1}</Text>
            </Box>
          )}
        </Box>

        {/* Criteria results */}
        {hasVerdict && verdict.results && verdict.results.length > 0 && (
          <Box flexDirection="column" marginTop={1}>
            <Text dimColor bold>Criteria:</Text>
            {verdict.results.map((r, i) => (
              <Box key={i} paddingLeft={1}>
                <Text color={r.passed ? "green" : "red"}>
                  {r.passed ? CHECKMARK : CROSS}
                </Text>
                <Text> </Text>
                <Text wrap="wrap" color={r.passed ? undefined : "red"}>
                  {r.description}
                </Text>
              </Box>
            ))}
            <Box marginTop={1}>
              <Text dimColor>
                {verdict.results.filter((r) => r.passed).length}/{verdict.results.length} criteria passed
              </Text>
            </Box>
          </Box>
        )}

        {/* Error info (if no verdict) */}
        {!hasVerdict && error && (
          <Box flexDirection="column" marginTop={1}>
            <Text color="red" bold>Error:</Text>
            <Box paddingLeft={1}>
              <Text color="red" wrap="wrap">{error}</Text>
            </Box>
          </Box>
        )}

        {/* Verdict summary */}
        {hasVerdict && verdict.summary && (
          <Box marginTop={1}>
            <Text dimColor wrap="wrap">{verdict.summary}</Text>
          </Box>
        )}
      </Box>

      {/* Return prompt */}
      <Box justifyContent="center" marginTop={2}>
        <Text dimColor>
          Press <Text color="cyan" bold>Enter</Text> to return to start  •  <Text color="cyan" bold>Esc</Text> to exit
        </Text>
      </Box>
    </Box>
  );
}
