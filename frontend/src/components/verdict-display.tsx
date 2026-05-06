import React from "react";
import { Box, Text } from "ink";
import type { GoalVerdict } from "../store/types.js";

interface VerdictDisplayProps {
  verdict: GoalVerdict;
}

export function VerdictDisplay({ verdict }: VerdictDisplayProps): React.ReactElement {
  const statusColor = verdict.achieved ? "green" : "red";
  const statusText = verdict.achieved ? "GOAL ACHIEVED" : "GOAL NOT ACHIEVED";

  return (
    <Box flexDirection="column">
      <Text bold color={statusColor}>
        {statusText}
      </Text>
      <Text dimColor>{verdict.summary}</Text>
      {verdict.results.map((r, i) => (
        <Box key={i} paddingLeft={1}>
          <Text color={r.passed ? "green" : "red"}>
            {r.passed ? "PASS" : "FAIL"}
          </Text>
          <Text>
            {" "}[{r.kind}] {r.description}
          </Text>
          {r.explanation && (
            <Text dimColor> — {r.explanation}</Text>
          )}
        </Box>
      ))}
    </Box>
  );
}
