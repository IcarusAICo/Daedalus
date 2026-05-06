import React from "react";
import { Box, Text } from "ink";

interface GoalDisplayProps {
  goal: string | null;
  isReplay?: boolean;
}

export function GoalDisplay({ goal, isReplay }: GoalDisplayProps): React.ReactElement | null {
  if (!goal) return null;

  return (
    <Box paddingX={1} marginBottom={1}>
      <Text wrap="wrap">
        {isReplay && <Text color="magenta" bold>{"[Replay] "}</Text>}
        <Text color="green" bold>{"❯ "}</Text>
        <Text bold>{goal}</Text>
      </Text>
    </Box>
  );
}
