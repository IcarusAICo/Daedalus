import React from "react";
import { Box, Text } from "ink";

interface ThinkingBlockProps {
  text: string;
}

export function ThinkingBlock({ text }: ThinkingBlockProps): React.ReactElement | null {
  if (!text) return null;

  const lines = text.split("\n").slice(-20);

  return (
    <Box flexDirection="column" marginY={0}>
      <Text dimColor italic>
        Thinking:
      </Text>
      {lines.map((line, i) => (
        <Text key={`think-${i}`} dimColor wrap="truncate">
          {line}
        </Text>
      ))}
    </Box>
  );
}
