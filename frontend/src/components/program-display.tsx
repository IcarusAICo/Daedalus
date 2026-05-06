import React from "react";
import { Box, Text } from "ink";
import type { Program } from "../store/types.js";

interface ProgramDisplayProps {
  program: Program;
}

export function ProgramDisplay({ program }: ProgramDisplayProps): React.ReactElement {
  return (
    <Box flexDirection="column">
      <Text dimColor>
        Generated program: <Text bold color="white">{program.name}</Text> ({program.steps.length} steps)
      </Text>
      {program.steps.map((step, i) => (
        <Box key={i} paddingLeft={1}>
          <Text dimColor>{(i + 1).toString().padStart(2)}. </Text>
          <Text color="cyan">{step.skillId}</Text>
          {Object.keys(step.inputs).length > 0 && (
            <Text dimColor>
              ({Object.entries(step.inputs)
                .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                .join(", ")
                .slice(0, 60)})
            </Text>
          )}
        </Box>
      ))}
    </Box>
  );
}
