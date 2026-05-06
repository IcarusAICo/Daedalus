import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import type { StepResult, Program } from "../store/types.js";

interface StepProgressProps {
  steps: StepResult[];
  program: Program | null;
}

export function StepProgress({
  steps,
  program,
}: StepProgressProps): React.ReactElement | null {
  if (!program || steps.length === 0) return null;

  return (
    <Box flexDirection="column">
      {steps.map((step) => (
        <Box key={step.idx}>
          {step.status === "running" ? (
            <Text color="cyan">
              <Spinner type="dots" />{" "}
            </Text>
          ) : step.status === "success" ? (
            <Text color="green">✓ </Text>
          ) : step.status === "error" ? (
            <Text color="red">✗ </Text>
          ) : (
            <Text dimColor>○ </Text>
          )}
          <Text
            color={step.status === "running" ? "white" : undefined}
            bold={step.status === "running"}
            dimColor={step.status === "success"}
          >
            Step {step.idx + 1}: {step.skillId}
          </Text>
          {step.duration !== undefined && (
            <Text dimColor> ({(step.duration / 1000).toFixed(1)}s)</Text>
          )}
          {step.error && (
            <Text color="red"> — {step.error}</Text>
          )}
        </Box>
      ))}
      {program && steps.length < program.steps.length && (
        <Text dimColor>
          ... {program.steps.length - steps.length} step
          {program.steps.length - steps.length !== 1 ? "s" : ""} remaining
        </Text>
      )}
    </Box>
  );
}
