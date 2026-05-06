import React from "react";
import { Box, Text } from "ink";
import type { LearnerFeedback } from "../store/types.js";

interface FeedbackDisplayProps {
  feedback: LearnerFeedback;
}

export function FeedbackDisplay({ feedback }: FeedbackDisplayProps): React.ReactElement {
  return (
    <Box flexDirection="column">
      <Text bold>Diagnosis:</Text>
      <Text>{feedback.summary}</Text>

      {feedback.failurePoint && (
        <Box marginTop={1}>
          <Text color="red" bold>
            Failure point:{" "}
          </Text>
          <Text color="red">{feedback.failurePoint}</Text>
        </Box>
      )}

      {feedback.suggestions.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="cyan">
            Suggestions:
          </Text>
          {feedback.suggestions.map((s, i) => (
            <Box key={i} paddingLeft={1}>
              <Text dimColor>- </Text>
              <Text color="yellow">[{s.category}]</Text>
              {s.affectedStepIdx !== undefined && (
                <Text dimColor> (step {s.affectedStepIdx})</Text>
              )}
              <Text> {s.description}</Text>
            </Box>
          ))}
        </Box>
      )}

      {feedback.newSkillCandidates.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="magenta">
            New skills proposed:
          </Text>
          {feedback.newSkillCandidates.map((c, i) => (
            <Box key={i} paddingLeft={1}>
              <Text dimColor>- </Text>
              <Text bold>{c.proposedId}</Text>
              <Text dimColor>: {c.description}</Text>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}
