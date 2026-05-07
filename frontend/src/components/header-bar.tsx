import React from "react";
import { Box, Text } from "ink";
import { useAgentStore } from "../store/agent-store.js";
import { useElapsedTime } from "../hooks/use-elapsed-time.js";

export function HeaderBar(): React.ReactElement {
  const config = useAgentStore((s) => s.config);
  const configPath = useAgentStore((s) => s.configPath);
  const connected = useAgentStore((s) => s.connected);
  const startedAt = useAgentStore((s) => s.startedAt);
  const runMode = useAgentStore((s) => s.runMode);
  const elapsed = useElapsedTime(startedAt);

  const connectionStr =
    config.backend.kind === "vnc"
      ? `vnc://${config.backend.host}:${config.backend.port}`
      : "mock://local";

  const statusColor = connected ? "green" : "gray";
  const modeColor = runMode === "explore" ? "magenta" : runMode === "plan" ? "blue" : "green";

  return (
    <Box
      borderStyle="single"
      borderColor="cyan"
      paddingX={1}
      justifyContent="space-between"
      width="100%"
      height={3}
      flexShrink={0}
    >
      <Box gap={2}>
        <Text bold color="cyan">
          DAEDALUS{runMode && <Text color={modeColor}>{` [${runMode}]`}</Text>} <Text dimColor>v0.0.1</Text>
        </Text>
        {configPath && (
          <Text dimColor>[{configPath}]</Text>
        )}
      </Box>
      <Text>
        <Text color={statusColor}>[{connectionStr}]</Text>
        {"  "}
        <Text bold>{elapsed}</Text>
      </Text>
    </Box>
  );
}
