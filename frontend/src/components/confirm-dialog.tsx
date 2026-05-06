import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { ConfirmRequest } from "../store/types.js";

interface ConfirmDialogProps {
  request: ConfirmRequest;
  onApprove: (id: number) => void;
  onDeny: (id: number, comments: string) => void;
}

type Mode = "choice" | "commenting";

export function ConfirmDialog({
  request,
  onApprove,
  onDeny,
}: ConfirmDialogProps): React.ReactElement {
  const [mode, setMode] = useState<Mode>("choice");
  const [selected, setSelected] = useState(0);
  const [comment, setComment] = useState("");

  const options = ["Approve", "Deny with comments", "Cancel"];

  useInput((input, key) => {
    if (mode === "choice") {
      if (key.upArrow) {
        setSelected((s) => Math.max(0, s - 1));
      } else if (key.downArrow) {
        setSelected((s) => Math.min(options.length - 1, s + 1));
      } else if (key.return) {
        if (selected === 0) {
          onApprove(request.id);
        } else if (selected === 1) {
          setMode("commenting");
        } else {
          onDeny(request.id, "");
        }
      }
    } else if (mode === "commenting") {
      if (key.return) {
        onDeny(request.id, comment);
      } else if (key.escape) {
        setMode("choice");
        setComment("");
      } else if (key.backspace || key.delete) {
        setComment((c) => c.slice(0, -1));
      } else if (input && !key.ctrl && !key.meta) {
        setComment((c) => c + input);
      }
    }
  });

  const title =
    request.type === "program"
      ? "Confirm Program"
      : request.type === "criteria"
        ? "Confirm Success Criteria"
        : "Confirm Skills";

  return (
    <Box
      flexDirection="column"
      borderStyle="double"
      borderColor="yellow"
      paddingX={1}
      paddingY={0}
      marginY={1}
    >
      <Text bold color="yellow">
        {title}
      </Text>

      {request.type === "program" && (
        <ProgramPreview payload={request.payload} />
      )}
      {request.type === "criteria" && (
        <CriteriaPreview payload={request.payload} />
      )}
      {request.type === "skills" && (
        <SkillsPreview payload={request.payload} />
      )}

      {mode === "choice" && (
        <Box flexDirection="column" marginTop={1}>
          {options.map((opt, i) => (
            <Box key={i}>
              <Text color={i === selected ? "cyan" : undefined}>
                {i === selected ? "❯ " : "  "}
                {opt}
              </Text>
            </Box>
          ))}
          <Box marginTop={1}>
            <Text dimColor>↑↓ navigate, Enter select</Text>
          </Box>
        </Box>
      )}

      {mode === "commenting" && (
        <Box flexDirection="column" marginTop={1}>
          <Text>Comments (Enter to submit, Esc to go back):</Text>
          <Box>
            <Text color="cyan">{"> "}</Text>
            <Text>{comment}</Text>
            <Text color="cyan">█</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

function ProgramPreview({ payload }: { payload: unknown }): React.ReactElement {
  const data = payload as Record<string, unknown>;
  const program = data?.program as Record<string, unknown> | undefined;

  if (!program) {
    return (
      <Box flexDirection="column" marginTop={1}>
        <Text dimColor wrap="wrap">{JSON.stringify(data, null, 2).slice(0, 500)}</Text>
      </Box>
    );
  }

  const steps = (program.steps as Array<Record<string, unknown>>) ?? [];
  const code = program.code as string | undefined;
  const description = program.description as string | undefined;

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text bold>{program.name as string}</Text>
      {description && (
        <Text dimColor wrap="wrap">{description}</Text>
      )}
      {steps.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor bold>Steps:</Text>
          {steps.slice(0, 12).map((step, i) => (
            <Box key={i} flexDirection="column" paddingLeft={1}>
              <Box>
                <Text dimColor>{(i + 1).toString().padStart(2)}. </Text>
                <Text color="cyan">{String(step.skill ?? "")}</Text>
                {typeof step.description === "string" && (
                  <Text dimColor>{" — "}{truncateStr(step.description, 50)}</Text>
                )}
              </Box>
              {typeof step.inputs === "object" && step.inputs !== null && Object.keys(step.inputs).length > 0 ? (
                <Box paddingLeft={5}>
                  <Text dimColor>{formatInputs(step.inputs as Record<string, unknown>)}</Text>
                </Box>
              ) : null}
            </Box>
          ))}
          {steps.length > 12 && (
            <Box paddingLeft={1}>
              <Text dimColor>... {steps.length - 12} more steps</Text>
            </Box>
          )}
        </Box>
      )}
      {code && !steps.length && (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor bold>Code:</Text>
          <Box paddingLeft={1}>
            <Text wrap="wrap">{code.slice(0, 500)}</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

function CriteriaPreview({ payload }: { payload: unknown }): React.ReactElement {
  const data = payload as Record<string, unknown>;
  const criteria = (data?.criteria as Array<Record<string, unknown>>) ?? [];
  const goalSummary = data?.goal_summary as string | undefined;
  const mustPassAll = data?.must_pass_all as boolean | undefined;

  if (!criteria.length && !goalSummary) {
    return (
      <Box flexDirection="column" marginTop={1}>
        <Text dimColor wrap="wrap">{JSON.stringify(data, null, 2).slice(0, 500)}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      {goalSummary && (
        <Box marginBottom={1}>
          <Text dimColor>Goal: </Text>
          <Text wrap="wrap">{goalSummary}</Text>
        </Box>
      )}
      <Text bold>Success Criteria{mustPassAll === false ? " (any one must pass)" : " (all must pass)"}:</Text>
      {criteria.map((c, i) => (
        <Box key={i} flexDirection="column" paddingLeft={1}>
          <Box>
            <Text dimColor>• </Text>
            <Text color="yellow">[{c.kind as string}]</Text>
            <Text wrap="wrap"> {c.description as string}</Text>
          </Box>
          {typeof c.visual_claim === "string" && (
            <Box paddingLeft={3}>
              <Text dimColor>claim: "{c.visual_claim}"</Text>
            </Box>
          )}
        </Box>
      ))}
    </Box>
  );
}

function SkillsPreview({ payload }: { payload: unknown }): React.ReactElement {
  const data = payload as Record<string, unknown>;
  const skills = (data?.skills as Array<Record<string, unknown>>) ?? [];

  if (skills.length === 0) {
    if (data && Object.keys(data).length > 0) {
      return (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor wrap="wrap">{JSON.stringify(data, null, 2).slice(0, 500)}</Text>
        </Box>
      );
    }
    return <Text dimColor>No skills proposed</Text>;
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text bold>Proposed New Skills:</Text>
      {skills.map((s, i) => (
        <Box key={i} flexDirection="column" paddingLeft={1}>
          <Box>
            <Text dimColor>• </Text>
            <Text color="cyan">{(s.proposed_id ?? s.name ?? "unknown") as string}</Text>
          </Box>
          {typeof s.description === "string" && (
            <Box paddingLeft={3}>
              <Text dimColor wrap="wrap">{s.description}</Text>
            </Box>
          )}
        </Box>
      ))}
    </Box>
  );
}

function truncateStr(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}

function formatInputs(inputs: Record<string, unknown>): string {
  const entries = Object.entries(inputs).slice(0, 4);
  const parts = entries.map(([k, v]) => {
    if (typeof v === "string") return `${k}="${truncateStr(v, 25)}"`;
    if (typeof v === "number" || typeof v === "boolean") return `${k}=${v}`;
    return `${k}=…`;
  });
  if (Object.keys(inputs).length > 4) parts.push("…");
  return parts.join(", ");
}
