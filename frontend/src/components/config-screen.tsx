import React, { useState, useEffect, useMemo } from "react";
import { Box, Text, useInput } from "ink";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { parse as parseYaml } from "yaml";
import { useAgentStore } from "../store/agent-store.js";
import { setLastConfig } from "../store/persistence.js";
import type { DaedalusConfig } from "../store/types.js";
import { ScrollList, type ScrollListItem } from "./scroll-list.js";
import { padEnd, visualWidth } from "../utils/term.js";

interface ConfigScreenProps {
  projectRoot: string;
  viewportRows: number;
  viewportCols: number;
}

const LABEL_WIDTH = 24;
const CURSOR_WIDTH = 2;
const DIVIDER_CHAR = "\u2500"; // ─

export function ConfigScreen({
  projectRoot,
  viewportRows,
  viewportCols,
}: ConfigScreenProps): React.ReactElement {
  const config = useAgentStore((s) => s.config);
  const configPath = useAgentStore((s) => s.configPath);
  const setConfig = useAgentStore((s) => s.setConfig);
  const setConfigPath = useAgentStore((s) => s.setConfigPath);
  const toggleConfig = useAgentStore((s) => s.toggleConfig);

  const [selected, setSelected] = useState(0);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");

  const locked = configPath !== null;

  const [configFiles, setConfigFiles] = useState<string[]>([]);
  const [traceDirs, setTraceDirs] = useState<string[]>([]);
  useEffect(() => {
    try {
      const files = readdirSync(projectRoot)
        .filter((f) => f.endsWith(".yaml") || f.endsWith(".yml"))
        .sort();
      setConfigFiles(files);
    } catch {
      setConfigFiles([]);
    }
    try {
      const tracesPath = resolve(projectRoot, config.tracesDir || "./traces");
      const dirs = readdirSync(tracesPath)
        .filter((d) => d.startsWith("t_"))
        .filter((d) => existsSync(join(tracesPath, d, "explorer", "observations.md")))
        .sort()
        .reverse();
      setTraceDirs(dirs);
    } catch {
      setTraceDirs([]);
    }
  }, [projectRoot, config.tracesDir]);

  const fields = useMemo(
    () => getFields(config, configPath, configFiles, traceDirs),
    [config, configPath, configFiles, traceDirs]
  );

  const rows = useMemo(() => buildRows(fields), [fields]);
  const selectedRowIdx = useMemo(
    () => rows.findIndex((r) => r.kind === "field" && r.fieldIdx === selected),
    [rows, selected]
  );

  // Clamp selected if fields change underneath us.
  useEffect(() => {
    if (selected >= fields.length) {
      setSelected(Math.max(0, fields.length - 1));
    }
  }, [fields.length, selected]);

  useInput((input, key) => {
    if (key.escape) {
      if (editing) {
        setEditing(false);
        setEditValue("");
      } else {
        toggleConfig();
      }
      return;
    }

    if (!editing) {
      if (input === "q") {
        toggleConfig();
        return;
      }

      if (key.upArrow) {
        setSelected((s) => Math.max(0, s - 1));
      } else if (key.downArrow) {
        setSelected((s) => Math.min(fields.length - 1, s + 1));
      } else if (key.return) {
        const field = fields[selected];
        if (!field) return;

        if (field.path === "_configPath") {
          if (locked) {
            setConfigPath(null);
            setLastConfig(null);
          } else {
            const firstFile = configFiles[0];
            if (firstFile) {
              setConfigPath(firstFile);
              setLastConfig(firstFile);
              loadConfigFromFile(resolve(projectRoot, firstFile), setConfig);
            }
          }
          return;
        }

        if (field.path === "_reuseExploration") {
          if (config.reuseExploration) {
            setConfig({ reuseExploration: null });
          } else {
            const first = traceDirs[0];
            if (first) {
              setConfig({ reuseExploration: first });
            }
          }
          return;
        }

        if (locked) return;

        if (field.type === "boolean") {
          applyField(field.path, !field.value, config, setConfig);
        } else if (field.type === "select") {
          const options = field.options!;
          const currentIdx = options.indexOf(String(field.value));
          const nextIdx = (currentIdx + 1) % options.length;
          applyField(field.path, options[nextIdx], config, setConfig);
        } else {
          setEditing(true);
          setEditValue(String(field.value));
        }
      } else if (key.tab || key.leftArrow || key.rightArrow) {
        const field = fields[selected];
        if (!field) return;

        if (field.path === "_configPath") {
          if (configFiles.length === 0) return;
          const currentIdx = configPath ? configFiles.indexOf(configPath) : -1;
          let nextIdx: number;
          if (key.leftArrow) {
            nextIdx = currentIdx <= 0 ? configFiles.length - 1 : currentIdx - 1;
          } else {
            nextIdx = currentIdx >= configFiles.length - 1 ? 0 : currentIdx + 1;
          }
          const nextFile = configFiles[nextIdx];
          setConfigPath(nextFile);
          setLastConfig(nextFile);
          loadConfigFromFile(resolve(projectRoot, nextFile), setConfig);
          return;
        }

        if (field.path === "_reuseExploration") {
          if (traceDirs.length === 0) return;
          const currentIdx = config.reuseExploration ? traceDirs.indexOf(config.reuseExploration) : -1;
          let nextIdx: number;
          if (key.leftArrow) {
            nextIdx = currentIdx <= 0 ? traceDirs.length - 1 : currentIdx - 1;
          } else {
            nextIdx = currentIdx >= traceDirs.length - 1 ? 0 : currentIdx + 1;
          }
          setConfig({ reuseExploration: traceDirs[nextIdx] });
          return;
        }

        if (locked) return;

        if (field.type === "boolean") {
          applyField(field.path, !field.value, config, setConfig);
        } else if (field.type === "select") {
          const options = field.options!;
          const currentIdx = options.indexOf(String(field.value));
          const dir = key.leftArrow ? -1 : 1;
          const nextIdx = (currentIdx + dir + options.length) % options.length;
          applyField(field.path, options[nextIdx], config, setConfig);
        }
      }
    } else {
      if (key.return) {
        const field = fields[selected];
        if (field) applyField(field.path, coerce(editValue, field.type), config, setConfig);
        setEditing(false);
        setEditValue("");
      } else if (key.backspace || key.delete) {
        setEditValue((v) => v.slice(0, -1));
      } else if (input && !key.ctrl && !key.meta) {
        setEditValue((v) => v + input);
      }
    }
  });

  const items: ScrollListItem[] = rows.map((row, rowIdx) => {
    if (row.kind === "divider") {
      return {
        key: `div-${rowIdx}`,
        height: 1,
        render: ({ cols }) => <DividerRow cols={cols} />,
      };
    }
    const field = row.field;
    const isSelected = field === fields[selected];
    return {
      key: `f-${row.fieldIdx}`,
      height: 1,
      render: ({ cols }) => (
        <FieldRow
          field={field}
          selected={isSelected}
          cols={cols}
          locked={locked}
          editing={editing && isSelected}
          editValue={editValue}
        />
      ),
    };
  });

  const helpText = `Esc/q: back  \u2191\u2193: navigate  Enter: toggle/edit  Tab/\u2190\u2192: cycle${locked ? "  (locked)" : ""}`;

  return (
    <Box flexDirection="column" width={viewportCols} height={viewportRows} flexShrink={0}>
      <Text bold color="cyan">{padEnd("Configuration", viewportCols)}</Text>
      <Text dimColor>{padEnd(helpText, viewportCols)}</Text>
      <ScrollList
        items={items}
        selectedIndex={selectedRowIdx}
        viewportRows={Math.max(0, viewportRows - 2)}
        viewportCols={viewportCols}
      />
    </Box>
  );
}

type ConfigRow =
  | { kind: "field"; field: FieldDef; fieldIdx: number }
  | { kind: "divider"; afterSection: number };

function buildRows(fields: FieldDef[]): ConfigRow[] {
  const out: ConfigRow[] = [];
  for (let i = 0; i < fields.length; i++) {
    if (i > 0 && fields[i].section !== fields[i - 1].section) {
      out.push({ kind: "divider", afterSection: fields[i - 1].section });
    }
    out.push({ kind: "field", field: fields[i], fieldIdx: i });
  }
  return out;
}

function DividerRow({ cols }: { cols: number }): React.ReactElement {
  const dashes = Math.max(0, Math.min(cols - 4, 60));
  const text = "  " + DIVIDER_CHAR.repeat(dashes);
  return (
    <Box width={cols} height={1} flexShrink={0}>
      <Text dimColor>{padEnd(text, cols)}</Text>
    </Box>
  );
}

interface FieldRowProps {
  field: FieldDef;
  selected: boolean;
  cols: number;
  locked: boolean;
  editing: boolean;
  editValue: string;
}

function FieldRow({ field, selected, cols, locked, editing, editValue }: FieldRowProps): React.ReactElement {
  const isConfigField = field.path === "_configPath";
  const isReuseField = field.path === "_reuseExploration";
  const isLocked = locked && !isConfigField && !isReuseField;

  let valueColor: string | undefined = "yellow";
  if (isLocked) valueColor = "gray";
  else if (field.type === "boolean") valueColor = field.value ? "green" : "red";
  else if (isConfigField) valueColor = String(field.value) !== "(none)" ? "green" : "yellow";
  else if (isReuseField) valueColor = String(field.value) !== "(none)" ? "magenta" : "yellow";

  const cursor = selected ? "\u276F " : "  ";
  const labelText = field.label.length > LABEL_WIDTH ? field.label.slice(0, LABEL_WIDTH) : field.label;
  const labelPadded = labelText + " ".repeat(Math.max(0, LABEL_WIDTH - labelText.length));

  const valueText = editing
    ? editValue
    : renderValue(field, isLocked, isConfigField);

  // Compute exact trailing pad so the row width = cols.
  const consumed = CURSOR_WIDTH + LABEL_WIDTH + visualWidth(valueText) + (editing ? 1 : 0);
  const trailing = " ".repeat(Math.max(0, cols - consumed));

  // If the row is wider than cols, truncate the value text instead.
  const effectiveValue = consumed > cols
    ? truncatePlain(valueText, Math.max(0, cols - CURSOR_WIDTH - LABEL_WIDTH - (editing ? 1 : 0)))
    : valueText;

  return (
    <Box width={cols} height={1} flexShrink={0}>
      <Text color={selected ? "cyan" : undefined}>{cursor}</Text>
      <Text bold={selected} color={isLocked ? "gray" : selected ? "white" : "gray"}>
        {labelPadded}
      </Text>
      {editing ? (
        <>
          <Text color="cyan">{effectiveValue}</Text>
          <Text color="cyan">{"\u2588"}</Text>
        </>
      ) : (
        <Text color={valueColor} dimColor={isLocked}>
          {effectiveValue}
        </Text>
      )}
      <Text>{trailing}</Text>
    </Box>
  );
}

function truncatePlain(s: string, cols: number): string {
  if (cols <= 0) return "";
  const chars = [...s];
  if (chars.length <= cols) return s;
  if (cols === 1) return chars[0];
  return chars.slice(0, cols - 1).join("") + "\u2026";
}

function renderValue(field: FieldDef, isLocked: boolean, isConfigField: boolean): string {
  if (field.type === "boolean") {
    return field.value ? "yes" : "no";
  }
  if (isConfigField) {
    const active = String(field.value) !== "(none)";
    return (active ? `\u2713 ${field.value}` : "(disabled)") + "  \u2190 Enter: on/off  Tab/\u2190\u2192: switch file";
  }
  if (field.path === "_reuseExploration") {
    const active = String(field.value) !== "(none)";
    return (active ? `\u2713 ${field.value}` : "(disabled)") + "  \u2190 Enter: on/off  Tab/\u2190\u2192: switch trace";
  }
  if (field.type === "select") {
    const v = String(field.value) || "(none)";
    return isLocked ? v : v + "  \u2190 Tab/\u2190\u2192";
  }
  return String(field.value) || "(empty)";
}

interface FieldDef {
  label: string;
  path: string;
  value: unknown;
  type: "string" | "number" | "boolean" | "select";
  options?: string[];
  section: number;
}

function getFields(
  config: DaedalusConfig,
  configPath: string | null,
  configFiles: string[],
  traceDirs: string[]
): FieldDef[] {
  return [
    { label: "Reuse Exploration", path: "_reuseExploration", value: config.reuseExploration ?? "(none)", type: "select", options: traceDirs, section: 0 },
    { label: "Config File", path: "_configPath", value: configPath ?? "(none)", type: "select", options: configFiles, section: 0 },
    { label: "Backend", path: "backend.kind", value: config.backend.kind, type: "select", options: ["vnc", "mock"], section: 1 },
    { label: "Host", path: "backend.host", value: config.backend.host, type: "string", section: 1 },
    { label: "Port", path: "backend.port", value: config.backend.port, type: "number", section: 1 },
    { label: "Host OS", path: "backend.hostOs", value: config.backend.hostOs, type: "string", section: 1 },
    { label: "Password Env", path: "backend.passwordEnv", value: config.backend.passwordEnv ?? "", type: "string", section: 1 },
    { label: "Username Env", path: "backend.usernameEnv", value: config.backend.usernameEnv ?? "", type: "string", section: 1 },
    { label: "Max Width", path: "backend.maxWidth", value: config.backend.maxWidth ?? "", type: "number", section: 1 },
    { label: "Max Height", path: "backend.maxHeight", value: config.backend.maxHeight ?? "", type: "number", section: 1 },
    { label: "Planner LLM", path: "llmRoles.planner", value: config.llmRoles.planner ?? "", type: "string", section: 2 },
    { label: "Explorer LLM", path: "llmRoles.explorer", value: config.llmRoles.explorer ?? "", type: "string", section: 2 },
    { label: "Implementor LLM", path: "llmRoles.implementor", value: config.llmRoles.implementor ?? "", type: "string", section: 2 },
    { label: "Learner LLM", path: "llmRoles.learner", value: config.llmRoles.learner ?? "", type: "string", section: 2 },
    { label: "Vision LLM", path: "llmRoles.vision", value: config.llmRoles.vision ?? "", type: "string", section: 2 },
    { label: "Cheap LLM", path: "llmRoles.cheap", value: config.llmRoles.cheap ?? "", type: "string", section: 2 },
    { label: "AWS Region", path: "llmAwsRegion", value: config.llmAwsRegion, type: "string", section: 2 },
    { label: "Request Timeout (s)", path: "llmRequestTimeoutS", value: config.llmRequestTimeoutS, type: "number", section: 2 },
    { label: "Creative Temp", path: "llmCreativeTemp", value: config.llmCreativeTemp, type: "number", section: 2 },
    { label: "Analytical Temp", path: "llmAnalyticalTemp", value: config.llmAnalyticalTemp, type: "number", section: 2 },
    { label: "Screen Width", path: "executor.defaultScreenWidth", value: config.executor.defaultScreenWidth, type: "number", section: 3 },
    { label: "Screen Height", path: "executor.defaultScreenHeight", value: config.executor.defaultScreenHeight, type: "number", section: 3 },
    { label: "Step Timeout (s)", path: "executor.stepTimeoutS", value: config.executor.stepTimeoutS, type: "number", section: 3 },
    { label: "Max Retries", path: "maxRetries", value: config.maxRetries, type: "number", section: 3 },
    { label: "Explore Steps", path: "exploreSteps", value: config.exploreSteps, type: "number", section: 3 },
    { label: "Record", path: "record", value: config.record, type: "boolean", section: 4 },
    { label: "Record FPS", path: "recordFps", value: config.recordFps, type: "number", section: 4 },
    { label: "No Strategy", path: "noStrategy", value: config.noStrategy, type: "boolean", section: 4 },
    { label: "Yolo (auto-approve)", path: "yolo", value: config.yolo, type: "boolean", section: 4 },
    { label: "Verbose", path: "verbose", value: config.verbose, type: "boolean", section: 4 },
    { label: "Skills Dir", path: "skillsDir", value: config.skillsDir, type: "string", section: 5 },
    { label: "Traces Dir", path: "tracesDir", value: config.tracesDir, type: "string", section: 5 },
    { label: "Tasks DB", path: "tasksDb", value: config.tasksDb, type: "string", section: 5 },
  ];
}

function loadConfigFromFile(
  filePath: string,
  setConfig: (partial: Partial<DaedalusConfig>) => void
): void {
  try {
    const raw = parseYaml(readFileSync(filePath, "utf-8")) || {};
    const patch: Partial<DaedalusConfig> = {};

    const be = raw.backend || {};
    const vnc = be.vnc || {};
    patch.backend = {
      kind: be.kind || "vnc",
      host: vnc.host || "127.0.0.1",
      port: vnc.port || 5900,
      hostOs: be.host_os || "unknown",
      passwordEnv: vnc.password_env,
      usernameEnv: vnc.username_env,
      maxWidth: vnc.max_width,
      maxHeight: vnc.max_height,
    };

    const llm = raw.llm || {};
    const roles = llm.roles || {};
    patch.llmRoles = {
      planner: roles.planner,
      explorer: roles.explorer,
      implementor: roles.implementor,
      learner: roles.learner,
      vision: roles.vision,
      cheap: roles.cheap,
    };
    if (llm.aws_region) patch.llmAwsRegion = llm.aws_region;
    if (llm.request_timeout_s) patch.llmRequestTimeoutS = llm.request_timeout_s;
    if (llm.creative_temperature !== undefined) patch.llmCreativeTemp = llm.creative_temperature;
    if (llm.analytical_temperature !== undefined) patch.llmAnalyticalTemp = llm.analytical_temperature;

    const exec = raw.executor || {};
    patch.executor = {
      defaultScreenWidth: exec.default_screen_width || 1920,
      defaultScreenHeight: exec.default_screen_height || 1080,
      stepTimeoutS: exec.step_timeout_s || 60,
    };

    const agent = raw.agent || {};
    if (agent.max_retries !== undefined) patch.maxRetries = agent.max_retries;
    if (agent.explore_steps !== undefined) patch.exploreSteps = agent.explore_steps;
    if (agent.no_strategy !== undefined) patch.noStrategy = agent.no_strategy;
    if (agent.verbose !== undefined) patch.verbose = agent.verbose;
    if (agent.record !== undefined) patch.record = agent.record;
    if (agent.record_fps !== undefined) patch.recordFps = agent.record_fps;

    const paths = raw.paths || {};
    if (paths.skills_dir) patch.skillsDir = paths.skills_dir;
    if (paths.traces_dir) patch.tracesDir = paths.traces_dir;
    if (paths.tasks_db) patch.tasksDb = paths.tasks_db;

    const ui = raw.ui || {};
    if (ui.overlay !== undefined) patch.noOverlay = !ui.overlay;

    setConfig(patch);
  } catch {
    // If file can't be read/parsed, leave config as-is.
  }
}

function coerce(value: string, type: "string" | "number" | "boolean" | "select"): unknown {
  if (type === "number") return Number(value) || 0;
  if (type === "boolean") return value === "true" || value === "yes";
  return value;
}

function applyField(
  path: string,
  value: unknown,
  config: DaedalusConfig,
  setConfig: (partial: Partial<DaedalusConfig>) => void
): void {
  const parts = path.split(".");
  if (parts.length === 1) {
    setConfig({ [parts[0]]: value } as Partial<DaedalusConfig>);
  } else if (parts[0] === "backend") {
    setConfig({
      backend: { ...config.backend, [parts[1]]: value },
    });
  } else if (parts[0] === "executor") {
    setConfig({
      executor: { ...config.executor, [parts[1]]: value },
    });
  } else if (parts[0] === "llmRoles") {
    setConfig({
      llmRoles: { ...config.llmRoles, [parts[1]]: value || undefined },
    });
  }
}
