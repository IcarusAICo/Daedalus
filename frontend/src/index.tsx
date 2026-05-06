import React from "react";
import { render } from "ink";
import { App } from "./app.js";

const args = process.argv.slice(2);
let goal: string | undefined;
let configPath: string | undefined;
let projectRoot = process.cwd();
let tracePath: string | undefined;

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if ((arg === "--goal" || arg === "-g") && args[i + 1]) {
    goal = args[++i];
  } else if ((arg === "--config" || arg === "-c") && args[i + 1]) {
    configPath = args[++i];
  } else if ((arg === "--cwd" || arg === "-d") && args[i + 1]) {
    projectRoot = args[++i];
  } else if ((arg === "--trace" || arg === "-t") && args[i + 1]) {
    tracePath = args[++i];
  } else if (arg === "--help" || arg === "-h") {
    console.log(`
daedalus - interactive terminal UI for the Daedalus computer control agent

Usage:
  daedalus [options]            Launch interactive UI
  daedalus run [options]        Run directly (headless, Python CLI)
  daedalus skills list          List skills
  daedalus traces list          List traces

Interactive UI Options:
  -g, --goal <text>     Start with a goal (skips goal prompt)
  -c, --config <path>   Path to daedalus YAML config
  -d, --cwd <path>      Working directory for daedalus backend
  -t, --trace <path>    View a previous run (trace dir path or task ID)
  -h, --help            Show this help message

Controls:
  Ctrl+C    Abort running task
  c         Toggle configuration screen
  q         Quit (when idle)
`);
    process.exit(0);
  }
}

if (!process.stdin.isTTY) {
  console.error(
    "error: daedalus interactive UI requires a TTY.\n" +
    "       Use `daedalus run --goal '...'` for non-interactive usage."
  );
  process.exit(1);
}

render(<App goal={goal} configPath={configPath} projectRoot={projectRoot} tracePath={tracePath} />, {
  exitOnCtrlC: false,
  alternateScreen: true,
});
