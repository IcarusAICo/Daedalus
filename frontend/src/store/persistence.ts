import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const STATE_DIR = join(homedir(), ".daedalus");
const STATE_FILE = join(STATE_DIR, "state.json");
const MAX_HISTORY = 100;

export interface PersistedState {
  lastConfigPath: string | null;
  goalHistory: string[];
}

const DEFAULT_STATE: PersistedState = {
  lastConfigPath: null,
  goalHistory: [],
};

function ensureDir(): void {
  if (!existsSync(STATE_DIR)) {
    mkdirSync(STATE_DIR, { recursive: true });
  }
}

export function loadState(): PersistedState {
  try {
    const raw = readFileSync(STATE_FILE, "utf-8");
    const parsed = JSON.parse(raw);
    return {
      lastConfigPath: parsed.lastConfigPath ?? null,
      goalHistory: Array.isArray(parsed.goalHistory) ? parsed.goalHistory : [],
    };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

export function saveState(state: PersistedState): void {
  try {
    ensureDir();
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2) + "\n");
  } catch {
    // Best-effort persistence — don't crash if write fails
  }
}

export function pushGoal(goal: string): void {
  const state = loadState();
  // Avoid duplicates at the top
  if (state.goalHistory[0] === goal) return;
  state.goalHistory = [goal, ...state.goalHistory.filter((g) => g !== goal)].slice(0, MAX_HISTORY);
  saveState(state);
}

export function setLastConfig(configPath: string | null): void {
  const state = loadState();
  state.lastConfigPath = configPath;
  saveState(state);
}
