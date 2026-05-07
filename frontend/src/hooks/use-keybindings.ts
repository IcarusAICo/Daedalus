import { useInput } from "ink";
import { useAgentStore } from "../store/agent-store.js";
import type { ProcessManager } from "../bridge/process-manager.js";

export function useKeybindings(
  manager: ProcessManager | null,
  inputActive: boolean
): void {
  useInput((_input, key) => {
    const store = useAgentStore.getState();

    // Ctrl+C always exits — top priority
    if (key.ctrl && _input === "c") {
      if (manager?.isRunning) {
        manager.abort();
        manager.stop();
      }
      process.exit(0);
    }

    // Ctrl+L: stop execution and force learner
    if (key.ctrl && _input === "l") {
      if (manager?.isRunning) {
        manager.forceLearn();
      }
      return;
    }

    // Escape: if config screen is open, let its own handler deal with it
    if (key.escape) {
      if (store.showConfig) return;
      if (!manager?.isRunning && !inputActive) {
        process.exit(0);
      }
      return;
    }

    // Don't intercept keys when user is typing
    if (inputActive || store.pendingConfirm || store.showConfig) {
      return;
    }
  });
}
