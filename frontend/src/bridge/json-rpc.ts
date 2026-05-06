import { ChildProcess, spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { createInterface, Interface } from "node:readline";
import type { TraceEvent } from "../store/types.js";

export interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params: Record<string, unknown>;
}

export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

type IncomingMessage = JsonRpcNotification | JsonRpcRequest;

export class JsonRpcBridge extends EventEmitter {
  private process: ChildProcess | null = null;
  private reader: Interface | null = null;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();

  constructor(
    private command: string,
    private args: string[],
    private cwd?: string
  ) {
    super();
  }

  start(): void {
    this.process = spawn(this.command, this.args, {
      cwd: this.cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    if (!this.process.stdout || !this.process.stdin) {
      throw new Error("Failed to open stdio pipes to backend");
    }

    this.reader = createInterface({ input: this.process.stdout });
    this.reader.on("line", (line) => this.handleLine(line));

    this.process.stderr?.on("data", (chunk: Buffer) => {
      this.emit("stderr", chunk.toString());
    });

    this.process.on("exit", (code) => {
      this.emit("exit", code);
      this.cleanup();
    });

    this.process.on("error", (err) => {
      this.emit("error", err);
    });

    this.emit("connected");
  }

  private handleLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;

    let msg: IncomingMessage | JsonRpcResponse;
    try {
      msg = JSON.parse(trimmed);
    } catch {
      this.emit("stderr", `[bridge] non-JSON line: ${trimmed}\n`);
      return;
    }

    if ("id" in msg && ("result" in msg || "error" in msg)) {
      const pending = this.pending.get(msg.id);
      if (pending) {
        this.pending.delete(msg.id);
        if (msg.error) {
          pending.reject(new Error(msg.error.message));
        } else {
          pending.resolve(msg.result);
        }
      }
      return;
    }

    if ("method" in msg) {
      if ("id" in msg) {
        this.emit("request", msg as JsonRpcRequest);
      } else {
        this.emit("notification", msg as JsonRpcNotification);
      }
    }
  }

  send(method: string, params: Record<string, unknown> = {}): void {
    const notification: JsonRpcNotification = {
      jsonrpc: "2.0",
      method,
      params,
    };
    this.write(notification);
  }

  async call(
    method: string,
    params: Record<string, unknown> = {}
  ): Promise<unknown> {
    const id = this.nextId++;
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.write(request);
    });
  }

  respond(id: number, result: unknown): void {
    const response: JsonRpcResponse = {
      jsonrpc: "2.0",
      id,
      result,
    };
    this.write(response);
  }

  respondError(id: number, code: number, message: string): void {
    const response: JsonRpcResponse = {
      jsonrpc: "2.0",
      id,
      error: { code, message },
    };
    this.write(response);
  }

  private write(msg: object): void {
    if (!this.process?.stdin?.writable) return;
    this.process.stdin.write(JSON.stringify(msg) + "\n");
  }

  stop(): void {
    if (this.process) {
      this.process.kill("SIGTERM");
      setTimeout(() => {
        if (this.process && !this.process.killed) {
          this.process.kill("SIGKILL");
        }
      }, 3000);
    }
    this.cleanup();
  }

  private cleanup(): void {
    this.reader?.close();
    this.reader = null;
    for (const [, pending] of this.pending) {
      pending.reject(new Error("Bridge closed"));
    }
    this.pending.clear();
  }

  get isRunning(): boolean {
    return this.process !== null && !this.process.killed;
  }
}
