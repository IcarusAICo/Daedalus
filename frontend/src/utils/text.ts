// Pure text utilities — no ink dependency, safe to import from tests.

const ANSI_RE = /\u001b\[[0-9;]*[a-zA-Z]/g;

export function stripAnsi(s: string): string {
  return s.replace(ANSI_RE, "");
}

export function visualWidth(s: string): number {
  return [...stripAnsi(s)].length;
}

export function truncateVisual(s: string, cols: number): string {
  if (cols <= 0) return "";
  const stripped = stripAnsi(s);
  const chars = [...stripped];
  if (chars.length <= cols) return s;
  if (cols === 1) return chars[0];
  return chars.slice(0, cols - 1).join("") + "\u2026";
}

export function padEnd(s: string, cols: number): string {
  const w = visualWidth(s);
  if (w === cols) return s;
  if (w > cols) return truncateVisual(s, cols);
  return s + " ".repeat(cols - w);
}

export function blankLine(cols: number): string {
  return " ".repeat(Math.max(0, cols));
}

/**
 * Word-wrap `text` into at most `maxLines` lines of width `cols`. Long words
 * are hard-broken. If the text needs more lines than `maxLines`, the last line
 * gets an ellipsis suffix to indicate truncation.
 */
export function wrapLines(text: string, cols: number, maxLines: number): string[] {
  if (!text || cols <= 0 || maxLines <= 0) return [];
  const out: string[] = [];
  const paragraphs = text.split(/\r?\n/);
  let truncated = false;

  outer: for (let p = 0; p < paragraphs.length; p++) {
    const para = paragraphs[p];
    if (para.length === 0) {
      if (out.length < maxLines) out.push("");
      if (out.length >= maxLines && p < paragraphs.length - 1) {
        truncated = true;
        break;
      }
      continue;
    }
    const words = para.split(/\s+/).filter(Boolean);
    let cur = "";
    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi];
      if (visualWidth(w) > cols) {
        if (cur) {
          out.push(cur);
          cur = "";
          if (out.length >= maxLines) {
            if (wi < words.length - 1 || p < paragraphs.length - 1) truncated = true;
            break outer;
          }
        }
        const chars = [...stripAnsi(w)];
        for (let i = 0; i < chars.length; i += cols) {
          out.push(chars.slice(i, i + cols).join(""));
          if (out.length >= maxLines) {
            if (i + cols < chars.length || wi < words.length - 1 || p < paragraphs.length - 1) {
              truncated = true;
            }
            break outer;
          }
        }
      } else if (!cur) {
        cur = w;
      } else if (visualWidth(cur) + 1 + visualWidth(w) <= cols) {
        cur += " " + w;
      } else {
        out.push(cur);
        if (out.length >= maxLines) {
          if (wi < words.length || p < paragraphs.length - 1) truncated = true;
          break outer;
        }
        cur = w;
      }
    }
    if (cur) {
      out.push(cur);
      if (out.length >= maxLines && p < paragraphs.length - 1) {
        truncated = true;
        break;
      }
    }
  }

  if (truncated && out.length > 0) {
    const last = out[out.length - 1];
    out[out.length - 1] = truncateVisual(last + " \u2026", cols);
  }

  return out.slice(0, maxLines);
}
