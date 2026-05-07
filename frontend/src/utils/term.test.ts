import { describe, it, expect } from "vitest";
import { visualWidth, truncateVisual, padEnd, wrapLines, stripAnsi } from "./text.js";

describe("visualWidth", () => {
  it("counts ASCII length", () => {
    expect(visualWidth("hello")).toBe(5);
  });

  it("ignores ANSI escape codes", () => {
    expect(visualWidth("\u001b[31mhello\u001b[0m")).toBe(5);
  });

  it("counts arrow glyphs as one column each", () => {
    expect(visualWidth("\u2190\u2192\u2191\u2193")).toBe(4);
  });

  it("counts box-drawing chars as one column each", () => {
    expect(visualWidth("\u2500".repeat(40))).toBe(40);
  });

  it("returns 0 for empty string", () => {
    expect(visualWidth("")).toBe(0);
  });
});

describe("stripAnsi", () => {
  it("removes color codes", () => {
    expect(stripAnsi("\u001b[31mred\u001b[0m")).toBe("red");
  });

  it("removes complex sequences", () => {
    expect(stripAnsi("\u001b[1;31;48;5;234mfoo\u001b[0m bar")).toBe("foo bar");
  });
});

describe("truncateVisual", () => {
  it("returns input unchanged when fits", () => {
    expect(truncateVisual("hi", 5)).toBe("hi");
  });

  it("truncates with ellipsis when too long", () => {
    expect(truncateVisual("hello world", 5)).toBe("hell\u2026");
  });

  it("returns empty string when cols is 0", () => {
    expect(truncateVisual("anything", 0)).toBe("");
  });

  it("returns first char when cols is 1", () => {
    expect(truncateVisual("hello", 1)).toBe("h");
  });
});

describe("padEnd", () => {
  it("right-pads short strings to exact width", () => {
    expect(padEnd("hi", 5)).toBe("hi   ");
  });

  it("returns input as-is when already exact width", () => {
    expect(padEnd("hello", 5)).toBe("hello");
  });

  it("truncates strings longer than cols", () => {
    expect(padEnd("hello world", 5)).toBe("hell\u2026");
  });

  it("ignores ANSI when measuring width", () => {
    const colored = "\u001b[31mhi\u001b[0m";
    const out = padEnd(colored, 5);
    // 2 visible chars + 3 spaces of padding (ANSI escapes are zero-width).
    expect(visualWidth(out)).toBe(5);
  });

  it("produces an exact-width line for box-drawing dividers", () => {
    const divider = "  " + "\u2500".repeat(40);
    const out = padEnd(divider, 80);
    expect(visualWidth(out)).toBe(80);
  });
});

describe("wrapLines", () => {
  it("returns empty array for empty input", () => {
    expect(wrapLines("", 10, 5)).toEqual([]);
  });

  it("returns single line when text fits", () => {
    expect(wrapLines("hello world", 20, 5)).toEqual(["hello world"]);
  });

  it("word-wraps to multiple lines", () => {
    const lines = wrapLines("the quick brown fox jumped over", 10, 10);
    expect(lines.length).toBeGreaterThan(1);
    for (const line of lines) {
      expect(visualWidth(line)).toBeLessThanOrEqual(10);
    }
  });

  it("respects existing newlines", () => {
    const lines = wrapLines("line one\nline two", 20, 5);
    expect(lines).toEqual(["line one", "line two"]);
  });

  it("truncates with ellipsis when content exceeds maxLines", () => {
    const lines = wrapLines("aa bb cc dd ee ff gg hh", 5, 2);
    expect(lines.length).toBe(2);
    expect(lines[1].endsWith("\u2026")).toBe(true);
  });

  it("hard-breaks long words", () => {
    const lines = wrapLines("supercalifragilisticexpialidocious", 5, 10);
    expect(lines.length).toBeGreaterThan(1);
    for (const line of lines) {
      expect(visualWidth(line)).toBeLessThanOrEqual(5);
    }
  });

  it("returns empty array when cols or maxLines is non-positive", () => {
    expect(wrapLines("hello", 0, 5)).toEqual([]);
    expect(wrapLines("hello", 5, 0)).toEqual([]);
  });
});
