import { describe, it, expect } from "vitest";
import { deriveFirstVisible, clampFirstVisible } from "./scroll-list-math.js";

describe("deriveFirstVisible", () => {
  it("returns 0 for empty list", () => {
    expect(deriveFirstVisible([], 0, 10)).toBe(0);
  });

  it("returns 0 when contentRows <= 0", () => {
    expect(deriveFirstVisible([1, 1, 1], 2, 0)).toBe(0);
    expect(deriveFirstVisible([1, 1, 1], 2, -1)).toBe(0);
  });

  it("returns 0 for negative selectedIndex", () => {
    expect(deriveFirstVisible([1, 1, 1, 1], -1, 2)).toBe(0);
  });

  it("anchors selected near the bottom of the viewport", () => {
    // 10 fixed-height-1 items; viewport holds 4. Selecting index 9 (last)
    // should put indices 6,7,8,9 in view.
    const heights = Array(10).fill(1);
    expect(deriveFirstVisible(heights, 9, 4)).toBe(6);
  });

  it("clamps to 0 when selected fits entirely from the top", () => {
    const heights = Array(10).fill(1);
    expect(deriveFirstVisible(heights, 2, 4)).toBe(0);
  });

  it("never produces a window taller than contentRows", () => {
    const heights = [1, 5, 1, 1, 1];
    const fv = deriveFirstVisible(heights, 4, 3);
    let used = 0;
    for (let i = fv; i <= 4; i++) used += heights[i];
    expect(used).toBeLessThanOrEqual(3);
  });

  it("handles a single tall item", () => {
    const heights = [10];
    expect(deriveFirstVisible(heights, 0, 5)).toBe(0);
  });

  it("clamps selectedIndex past the end", () => {
    const heights = [1, 1, 1];
    expect(deriveFirstVisible(heights, 99, 2)).toBe(1);
  });

  it("places selected at top when preceding items don't fit at all", () => {
    // Item 0 alone is taller than viewport; selecting item 1 should anchor
    // at item 1.
    const heights = [10, 1, 1];
    expect(deriveFirstVisible(heights, 1, 3)).toBe(1);
  });
});

describe("clampFirstVisible", () => {
  it("returns 0 for empty list", () => {
    expect(clampFirstVisible(0, [], 5)).toBe(0);
    expect(clampFirstVisible(99, [], 5)).toBe(0);
  });

  it("never lets the window go negative", () => {
    expect(clampFirstVisible(-5, [1, 1, 1], 2)).toBe(0);
  });

  it("respects firstVisible when items extend past the bottom", () => {
    const heights = Array(10).fill(1);
    expect(clampFirstVisible(3, heights, 4)).toBe(3);
  });

  it("walks back when items don't fill the viewport", () => {
    const heights = Array(5).fill(1);
    // firstVisible=4 leaves only 1 item, contentRows=3 wants 3 → walk back to 2
    expect(clampFirstVisible(4, heights, 3)).toBe(2);
  });

  it("walks back to 0 when total items are less than viewport", () => {
    const heights = [1, 1];
    expect(clampFirstVisible(1, heights, 5)).toBe(0);
  });

  it("preserves firstVisible when exactly enough items remain", () => {
    const heights = Array(10).fill(1);
    expect(clampFirstVisible(7, heights, 3)).toBe(7);
  });

  it("handles variable heights when walking back", () => {
    const heights = [1, 1, 1, 1, 5];
    // firstVisible=4 (the tall one), it consumes 5 — already > 3, no walk back needed
    expect(clampFirstVisible(4, heights, 3)).toBe(4);
  });

  it("respects total height when walking back from end", () => {
    const heights = [1, 1, 1, 1, 1];
    // firstVisible=3, items 3,4 total height 2; viewport=4; walk back to include 1,2 → first=1
    expect(clampFirstVisible(3, heights, 4)).toBe(1);
  });
});
