// Pure scroll-positioning math for ScrollList. Kept ink-free so it can be
// unit-tested directly under any Node version.

/**
 * Derive `firstVisible` such that the selected row is fully visible. We
 * anchor toward the bottom of the window: `firstVisible` is chosen to be the
 * smallest index where `[firstVisible..selected]` heights still fit in
 * `contentRows`. With keyboard-driven navigation this means the selected row
 * appears near the bottom while moving down, and the top while moving up,
 * which mirrors typical terminal-list UX.
 */
export function deriveFirstVisible(
  heights: number[],
  selectedIndex: number,
  contentRows: number
): number {
  if (heights.length === 0 || contentRows <= 0) return 0;
  if (selectedIndex < 0) return 0;
  const sel = Math.min(heights.length - 1, Math.max(0, selectedIndex));

  let first = sel;
  let used = heights[sel];
  while (first > 0) {
    const prev = heights[first - 1];
    if (used + prev > contentRows) break;
    first--;
    used += prev;
  }
  return first;
}

/**
 * Ensure `firstVisible` doesn't leave the list with empty space at the bottom
 * when items would otherwise still fit. Keeps the window full when scrolled
 * near the end.
 */
export function clampFirstVisible(
  firstVisible: number,
  heights: number[],
  contentRows: number
): number {
  if (heights.length === 0 || contentRows <= 0) return 0;
  let first = Math.max(0, Math.min(firstVisible, heights.length - 1));

  let total = 0;
  for (let i = first; i < heights.length; i++) total += heights[i];
  if (total >= contentRows) return first;

  while (first > 0) {
    const prev = heights[first - 1];
    if (total + prev > contentRows) break;
    first--;
    total += prev;
  }
  return first;
}
