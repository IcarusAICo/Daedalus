import React from "react";
import { Box, Text } from "ink";
import { padEnd, blankLine } from "../utils/term.js";
import { deriveFirstVisible, clampFirstVisible } from "./scroll-list-math.js";

// Re-export so existing imports `from "./scroll-list.js"` still work.
export { deriveFirstVisible, clampFirstVisible };

export interface ScrollListItem {
  /** Unique React key for the row. */
  key: string;
  /** Number of terminal rows this item occupies. Defaults to 1. */
  height?: number;
  /** Render the item. The renderer MUST produce content that occupies exactly `height` rows. */
  render: (props: ScrollListRenderProps) => React.ReactNode;
}

export interface ScrollListRenderProps {
  selected: boolean;
  cols: number;
  index: number;
}

export interface ScrollListProps {
  items: ScrollListItem[];
  /** Selected index, or -1 for no selection. */
  selectedIndex: number;
  /** Total rows the list occupies (including the two indicator rows). */
  viewportRows: number;
  /** Width in columns; rows are padded/truncated to this. */
  viewportCols: number;
  /**
   * Optional controlled first-visible index. When provided, the consumer owns
   * scroll state. When omitted, ScrollList derives `firstVisible` so that the
   * selected row is in view (anchored toward the bottom of the window).
   */
  firstVisible?: number;
  /** Hide the top/bottom indicator rows entirely (gives 2 extra content rows). */
  hideIndicators?: boolean;
}

/**
 * A fixed-height scroll list. Always emits exactly `viewportRows` terminal
 * rows: 1 top indicator (or blank), N content rows, 1 bottom indicator (or
 * blank). Items can declare per-item heights for variable-height rows.
 *
 * The list is fully stateless: scroll position is derived from
 * `(items, selectedIndex)` or controlled via `firstVisible`. This guarantees
 * frame-to-frame stability and avoids the React-state / useEffect timing
 * issues that produced visual artifacts in the previous implementation.
 */
export function ScrollList({
  items,
  selectedIndex,
  viewportRows,
  viewportCols,
  firstVisible,
  hideIndicators,
}: ScrollListProps): React.ReactElement {
  const indicatorRows = hideIndicators ? 0 : 2;
  const contentRows = Math.max(0, viewportRows - indicatorRows);

  // Items with height === 0 are skipped entirely (no row consumed). All other
  // heights are clamped to a minimum of 1.
  const heights = items.map((it) => {
    const h = it.height ?? 1;
    if (h === 0) return 0;
    return Math.max(1, h);
  });

  const start = clampFirstVisible(
    firstVisible ?? deriveFirstVisible(heights, selectedIndex, contentRows),
    heights,
    contentRows
  );

  const visible: { item: ScrollListItem; index: number; height: number }[] = [];
  let used = 0;
  for (let i = start; i < items.length; i++) {
    const h = heights[i];
    if (h === 0) continue;
    if (used + h > contentRows) break;
    visible.push({ item: items[i], index: i, height: h });
    used += h;
  }

  const lastVisibleIdx = visible.length > 0 ? visible[visible.length - 1].index : start - 1;
  const hasAbove = heights.slice(0, start).some((h) => h > 0);
  const hasBelow = heights.slice(lastVisibleIdx + 1).some((h) => h > 0);

  const blanksNeeded = Math.max(0, contentRows - used);

  return (
    <Box flexDirection="column" width={viewportCols} height={viewportRows} flexShrink={0}>
      {!hideIndicators && (
        <Text dimColor>
          {hasAbove ? padEnd("  \u2191 more above", viewportCols) : blankLine(viewportCols)}
        </Text>
      )}
      {visible.map(({ item, index }) => (
        <React.Fragment key={item.key}>
          {item.render({ selected: index === selectedIndex, cols: viewportCols, index })}
        </React.Fragment>
      ))}
      {Array.from({ length: blanksNeeded }, (_, i) => (
        <Text key={`blank-${i}`}>{blankLine(viewportCols)}</Text>
      ))}
      {!hideIndicators && (
        <Text dimColor>
          {hasBelow ? padEnd("  \u2193 more below", viewportCols) : blankLine(viewportCols)}
        </Text>
      )}
    </Box>
  );
}

