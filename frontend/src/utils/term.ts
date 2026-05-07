import { useWindowSize, measureElement, type DOMElement } from "ink";
import { useLayoutEffect, useRef, useState } from "react";

// Re-export pure text utilities so existing imports `from "../utils/term.js"`
// continue to work. Tests should import from `./text.js` directly to avoid
// pulling in Ink (which uses ES2024 regex flags unsupported on older Node).
export {
  stripAnsi,
  visualWidth,
  truncateVisual,
  padEnd,
  blankLine,
  wrapLines,
} from "./text.js";

export interface ViewportSize {
  rows: number;
  cols: number;
}

const FALLBACK_ROWS = 24;
const FALLBACK_COLS = 80;
const MIN_ROWS = 10;
const MIN_COLS = 40;

export function useViewport(): ViewportSize {
  const size = useWindowSize();
  const rows = Number.isFinite(size?.rows) && size.rows > 0 ? size.rows : FALLBACK_ROWS;
  const cols = Number.isFinite(size?.columns) && size.columns > 0 ? size.columns : FALLBACK_COLS;
  return {
    rows: Math.max(MIN_ROWS, rows),
    cols: Math.max(MIN_COLS, cols),
  };
}

/**
 * Measure a `<Box>` after layout. Returns a ref to attach to the box and the
 * latest measured size. The size starts at the provided fallback so the first
 * frame renders sensibly; it then updates whenever any of `deps` changes.
 *
 * Use this for components whose available space depends on sibling layout
 * (e.g. ChatFeed sitting next to a variable-height GoalDisplay or spinner).
 */
export function useMeasuredBox(
  fallback: ViewportSize,
  deps: readonly unknown[] = []
): { ref: React.RefObject<DOMElement | null>; size: ViewportSize } {
  const ref = useRef<DOMElement | null>(null);
  const [size, setSize] = useState<ViewportSize>(fallback);

  useLayoutEffect(() => {
    if (!ref.current) return;
    const measured = measureElement(ref.current);
    if (measured.height > 0 && measured.width > 0) {
      setSize({
        rows: Math.max(1, measured.height),
        cols: Math.max(1, measured.width),
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ref, size };
}
