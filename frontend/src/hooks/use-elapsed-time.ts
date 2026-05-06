import { useState, useEffect, useRef } from "react";

export function useElapsedTime(startedAt: string | null): string {
  const [elapsed, setElapsed] = useState("0s");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!startedAt) {
      setElapsed("0s");
      return;
    }

    const start = new Date(startedAt).getTime();

    const update = () => {
      const diff = Math.floor((Date.now() - start) / 1000);
      if (diff < 60) {
        setElapsed(`${diff}s`);
      } else if (diff < 3600) {
        const m = Math.floor(diff / 60);
        const s = diff % 60;
        setElapsed(`${m}m ${s.toString().padStart(2, "0")}s`);
      } else {
        const h = Math.floor(diff / 3600);
        const m = Math.floor((diff % 3600) / 60);
        setElapsed(`${h}h ${m.toString().padStart(2, "0")}m`);
      }
    };

    update();
    intervalRef.current = setInterval(update, 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [startedAt]);

  return elapsed;
}
