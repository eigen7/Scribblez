import { useEffect, useState } from 'react';

// Holds `value` back until it stops changing for `delayMs`, so a fast-changing
// input (e.g. a generation slider being dragged) doesn't fire one network
// request per intermediate value it passes through.
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}
