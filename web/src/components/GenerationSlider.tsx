import React from 'react';

// A generation slider + "Latest" checkbox, shared across React dashboard tabs.
//
// The checkbox tracks "is the slider at the newest generation": sliding away from
// the rightmost unchecks it, sliding back rechecks it, and (re)checking it jumps to
// the newest. Unchecking keeps the knob where it is. The slider is never disabled.
// This mirrors the Bokeh `_gen_controls` so the React and Bokeh dashboards behave
// identically.
//
// Controlled: the parent owns `valueIdx` (the slider index) and `follow` (Latest),
// and `onChange(idx, follow)` reports both at once. To follow new generations while
// Latest is checked, the parent pins `valueIdx` to the newest as the count grows.
export default function GenerationSlider({
  count,
  valueIdx,
  follow,
  onChange,
  label,
}: {
  count: number;
  valueIdx: number;
  follow: boolean;
  onChange: (idx: number, follow: boolean) => void;
  label?: React.ReactNode;
}) {
  const end = Math.max(count - 1, 0);
  return (
    <span className="gen-slider">
      <input
        type="range"
        min={0}
        max={end}
        step={1}
        value={Math.min(valueIdx, end)}
        onChange={(e) => {
          const v = Number(e.target.value);
          onChange(v, v >= end); // sliding to the rightmost (re)checks Latest
        }}
      />
      {label != null && <span className="muted gen-label">{label}</span>}
      <label className="gen-latest">
        <input
          type="checkbox"
          checked={follow}
          onChange={(e) => onChange(e.target.checked ? end : valueIdx, e.target.checked)}
        />
        Latest
      </label>
    </span>
  );
}
