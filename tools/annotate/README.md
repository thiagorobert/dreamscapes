# Hotspot annotator

A dependency-free page for drawing or correcting a scene's hotspot polygon by hand: when
`dsp hotspot` marks a scene `needs_manual_hotspot`, when the segmenter picks the wrong thing, or
to make a hotspot deliberately tighter.

Open `tools/annotate/index.html` directly in a browser (no server, no build step), then:

1. **Open image…** (or drop a file on the page) and pick the scene's master, for example
   `assets/masters/prison-yard.png`. The `assets/review/<scene>/hotspot-<n>.png` overlays
   written by `dsp hotspot` show the automatic outlines it tried, and `hotspot.json` says why
   each was rejected.
2. Click around the bridge object to place vertices. A click near an edge inserts a vertex into
   it; otherwise the new vertex goes after the selected one. Drag vertices to adjust them,
   <kbd>Delete</kbd>/<kbd>Backspace</kbd> removes the selected one, arrow keys nudge it by one
   image pixel (ten with <kbd>Shift</kbd>), <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Z</kbd> undoes.
   **Fit**, **100%** and **200%** change the zoom.
3. Watch the checks: they are the hotspot rules of `dsp validate` (simple polygon, 1.5-12% of
   the frame, centroid inside the dashed safe area), evaluated on the rounded coordinates that
   will be pasted.
4. **Copy YAML** and, under the scene's `bridge:` in `content/scenes.yaml`, replace every hotspot
   field (`polygon:` through `area:`, and any `score:`) with the snippet. Keep `object:` and
   `placement:`. The snippet marks the hotspot `source: manual`, `verified: true`.
5. Run `uv run dsp validate` to confirm.

To edit an existing hotspot, paste its `polygon:` lines (or the whole `bridge:` block, or JSON
`[[x, y], ...]`) into **Edit an existing polygon** and press **Load polygon**. The page also
remembers the last polygon per image file name in the browser's local storage.

`dsp hotspot --all` leaves hand-drawn hotspots (`source: manual`) and verified ones alone;
`--scene <id>` replaces one when asked by name.
