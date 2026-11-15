# WaveTrack annotation review

WILDTRACK is supported in **read-only** mode. From the dataset workspace, run:

```powershell
python wave_tool/wave_tool.py --dataset wildtrack --source "D:\Data\WILDTRACK"
```

The startup dialog should select `Image_subsets` (or the dataset root),
`calibrations`, and `annotations_positions`. With the default `--dataset auto`,
the tool also detects WILDTRACK by `intrinsic_zero/intr_CVLab1.xml`.
Official calibration names (`CVLab1`–`CVLab4`, `IDIAP1`–`IDIAP3`) and XML
vectors are supported without creating `grid.json`. Images must be the
1920×1080 undistorted `Image_subsets/C1`–`C7` images.

All WILDTRACK frames stay locked, including empty frames and frames without
annotations. Unlock, annotation edits, undo, copy, and saving are disabled.
The tool does not write annotations or `.editor` files into the WILDTRACK
dataset. Startup path preferences are still stored in the tool's `.settings.json`.
Original person IDs and bounding boxes are used for display and crops; cylinder
view is an optional geometric preview. The overview includes all seven cameras
and BEV, with camera shortcuts `1`–`7`. Navigation, playback, selection,
zoom, pan, and display toggles remain available. WaveTrack retains its existing
editing workflow.

To use on another machine, install Python 3.10 or newer with Tkinter
(included in the standard Windows Python installer). Extract the tool, open a
terminal in its folder, and run:

```powershell
python -m pip install -r requirements.txt
python wave_tool.py
```

Choose the image, calibration and annotation folders on that machine in the
startup dialog. Dataset files are supplied separately. Crops and Controls open
by default. Delete `.settings.json` to clear remembered paths before sharing.

Run `python wave_tool/wave_tool.py` from the dataset root. At every startup, a
single dialog asks for three folders: images (dataset root or `Image_subsets`),
calibration (containing `extrinsic/`, `intrinsic_zero/` and `grid.json`), and
per-frame annotation JSON files. Each field supports typing or Browse.
The dialog remembers all three folders; `--source PATH`, `--calib PATH` and
`--labels-dir PATH` override their suggestions for that launch. On first launch,
images default to the dataset root, calibration to `<dataset>/calibrations`, and
annotations to `<dataset>/annotations`. All folders must already exist.
Cancel closes startup without changing settings. Accepted paths are stored in
`wave_tool/.settings.json`.

QA exports open as yellow, unlocked drafts. Filenames must be frame numbers, for
example `00000030.json`. The tool uses `position_m` when present, otherwise the
WaveTrack `positionID` grid. Loading does not rewrite annotation files, including
empty frames. People outside the calibrated ROI remain visible for correction;
move them inside before locking.

Lock replaces that frame's JSON **in the selected folder** with WaveTrack records:
`positionID`, `personID` (non-negative integer or `null`), and `views` for all cameras.
Positions are snapped to the WaveTrack grid and boxes are recomputed from the
cylinders. Existing person IDs are preserved, including QA imports. Metre
positions and custom boxes are replaced when locking.

For WaveTrack, select a person by clicking their cylinder/bbox or BEV marker,
or press **Tab** to cycle people. Unlock the frame if necessary, then click
**Person ID** in Controls or press **r**. Enter a non-negative integer; leave
the field blank to clear it, or Cancel to keep the current ID. Each assigned ID
must be unique within its frame. Use the same ID for the same person across
frames. New people start with no ID; **c** copies previous-frame positions and
IDs. This is manual ID assignment, without automatic identity matching.
IDs appear on camera overlays, BEV, and the selected person's crop sidebar.
**Ctrl+Z** also undoes ID changes. **Lock / Enter** saves IDs with the frame;
opening a file never adds IDs or rewrites it automatically.

Lock status and cylinder dimensions live in `.editor/<frame>.json`. Unlock keeps
the converted annotation and saves the unlocked status, including across restarts.
Edits after unlocking remain in memory until the frame is locked again.
Press **Ctrl+Z** to undo annotation edits in the current frame: add, delete,
move, copy previous, or change cylinder radius/height. A continuous drag is one
undo step. Each frame remembers up to 100 steps during the session. Successful
Lock clears that frame's undo history; undo changes drafts in memory and does
not rewrite saved JSON. Unlock before editing a locked frame.
Existing WaveTrack files with grid-node editor metadata open locked unless their
metadata explicitly says `locked: false`.

Click the number in **Frame [number] / 26070**, type an available frame number,
and press Enter to jump there. Invalid or missing frames, such as `10891`, are
rejected; Escape cancels typing. While editing, number keys and shortcuts belong
to the input field and do not change camera focus or annotations.

Click the arrow next to the number to open the frame list. Use the mouse wheel,
the list scrollbar, arrow keys, or Page Up/Page Down to browse. Click a row or
press Enter after using the arrow keys to select a frame. Only available image
frames appear (`0, 30, 60, ... 26070` for the current subset).
`a`/`d` still move one available image at a time when the input is not focused.
`Shift+A`/`Shift+D` move ten images. Manual keyboard navigation pauses playback;
queued keys are applied in order before rendering the next image.

Run tests from the dataset root with:

```powershell
python -m unittest discover -s wave_tool/tests -t wave_tool
```
