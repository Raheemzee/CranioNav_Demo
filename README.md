# CranioNav — navigated brain tumour removal

A guided simulation of brain tumour removal on real BraTS MRI, done entirely in the navigation view.
The navigation shows the brain, the tumour, the eloquent areas, the skull and the venous sinuses
from the MRI, and is aligned to the patient's head. You then open the head layer by layer and
remove the tumour in that view, while the navigation tracks your instrument, warns you near
eloquent cortex and measures the extent of resection.

Training and research simulator. **Not for clinical use.**

## Run (Windows, CPU only)

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5060 in Chrome or Edge.

1. Edit `DATA_DIR` near the top of `app.py` so it points at your BraTS folder. You can also set
   the environment variable `CRANIONAV_DATA` instead.
2. Open **Case** and pick a case. The first time, the case is prepared in about 35–45 s on a
   normal CPU, with progress shown on screen. Prepared cases are cached in `cases\`.
3. To rebuild a case, delete its folder in `cases\` and open it again.

Do not open `index.html` straight from disk. Browsers block ES modules on `file://`, so use the
server.

## Demo link on Render

The repository deploys as a free Render static site: the browser does all the work on the two
bundled cases, so the link opens at once (nothing to start, no sleeping server) and there is
nothing that can fail at deploy time (no gunicorn, no database, no Python).

1. Put this folder in a GitHub repository (all of it, including `cases\`):
   ```
   git init
   git add .
   git commit -m "CranioNav demo"
   git branch -M main
   git remote add origin https://github.com/<you>/cranionav.git
   git push -u origin main
   ```
2. In Render: **New → Blueprint**, pick the repository, then **Apply**. `render.yaml` creates the
   static site `cranionav-demo`; its address (`https://cranionav-demo.onrender.com`, or with a
   suffix if the name is taken) is the demo link.
3. Every push to `main` redeploys it.

To show more cases in the demo, open them once locally (`python app.py`) so they are prepared into
`cases\`, then commit and push.

If the demo cannot show its cases, the Case tab and the loading screen say why, with the file
that failed (for example `cases/index.json answered 404`): almost always, the `cases` folder did
not reach the repository. The Render build also stops with a clear message when
`cases/index.json` is missing. On a slow connection the first visit downloads about 7 MB for the
app and 7 MB per case, with the progress on the loading screen; later visits use the browser's
copy.

If you prefer a Python web service (like a Flask app with gunicorn), `render.yaml` has the
settings at the bottom: `pip install -r requirements-render.txt`, then
`gunicorn app:app --worker-class gthread --workers 1 --threads 8 --timeout 120`, health check
`/healthz`. On Render, `app.py` runs as a read-only demo: it serves the bundled cases (compressed),
builds nothing and writes nothing, and needs no database or setup before its first request. The
free plan's web service sleeps after 15 minutes without visitors, and the next visit then waits
about a minute; the static site does not.

## The five steps

Every step happens in the navigation view. Each step tells you what to do in the **Guide** panel and
in the bar at the bottom. Every step has a **Do it for me** button if you only want to watch. It
shows the step being done in the navigation view:
- **Open**: the stage's instrument does the stage in front of you. The pen draws the marking,
  the scalpel follows it, the drill works each burr hole, the craniotome joins the holes before
  the flap lifts, and the scissors open the dura. Each stage takes about 5 seconds.
- **Remove**: the aspirator works through the cavity, nearest tissue first, in about 8 seconds.

While a demonstration runs, the button turns into **Stop**. Stopping keeps what was done so far,
and you carry on with the mouse.

1. **Plan**: the approach and the craniotomy are planned for you. The planner aims for the least
   brain to cross and keeps away from eloquent cortex. The plan shows:
   - the path, the brain you would cross, eloquent cortex on the way, and the clearance from the
     venous sinuses;
   - a warning if the bone flap would cross a sinus.

   To change the plan, click the scalp in the 3D view, Shift-click a slice to move the target, or
   change the margin. Then press **Approve the plan**.
2. **Align**: press **Scan the face and align**. A simulated face scanner captures the face and
   matches it to the scalp from the MRI, so the navigation knows where the head is. The scanned
   points float off the scalp until the alignment is computed, then slide onto it. You can check
   accuracy by touching a bony point with the pointer. Landmark and surface-trace alignment are
   under *Other ways to align*.
3. **Open**: five layers, each with its own instrument, picked up for you, and a hologram guide
   line:
   1. Mark the incision with the pen.
   2. Open the scalp with the scalpel along your marking.
   3. Drill three burr holes on the cyan rings.
   4. Join the holes with the craniotome and lift the bone flap.
   5. Open the dura with the scissors.
4. **Remove**: hold the left button to aspirate where the ultrasonic aspirator touches. Tissue
   really is removed from a 1 mm voxel model of the case, and the cavity is drawn in 3D. While you
   work:
   - The tumour and the eloquent cortex show through the opening, over the exposed brain, and the
     holograms and the three slices show what is left.
   - The gauge and the on-screen readout show the tissue at the tip, the distance to the tumour
     margin, the nearest eloquent area and the ventricle.
   - A tone speeds up near eloquent cortex, and there are warnings when normal eloquent tissue
     is removed or the ventricle is opened.
5. **Review**:
   - extent of resection, with gross total / near total / subtotal;
   - tumour left, normal brain removed, and eloquent cortex injured;
   - alignment error, including the true error, which only the simulation knows;
   - marking accuracy, tracking losses and time;
   - an axial slice through the residual tumour.

   Sessions save as JSON to `sessions\`.

The resection target follows the tumour type. For enhancing or cystic lesions (high-grade glioma,
metastasis) it is the core: enhancing plus necrotic. For non-enhancing lesions (low-grade-like)
it is the whole FLAIR abnormality.

## The navigation view

There is one 3D view, the navigation view, beside three slices that follow the instrument tip. It is
in image space, with superior up, and frames itself for each step: the plan, the face while you
align, the operative field while you open, and the opening while you remove.

The card at the top left of the view (also in the **Layers** tab) sets how the scalp and skull look:
- **Hologram**: a glassy scalp and skull with the brain, tumour and other holograms inside. From the
  Open step on, the real scalp, bone, dura and brain show in the operative field around the
  craniotomy, so you see every layer you cut.
- **Opaque**: the real scalp and skull of the patient, as in theatre.

Each has a transparency slider for the scalp and one for the skull. Lower them to see the skull
through the scalp, and the brain and tumour through both.

The skull is a real one: the bone surface of a CT scan (orbits, cheekbones, jaw, skull base,
sutures and foramina), shaded with the occlusion of the CT so hollows and ridges read under any
light. Each case fits it under the face and around the brain (see *What is real and what is
estimated*). When it is see-through, only its nearest surface shows, evenly, like glass.

The blue tip and line are the navigated instrument: where the navigation puts it through the
alignment. Before aligning they sit off the real instrument. Once aligned they sit on its tip.

### Pan, zoom and rotate

Rotation follows the mouse: drag right and the side of the patient facing you moves right, drag
down and it moves down (right-drag anywhere, or left-drag away from the instrument). There are two
kinds, under **Layers → Navigation view → Rotation**:
- **Upright** (the default): left and right turn the patient about its superior axis, up and down
  tip it towards or away from you; superior stays up on the screen, as on a PACS 3D view.
- **Free**: the patient turns about the screen's own axes, like a ball in the hand, from any side;
  the picture can tilt, and **Reset view** (or the next step) brings superior back up.

Pan moves the picture like a PACS viewer: the patient follows the mouse exactly and keeps its look,
and rotating or zooming afterwards turns the patient where you moved it. **Reset view** brings it
back to the centre. To pan:
- press the **hand button** above the zoom bar (or **H**), then drag with the left button anywhere,
  even over the patient; press it again to go back to the instrument;
- or drag with the middle button, Shift + drag (left button off the instrument, or right button),
  or Space + drag;
- or use the arrow keys: the picture moves the way the arrow points.

### Undo and redo

The **undo** and **redo** buttons at the top right (also in focus mode), or **Ctrl+Z** and
**Ctrl+Y** (or Ctrl+Shift+Z), step back and forward through the procedure: plan changes, approving
the plan, aligning, landmarks and accuracy checks, each pen stroke, cut, burr hole and finished
stage, each press of the aspirator, the *Do it for me* demonstrations and moving to the next step.
The button's tooltip names what it will undo. The view (camera, looks and layers) is not part of
it. Up to 40 steps are kept; opening another case starts afresh.

| Key | |
|---|---|
| **F** | Browser full screen (the button at the top right) |
| **I** | Focus mode: hides every panel, keeps a small step card and buttons over the 3D view; the slices button brings the slices back beside it |
| **M** | Maximise the 3D view; each slice also has a maximise button (or double-click its header) |
| **Esc** | Leave a maximised view or focus mode |

## Controls

| Input | Action |
|---|---|
| Mouse | Moves the instrument over the patient (it rests on whatever layer is exposed) |
| Hold left button on the patient | Use the instrument: draw, cut, drill, aspirate |
| Right-drag, or left-drag away from the patient | Rotate the view |
| Hand button or **H**, then drag | Pan the view (left button, anywhere) |
| Middle-drag, Shift + drag, or Space + drag | Pan the view |
| Arrow keys | Pan the view the way the arrow points (hold Shift for bigger steps) |
| Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) | Undo / redo |
| Zoom bar (right edge of the 3D view) | Drag, or press + and −, to zoom; it follows the wheel |
| + / − keys | Zoom in or out |
| Wheel | Zoom in or out (Shift + wheel sets the pointer's virtual extension) |
| Reset view button | Frame the view for the current step again |
| Shift + move | Tilt the instrument about its tip |
| C | Collect a landmark or check accuracy while aligning |
| P | Switch between the pointer and the step's instrument |
| [ ] | Aspirator tip size |

The brain hologram is drawn from the case's own MRI: a see-through grey cortex with its real gyri
and dark sulci, eloquent areas painted on, and the tumour in red. Where the tumour lies under the
cortex, the cortex thins so the red shows through. The opacity slider sets how see-through it is.

The **Layers** tab has the scalp and skull look and switches each hologram on or off:
- cortex, eloquent areas and tumour, with the brain look: **MRI (realistic)** or **Outline**
  (a lighter model of the brain surface);
- oedema and ventricles;
- the skull (the real CT skull, see-through as its slider says) and the sinuses;
- the plan and guide lines;
- the slice overlays.

If a graphics driver cannot draw the MRI brain, CranioNav switches to the Outline brain and says
so on screen, instead of showing an empty hologram.

The **Case** tab has the case list and training options:
- block the tracking camera (the instrument is then lost), add tracking jitter or bump the
  reference array;
- the instructor view with the true errors;
- the proximity tone and quality settings.

## Performance (PCs without a graphics card)

CranioNav picks a quality level from the graphics it finds. You can change it under
**Case → Performance**:

| Level | For | What changes |
|---|---|---|
| **Smooth** | PCs without a graphics card, or with graphics turned off (software rendering) | Simpler lighting, no reflections or skin detail maps, lower-resolution textures, fewer slice refreshes |
| **Balanced** | Laptops and integrated graphics (Intel, AMD) | Full lighting and reflections, no edge smoothing |
| **Best quality** | A dedicated graphics card | Edge smoothing, higher resolution |

What keeps it smooth, at every level except Best quality:
- The 3D view redraws only when something changes. An idle screen costs almost nothing.
- The patient's layers and the holograms are drawn once and reused. Each frame only adds what
  moves: the instruments, the ink and cut lines, and the navigated instrument.
- Aspirating redraws only the few millimetres of screen around the removed tissue.
- Only what can be seen is drawn: an opaque scalp or skull hides the holograms inside, the skull
  is drawn whole only when it shows through the scalp, and then from a lighter copy (lighter
  still when the skull itself is see-through).
- Panning shifts the picture already drawn; it is redrawn once, when you let go.
- While the view moves or tissue is removed, the resolution drops to keep frames coming. It returns
  to full sharpness about a second after you stop.
- Hiding the 3D view (maximising a slice) stops it from drawing.

Measured in the worst case, software rendering on two CPU cores with no graphics at all:

| | Frames per second |
|---|---|
| Moving the instrument (any step, either look) | 14–17 |
| Aspirating tumour (holding the button) | about 6 |
| Rotating the view (hologram look) | about 6 |
| Panning the view | 12–23 |
| Idle | 45–50, at full resolution |
| Start-up | 4.5 s, then 2.5 s to open a case (fitting the CT skull takes about 0.5 s of that) |

A PC with integrated graphics, which most laptops and desktops have, is many times faster than
this.

Edge smoothing (antialiasing) is fixed when the page loads. After switching to or from Best
quality, the Performance section offers a reload.

## BraTS data

Any folder layout works up to three levels deep. Each case folder needs these four
modalities; the segmentation is optional:

| Release | File names |
|---|---|
| BraTS 2017–2021 | `*_flair`, `*_t1`, `*_t1ce`, `*_t2`, `*_seg` (labels 1 NCR, 2 ED, 4 ET) |
| BraTS 2023+ (GLI, MET, MEN, PED, SSA) | `*-t2f`, `*-t1n`, `*-t1c`, `*-t2w`, `*-seg` (label 3 = ET) |

If a case has no `seg` file, the tumour is segmented automatically. By default this is an
intensity heuristic. For better results, set `SEGRESNET_CHECKPOINT` to a MONAI SegResNet `.pth`
file (4 channels FLAIR, T1, T1CE, T2) and install `torch` and `monai`.

## What is real and what is estimated

| | Source | Checked against |
|---|---|---|
| MRI, tumour labels | The BraTS case | — |
| Scalp | A real 3D head scan, fitted around the brain | Real scalp MRI of TCGA-DU-7294: mean error 6 mm |
| Skull | A real CT skull (TotalSegmentator, subject s1397), carried under the face by the head fit. Its vault is then mapped onto the skull estimated around this brain (inner table just outside the brain, outer table on the estimate); the bone is kept at least 2 mm clear of the brain and 1.5 mm under the skin | Vault within 0.7 mm (TCGA-DU-7294) and 0.8 mm (BraTS-MET-00110-000) of the estimate on average; the estimate itself is within 2.2 mm mean (+0.8 mm median) of the outer table on a real MRI with scalp |
| Dura | Estimated around the brain from regional distances measured on a real MRI with scalp | Same MRI |
| Venous sinuses | Superior sagittal sinus along the inner table in the midline; transverse sinuses from the torcula | Anatomical rules only |
| Eloquent areas | Harvard-Oxford atlas registered to the case (motor, sensory, Broca, Wernicke, visual) | **An estimate, not fMRI or DTI** |
| Face-scan alignment | Simulated face-scanner depth scan plus multi-start ICP | True error at the tumour at most 0.44 mm (TCGA-DU-7294) and 0.55 mm (BraTS-MET-00110-000) over 30 random starting misalignments each |

BraTS volumes are skull-stripped, so the scalp, skull, dura and sinuses can only be estimated.
Real tumours also shift and distort the cortex, which an atlas cannot follow.

## Files

| Path | Purpose |
|---|---|
| `app.py` | BraTS pipeline (labels, atlas, head fit, skull, dura, sinuses, plan, meshes) and the Flask server (also `gunicorn app:app`) |
| `render.yaml` | Render Blueprint for the demo link (static site; a web service alternative is in its comments) |
| `requirements-render.txt` | What the web service alternative needs on Render (Flask, gunicorn, NumPy) |
| `index.html` | The whole simulator: navigation view, patient layers, holograms, tracking, alignment, removal, review |
| `atlas/` | MNI152 template, brain mask, Harvard-Oxford cortical and ventricle atlases |
| `assets/head/` | 3D head scan (glTF JSON and textures) |
| `assets/skull/` | The CT skull: three levels of detail with baked shading (`skull_s1397.png`, the data carried losslessly in a PNG so any web host serves it) |
| `vendor/three/` | three.js r170, bundled so it runs offline |
| `cases/` | Prepared cases (two samples included) |

## Credits and licences

- Patient head: "Infinite, 3D Head Scan" by Lee Perry-Smith, Infinite-Realities, licensed
  CC BY 3.0 (`assets/head/LeePerrySmith_License.txt`).
- Skull: derived from subject s1397 of the TotalSegmentator dataset (Wasserthal et al., Radiology:
  AI 2023), licensed CC BY 4.0 (`assets/skull/LICENSE.txt` lists the changes made).
- three.js, MIT licence (`vendor/three/LICENSE`).
- MNI152 template and Harvard-Oxford atlases, as packaged by neuroparc (Apache 2.0,
  `atlas/NEUROPARC_LICENSE.md`). The Harvard-Oxford atlas is distributed with FSL for
  non-commercial research use. For a commercial product, check its licence or replace it with an
  atlas you are licensed to ship.
- Sample cases:
  - TCGA-DU-7294 (BraTS 2021 format) from the BraTS-Toolkit example data.
  - BraTS-MET-00110-000 from the AURORA example data.

  BraTS data comes under its own terms of use, which require citing the BraTS papers.
