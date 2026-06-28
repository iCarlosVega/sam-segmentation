# SAM Segmentation — LichtFeld Studio Plugin

Text-prompted object segmentation for sequential image datasets, producing per-frame
binary masks ready for mask-aware Gaussian Splatting training in LichtFeld Studio.

Type something like `"action figure"` or `"person sleeping"`, and the plugin writes one
`.png` mask per frame into your dataset's `masks/` directory. LichtFeld's training
panel picks them up automatically — set **Mask Mode** to `segment` and only the
prompted object survives training.

---

## What's inside

The plugin combines two models:

| Stage | Model | Purpose |
|---|---|---|
| 1 | **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) | Text prompt + first frame → bounding box around matching object |
| 2 | **SAM 2** (`facebook/sam2-hiera-small`) | Bounding box → segmentation mask, propagated across every frame |

This is the "Grounded SAM 2" pattern: text understanding via Grounding DINO, dense
mask quality via SAM 2's video tracking. The full pipeline runs locally on your GPU.

---

## Requirements

- **LichtFeld Studio** `≥ 0.5.2`
- **Plugin API** `1.x`
- **Python** `3.12+` (provided by LichtFeld's embedded interpreter)
- **GPU**: CUDA-capable; tested on RTX 5070 (12 GB). 4–8 GB is enough for the `small` model variant.
- **OS**: Windows 10/11, Linux. Apple Silicon untested.
- **Disk**: ~3 GB for dependencies + ~1.2 GB for model weights downloaded on first run.

---

## Install

### From the LichtFeld plugin marketplace (recommended once listed)

Open **Settings → Plugins → Marketplace**, find `SAM Segmentation`, click **Install**.

### Manual install (current path)

```powershell
git clone https://github.com/iCarlosVega/sam-segmentation.git "$env:USERPROFILE\.lichtfeld\plugins\sam_segment"
```

Then in LichtFeld: **Settings → Plugins → Local Installations → Load**.

First load takes 60–120 seconds while `uv` installs `sam2`, `transformers`, and `torch`.

---

## Quick start

1. **Open a COLMAP dataset** — the plugin needs sequential frames and camera nodes. Drop the folder containing `images/` and `sparse/` into LichtFeld.
2. **Click the `SAM Segmentation` tab** in the right-side panel.
3. **Type a prompt** describing the object you want to keep — e.g. `"person"`, `"red car"`, `"bouquet of flowers"`.
4. **Click `Preview on Selected Camera`** (after selecting any camera node). On first run, this downloads ~1.2 GB of model weights. Open the resulting mask PNG (path is logged) to confirm Grounding DINO + SAM 2 found the right object.
5. **Click `Generate All Masks`**. The plugin writes `<dataset>/masks/<stem>.png` for every camera. Watch the log for `[N/total]` progress.
6. **Train with masks**: open the **Training** panel, set **Mask Mode** to `segment`, click **Start Training**. The resulting splat contains only the masked object; the background fades because its photometric loss is zeroed out.

---

## How it works under the hood

```
text prompt + frame 0 ──► Grounding DINO ──► bounding box
                                                  │
                                                  ▼
                                           SAM 2 (init_state)
                                                  │
              ┌───────────────────────────────────┴───────────────────────────────┐
              ▼                                                                   ▼
  add_new_points_or_box(frame=0, box)                                  propagate_in_video
                                                                                  │
                                                                                  ▼
                                                       (frame_idx, mask) for every frame
                                                                                  │
                                                                                  ▼
                                                                  PIL.Image.fromarray(mask)
                                                                                  │
                                                                                  ▼
                                                            <dataset>/masks/<stem>.png
```

Mask values: **255** = keep object (foreground), **0** = discard (background).

---

## Configuration

The Grounding DINO thresholds are conservative defaults that suit most datasets. If
you get `RuntimeError: GroundingDINO found no match for prompt: ...`:

- Re-phrase: prefer concrete nouns (`"woman in red shirt"`) over vague ones (`"thing"`).
- Add specificity: `"the person on the sofa"` beats `"person"` in cluttered scenes.
- Lower the box threshold from `0.35` to `0.25` in `operators/grounded_sam2_backend.py` (the `threshold=` kwarg passed to `post_process_grounded_object_detection`).

You can also swap to a stronger Grounding DINO variant for harder prompts by changing
`gdino_id` from `"IDEA-Research/grounding-dino-tiny"` to `"IDEA-Research/grounding-dino-base"`
(adds ~1 GB of weights).

---

## Limitations & known quirks

- **First frame is decisive.** Grounding DINO only runs on frame 0; if the object isn't visible or recognizable there, the run fails. SAM 2 then tracks whatever was bounded in frame 0, so dramatic occlusion mid-sequence may degrade masks for those frames.
- **One object only.** Current backend takes the highest-confidence Grounding DINO box. Multi-object segmentation would need iterating `obj_id`s — happy to accept a PR.
- **Windows + triton.** `sam2`/`torch` may import `triton` at module level on Windows, where it has no wheels. A permissive triton stub is installed in `grounded_sam2_backend.py` before the `sam2` import to satisfy attribute probes. This is harmless because the SAM 2 inference path doesn't actually execute triton kernels.
- **Mask quality scales with the SAM 2 model.** The default `sam2-hiera-small` is fast; swap to `sam2-hiera-large` for ~2× sharper masks at the cost of VRAM and time.
- **The first preview hangs for several minutes** while ~1.2 GB of weights download. Subsequent runs reuse the cached weights.
- **Models stay resident in VRAM until LichtFeld exits.** Once loaded, Grounding DINO and SAM 2 (~4–6 GB combined) are kept in GPU memory so subsequent **Generate All Masks** runs skip the 15–30 s model-load step. If you need that VRAM back without restarting LichtFeld, unload the plugin.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Marketplace shows `Load failed: uv sync failed: ... triton ... no wheels` | An older revision had `triton` as a declared dep. Pull latest from this repo; the current `pyproject.toml` no longer requires it. |
| `RuntimeError: GroundingDINO found no match` | Prompt is too vague or the object isn't in frame 0. Try a more specific noun, lower the threshold, or pick a different starting frame. |
| Tab visible but unclickable | The plugin needs `id` and `template` attributes on the `Panel` class. Both are set in `panels/sam_panel.py`; if you forked an older revision, check those exist. |
| `AttributeError: __delete__` printed after every operator | Cosmetic UI quirk in LichtFeld's RML panel adapter that fires whenever an operator completes. Doesn't affect the result. |

---

## Acknowledgments

- **SAM 2** by Meta AI — https://github.com/facebookresearch/sam2
- **Grounding DINO** by IDEA-Research — https://github.com/IDEA-Research/GroundingDINO (HuggingFace port: `IDEA-Research/grounding-dino-tiny`)
- **LichtFeld Studio** — https://lichtfeld.io

---

## License

GPL-3.0-or-later, matching LichtFeld Studio. See `LICENSE` if present.
