# ele

**ele** is a CLI-first pseudo-RAW master preprocessing engine.

It converts JPEG, PNG, or AI-generated images into 16-bit editable master files suitable for grading in Lightroom, Photoshop, and Capture One.

---

## What ele is

A computational preprocessing engine that:

- analyses input image degradation (noise, compression artefacts, clipping, dynamic range)
- applies faithful restoration (deblocking, mild denoising, highlight pre-softening)
- builds a heuristic scene map (sky, skin, foliage, architecture, hair, fabric)
- reconstructs a pseudo-RAW master: flatter contrast, expanded shadows, compressed highlights, reduced brittle saturation
- exports a 16-bit TIFF ready for external RAW-style editing workflows

## What ele is NOT

ele is not an editor. It does not apply looks, presets, grades, or artistic colour decisions. The output is intentionally flat and needs further grading in your DAW of choice.

---

## Supported Input Formats

- JPEG
- PNG
- TIFF

---

## Color Pipeline

| Stage             | Space                            | Encoding                       |
|-------------------|----------------------------------|--------------------------------|
| Internal (all stages) | Linear sRGB primaries (D65)  | Linear float32                 |
| TIFF export       | ProPhoto RGB / ROMM RGB (D50)    | ROMM gamma ~1.8 (ISO 22028-2)  |
| Browser preview   | sRGB (D65)                       | sRGB gamma (IEC 61966-2-1)     |

**Why the preview and TIFF look slightly different:**
- The browser preview is sRGB (narrower gamut, gamma 2.2-ish).
- The TIFF export converts from sRGB primaries to ProPhoto primaries (3×3 matrix) and applies ROMM gamma ~1.8 before saving.
- In a colour-managed editor (Lightroom, Photoshop with proper colour settings) the TIFF should look equivalent to the preview.

**If colours appear dramatically stronger (reds/oranges/cyans) in Photoshop:**
1. Verify that Photoshop's colour settings use a colour-managed workflow (`Edit → Color Settings → More Options → sRGB or ProPhoto working space`).
2. When opening the TIFF, ensure Photoshop uses the **embedded ProPhoto RGB profile** (not a conversion to a different space).
3. Use `--debug-srgb-export` to generate a comparison sRGB TIFF. If the sRGB TIFF looks correct and the ProPhoto TIFF does not, the issue is with how your editor handles the embedded ICC profile.

| Property          | Value                            |
|-------------------|----------------------------------|
| Internal working space | Linear sRGB primaries (D65) |
| Export primaries  | ProPhoto RGB (ROMM RGB, D50)     |
| Export encoding   | ROMM gamma (ISO 22028-2)         |
| Internal dtype    | float32                          |
| Export bit depth  | 16-bit                           |

### ICC Profile Embedding

ele **always** embeds a ProPhoto RGB ICC profile in TIFF tag 34675.

Profile source priority:
1. Bundled profile at `ele/export/profiles/ProPhotoRGB.icc`
2. System profile — macOS ColorSync (`ROMM RGB.icc`), Linux colord (`ProPhoto.icc`)
3. Programmatically generated minimal ICC v2 profile (fallback, logged as a warning)

The output TIFF will be correctly colour-managed in Lightroom, Photoshop, and Capture One without any manual profile assignment.

---

## DNG Export

Linear DNG output is planned for the next release. The pro mode currently outputs 16-bit TIFF. Requesting `.dng` output raises `NotImplementedError` with a descriptive message.

---

## Pipeline Stages

| # | Stage                      | Description                                             |
|---|----------------------------|---------------------------------------------------------|
| 1 | Degradation Analysis       | Score compression, clipping, sharpness, noise, DR      |
| 2 | Faithful Restoration       | Deblocking, denoise, highlight pre-softening            |
| 3 | Scene Reconstruction       | Heuristic sky/skin/foliage/architecture masks + adjust |
| 4 | Pseudo-RAW Reconstruction  | Shadow expansion, highlight shoulder, WB partial, desat |
| 5 | Super Resolution (optional)| Lanczos ×2/4/6 + local contrast recovery (print mode)  |
| 6 | Export                     | 16-bit TIFF, LZW, ProPhoto ICC if available            |

---

## Modes

| Mode      | Max Long Edge | SR | Output       |
|-----------|--------------|-----|--------------|
| `free`    | 4000 px      | No  | 16-bit TIFF  |
| `creator` | 8000 px      | No  | 16-bit TIFF  |
| `pro`     | 8000 px      | No  | 16-bit TIFF  |
| `print`   | 16000 px     | Yes | 16-bit TIFF  |

---

## Install

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

---

## CLI Usage

```bash
# Creator mode (default, recommended for most use cases)
ele input.jpg --mode creator --output out.tiff

# Pro mode
ele input.png --mode pro --output out.tiff

# Print mode with ×4 super resolution
ele input.jpg --mode print --scale 4 --output out_print.tiff

# Free mode (4000px max)
ele input.jpg --mode free --output out_free.tiff

# Print degradation report and metadata
ele input.jpg --mode creator --output out.tiff --report

# Version
ele --version
```

---

## Architecture

```
ele/
  config.py              — constants, limits, pipeline strings
  types.py               — Mode, DegradationReport, SceneMap, PipelineResult
  utils.py               — image I/O helpers, array math
  core/
    degradation_analysis.py    — Stage 1
    restoration.py             — Stage 2
    scene_reconstruction.py    — Stage 3
    pseudo_raw_reconstruction.py — Stage 4 (core transform)
    super_resolution.py        — Stage 5 (print only)
  export/
    color_management.py  — ICC profile lookup, metadata
    tiff_export.py        — 16-bit TIFF writer
    dng_export.py         — Linear DNG stub (NotImplementedError)
  pipeline/
    free_pipeline.py
    creator_pipeline.py
    pro_pipeline.py
    print_pipeline.py
  cli/main.py            — Typer CLI
  api/app.py             — FastAPI scaffold
tests/
  test_cli.py
  test_tiff_export.py
```

---

## Web App

ele includes a web interface that runs alongside the CLI.

### Start locally

```bash
source .venv/bin/activate
uvicorn ele.api.app:app --reload
```

Open: **http://127.0.0.1:8000**

### Quick Export flow

1. Upload a JPEG, PNG, or TIFF
2. Choose mode (Free / Creator / Pro / Print)
3. Click **Process Image** — ele runs the full pipeline
4. Download the 16-bit TIFF master immediately

### Preview & Target Export flow

1. Upload an image, select **Preview & Target Export**
2. ele processes the image and shows a side-by-side comparison
3. Choose export target:
   - **Lightroom** → 16-bit TIFF, ProPhoto RGB
   - **Photoshop** → 16-bit TIFF, ProPhoto RGB
   - **Capture One** → 16-bit TIFF, ProPhoto RGB
   - **Print TIFF** → super-resolution (×2 / ×4 / ×6), 16-bit TIFF
   - **Adobe DNG** → not available in this build (planned)
4. Download the exported master

### API routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Home / upload page |
| GET | `/health` | Health check (JSON) |
| POST | `/upload` | Upload and process image |
| GET | `/preview/{job_id}` | Side-by-side preview + target selector |
| POST | `/export/{job_id}` | Apply export target |
| GET | `/result/{job_id}` | Result page + download link |
| GET | `/download/{job_id}` | Download output TIFF |

### DNG limitation

Linear DNG export is planned for the next release.
The UI shows a clear "planned / not available" message when DNG is selected.
All current exports use 16-bit TIFF with ProPhoto RGB.

## Pricing and Access

ele has no free plan. All exports require one of the following:

| Plan | Access | Cost |
|------|--------|------|
| **Creator** | Pay-per-export credit packs | 1 credit (standard export) / 3 credits (print) |
| **Pro** | Monthly subscription | Unlimited exports, all modes |
| **Ambassador** | Ambassador key | Unlimited, free — issued directly by ele |

### Credit packs

- 10 credits — try it out
- 50 credits — standard use
- 200 credits — high volume

Credits do not expire.

### Magazine discount codes

If you received a discount code from a magazine or partner promotion, enter it
at Stripe checkout. Example: `MAG-ELE20` for 20% off.

---

## Web App — Running Locally

### 1. Copy `.env.example` to `.env` and fill in your Stripe keys

```bash
cp .env.example .env
# Edit .env — add STRIPE_SECRET_KEY, STRIPE_PRICE_* etc.
```

### 2. Start the server

```bash
source .venv/bin/activate
uvicorn ele.api.app:app --reload
```

Open: **http://127.0.0.1:8000**

### 3. Testing Stripe webhooks locally

Install the Stripe CLI, then in a second terminal:

```bash
stripe listen --forward-to localhost:8000/billing/webhook
```

Copy the printed webhook signing secret into `.env` as `STRIPE_WEBHOOK_SECRET`.

---

## Admin — Ambassador Keys and Discount Codes

After `pip install -e .`, the `ele-admin` CLI is available:

```bash
# Create an ambassador key
ele-admin create-ambassador-key --label "photographer_name"

# List all ambassador keys
ele-admin list-ambassador-keys

# Create a magazine discount code (20% off, max 500 uses)
ele-admin create-discount-code --code MAG-ELE20 --percent 20 --max-uses 500

# List all discount codes
ele-admin list-discount-codes
```

There is also a read-only admin page at `/admin/keys?token=YOUR_ADMIN_TOKEN`
(requires `ADMIN_TOKEN` to be set in `.env`).

---

## Environment Variables

All required environment variables:

| Variable | Description |
|----------|-------------|
| `STRIPE_SECRET_KEY` | Stripe secret key (starts with `sk_`) |
| `STRIPE_PUBLISHABLE_KEY` | Stripe publishable key (starts with `pk_`) |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret (`whsec_...`) |
| `STRIPE_PRICE_CREATOR_10` | Stripe Price ID for 10-credit pack |
| `STRIPE_PRICE_CREATOR_50` | Stripe Price ID for 50-credit pack |
| `STRIPE_PRICE_CREATOR_200` | Stripe Price ID for 200-credit pack |
| `STRIPE_PRICE_PRO_MONTHLY` | Stripe Price ID for Pro monthly subscription |
| `APP_BASE_URL` | Public URL (e.g. `https://ele.example.com`) |
| `SESSION_SECRET` | Long random string for session cookie signing |
| `DATABASE_URL` | SQLAlchemy DB URL (default: `sqlite:///./storage/ele.db`) |
| `ADMIN_TOKEN` | Token for `/admin/keys` page (leave blank to disable) |

---

## Running Tests

```bash
pytest
```
