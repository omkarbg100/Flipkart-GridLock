from __future__ import annotations

PAGE_TITLE = "GridLock Traffic Intelligence"
PAGE_DESCRIPTION = (
    "Upload-first traffic violation dashboard with image and video preview, rule controls, evidence review, "
    "benchmark evaluation, repeat offender tracking, hotspot maps, and analytics."
)

FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
"""

APP_CSS = """
:root {
  --bg-a: #060b14;
  --bg-b: #0a1220;
  --panel: rgba(15, 23, 42, 0.92);
  --panel-border: rgba(148, 163, 184, 0.18);
  --muted: #9fb0c7;
  --text: #e6eef8;
  --accent: #4d96ff;
  --accent-2: #2dd4bf;
  --warn: #f4b860;
  --danger: #f87171;
}

body, .q-page-container {
  background:
    radial-gradient(circle at top left, rgba(77, 150, 255, 0.12), transparent 24%),
    radial-gradient(circle at top right, rgba(45, 212, 191, 0.12), transparent 22%),
    linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 55%, #050911 100%);
  color: var(--text);
}

body {
  font-family: "Inter", "Segoe UI", sans-serif;
}

.gridlock-shell {
  max-width: 1760px;
  margin: 0 auto;
  padding: 20px 18px 36px;
}

.hero-card,
.glass-card,
.metric-card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 24px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
  backdrop-filter: blur(18px);
}

.hero-card {
  padding: 28px;
}

.hero-eyebrow,
.section-eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.24em;
  font-size: 0.72rem;
  color: #8fb3ff;
}

.hero-title {
  margin-top: 8px;
  font-family: "Space Grotesk", "Inter", sans-serif;
  font-size: clamp(2rem, 3.4vw, 4rem);
  line-height: 1.02;
  letter-spacing: -0.04em;
}

.hero-copy {
  margin-top: 10px;
  max-width: 980px;
  color: var(--muted);
  line-height: 1.6;
}

.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 18px;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.38rem 0.72rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(148, 163, 184, 0.08);
  color: #dbe7f8;
  font-size: 0.82rem;
}

.pill.good {
  border-color: rgba(45, 212, 191, 0.32);
  background: rgba(45, 212, 191, 0.12);
}

.pill.warn {
  border-color: rgba(244, 184, 96, 0.34);
  background: rgba(244, 184, 96, 0.12);
}

.pill.info {
  border-color: rgba(77, 150, 255, 0.34);
  background: rgba(77, 150, 255, 0.12);
}

.section-title {
  font-family: "Space Grotesk", "Inter", sans-serif;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 2px;
}

.section-copy {
  color: var(--muted);
  margin-bottom: 12px;
  line-height: 1.55;
}

.metric-card {
  padding: 18px 18px 16px;
  min-width: 220px;
  flex: 1 1 220px;
}

.metric-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: #8fb3ff;
}

.metric-value {
  font-family: "Space Grotesk", "Inter", sans-serif;
  font-size: 2.1rem;
  font-weight: 700;
  margin-top: 6px;
  line-height: 1;
}

.metric-note {
  color: var(--muted);
  margin-top: 8px;
  font-size: 0.88rem;
}

.metric-blue { border-top: 4px solid #4d96ff; }
.metric-teal { border-top: 4px solid #2dd4bf; }
.metric-green { border-top: 4px solid #4ade80; }
.metric-amber { border-top: 4px solid #f4b860; }
.metric-red { border-top: 4px solid #fb7185; }

.preview-stage {
  min-height: 560px;
  padding: 20px;
  transition: transform 220ms ease, opacity 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
  overflow: hidden;
}

.preview-stage:hover {
  transform: translateY(-1px);
}

.preview-dual-grid,
.control-grid {
  display: grid;
  gap: 1rem;
}

.preview-dual-grid {
  grid-template-columns: minmax(0, 1.02fr) minmax(0, 0.98fr);
  align-items: start;
}

.control-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.control-shell {
  padding: 20px;
}

.control-group {
  padding: 18px;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(10, 16, 29, 0.55);
}

.control-group-title {
  font-family: "Space Grotesk", "Inter", sans-serif;
  font-size: 1rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.control-group-copy {
  color: var(--muted);
  margin-bottom: 12px;
  line-height: 1.5;
  font-size: 0.84rem;
}

.preview-stage .section-copy {
  margin-bottom: 10px;
}

.video-shell video {
  width: 100%;
  border-radius: 18px;
  background: #000;
}

.evidence-frame,
.live-frame {
  width: 100%;
  display: block;
  aspect-ratio: 16 / 9;
  object-fit: contain;
  border-radius: 18px;
  background: #000;
  overflow: hidden;
  transition: opacity 220ms ease, transform 220ms ease, filter 220ms ease;
}

.evidence-frame:hover,
.live-frame:hover {
  filter: brightness(1.03);
}

.stage-placeholder {
  min-height: 360px;
  border-radius: 18px;
  border: 1px dashed rgba(148, 163, 184, 0.28);
  background:
    linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.55)),
    radial-gradient(circle at top, rgba(77, 150, 255, 0.08), transparent 45%);
  padding: 28px 24px;
  text-align: center;
}

.stage-footnote {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(148, 163, 184, 0.05);
}

.scan-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 10px;
}

.scan-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.3rem 0.68rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(148, 163, 184, 0.08);
  font-size: 0.76rem;
  color: #dbe7f8;
}

.table-shell .q-table__top,
.table-shell .q-table__bottom {
  background: transparent;
}

.table-shell .q-table__container {
  background: transparent;
}

.field-label {
  color: #dbe7f8;
  font-size: 0.9rem;
  font-weight: 600;
}

.field-hint {
  color: var(--muted);
  font-size: 0.86rem;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.32rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(148, 163, 184, 0.08);
  font-size: 0.77rem;
}

.status-chip.good {
  color: #4ade80;
}

.status-chip.warn {
  color: #f4b860;
}

.status-chip.bad {
  color: #f87171;
}

.map-shell .nicegui-leaflet {
  width: 100%;
  height: 540px;
  border: 0;
  border-radius: 18px;
  overflow: hidden;
}

.mappls-hotspot-map {
  width: 100%;
  height: 540px;
  border: 0;
  border-radius: 18px;
  overflow: hidden;
}

@media (max-width: 1280px) {
  .preview-dual-grid,
  .control-grid {
    grid-template-columns: 1fr;
  }
}
"""
