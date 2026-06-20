from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from nicegui import app, ui

from config.settings import DEFAULT_CAMERAS, VIOLATION_TYPES
from .helpers import (
    build_options_from_controls,
    camera_options,
    detect_media_kind,
    dataframe_to_rows,
    format_job_rows,
    get_analysis_scope,
    human_size,
    human_time,
    safe_float,
    safe_int,
    save_uploaded_media,
    seed_user_state,
    set_control_values_from_settings,
    today_prefix,
)
from .theme import APP_CSS, FONT_LINKS, PAGE_TITLE
from .widgets import (
    build_bar_fig,
    build_line_fig,
    build_pie_fig,
    render_data_table,
    render_hotspot_map,
    render_metric_card,
    render_slider,
    render_toggle,
)
from services.evaluation import load_benchmark_package, summarize_benchmark
from services.db_service import (
    add_camera_if_not_exists,
    clear_database,
    get_camera_wise_violations,
    get_cameras_with_density,
    get_counts_summary,
    get_recent_violations,
    get_repeat_offender_alerts,
    get_violations_by_type,
    get_violations_over_time,
    init_db,
    search_violations,
)
from services.runtime import get_job_manager, get_settings_store


settings_store = get_settings_store()
job_manager = get_job_manager()


@ui.page("/")
def dashboard() -> None:
    try:
        init_db()
    except Exception as exc:
        ui.label(f"Database initialization failed: {exc}").classes("text-red-300")
        return

    storage = app.storage.user
    settings = settings_store.get()
    seed_user_state(storage, settings)
    dependency_state = job_manager.processor.dependency_status()
    analysis_ready = bool(dependency_state.get("opencv")) and bool(dependency_state.get("detector_available"))

    ui.page_title(PAGE_TITLE)
    ui.add_head_html(FONT_LINKS)
    ui.add_css(APP_CSS)

    controls: dict[str, Any] = {}
    camera_form: dict[str, Any] = {}
    clear_confirm: dict[str, Any] = {}
    benchmark_state: dict[str, Any] = {
        "summary": None,
        "ground_truth": None,
        "predictions": None,
        "error": "",
        "name": "",
    }

    def refresh_live_sections(include_evidence: bool = False) -> None:
        summary_strip.refresh()
        overview_panel.refresh()
        hotspots_panel.refresh()
        alerts_panel.refresh()
        jobs_panel.refresh()
        live_processing_panel.refresh()
        if include_evidence:
            evidence_panel.refresh()

    async def handle_upload(event) -> None:
        try:
            destination = await save_uploaded_media(event.file)
            media_kind = detect_media_kind(event.file.name)
            storage["uploaded_media_path"] = str(destination)
            storage["uploaded_media_name"] = event.file.name
            storage["uploaded_media_kind"] = media_kind
            storage["uploaded_video_path"] = str(destination)
            storage["uploaded_video_name"] = event.file.name
            storage["uploaded_video_size"] = destination.stat().st_size
            storage["selected_violation_id"] = ""
            ui.notify(f"Uploaded {event.file.name}", type="positive")
            evidence_preview.refresh()
        except Exception as exc:
            ui.notify(f"Upload failed: {exc}", type="negative")

    def clear_upload() -> None:
        storage["uploaded_media_path"] = ""
        storage["uploaded_media_name"] = ""
        storage["uploaded_media_kind"] = ""
        storage["uploaded_video_path"] = ""
        storage["uploaded_video_name"] = ""
        storage["uploaded_video_size"] = 0
        storage["selected_violation_id"] = ""
        evidence_preview.refresh()
        ui.notify("Cleared current upload", type="info")

    def apply_filters() -> None:
        storage["filter_query"] = str(controls["filter_query"].value or "").strip()
        storage["filter_violation_type"] = str(controls["filter_violation_type"].value or "ALL")
        storage["filter_camera_id"] = str(controls["filter_camera_id"].value or "ALL")
        storage["filter_date_prefix"] = str(controls["filter_date_prefix"].value or "").strip() or ""
        storage["selected_violation_id"] = ""
        refresh_live_sections(include_evidence=True)
        scope_toolbar.refresh()
        ui.notify("Updated dashboard filters", type="positive")

    def reset_filters() -> None:
        controls["filter_query"].value = ""
        controls["filter_violation_type"].value = "ALL"
        controls["filter_camera_id"].value = "ALL"
        controls["filter_date_prefix"].value = today_prefix()
        apply_filters()

    def start_upload_analysis() -> None:
        if not analysis_ready:
            ui.notify("Install the optional CV stack first", type="warning")
            return
        media_path = storage.get("uploaded_media_path") or storage.get("uploaded_video_path", "")
        if not media_path:
            ui.notify("Upload an image or video first", type="warning")
            return
        try:
            media_name = storage.get("uploaded_media_name") or storage.get("uploaded_video_name") or Path(media_path).name
            options = build_options_from_controls(
                controls,
                settings,
                source_label=media_name,
            )
            if detect_media_kind(media_path) == "image":
                job = job_manager.submit_image(media_path, options)
            else:
                job = job_manager.submit_upload(media_path, options)
            storage["selected_job_id"] = job.id
            ui.notify(f"Started analysis job {job.id}", type="positive")
            jobs_panel.refresh()
            live_processing_panel.refresh()
            summary_strip.refresh()
        except Exception as exc:
            ui.notify(f"Could not start upload analysis: {exc}", type="negative")

    def start_webcam_analysis() -> None:
        if not analysis_ready:
            ui.notify("Install the optional CV stack first", type="warning")
            return
        try:
            options = build_options_from_controls(controls, settings, source_label="Webcam 0")
            job = job_manager.start_live_camera(options, camera_index=0)
            storage["selected_job_id"] = job.id
            ui.notify(f"Started webcam job {job.id}", type="positive")
            jobs_panel.refresh()
            live_processing_panel.refresh()
            summary_strip.refresh()
        except Exception as exc:
            ui.notify(f"Could not start webcam analysis: {exc}", type="negative")

    def cancel_active_job() -> None:
        active_job = job_manager.get_active_job()
        if not active_job:
            ui.notify("No active job to cancel", type="warning")
            return
        job_manager.cancel(active_job.id)
        ui.notify(f"Cancellation requested for {active_job.id}", type="warning")
        jobs_panel.refresh()

    def focus_latest_job() -> None:
        latest = job_manager.latest_snapshot()
        if not latest:
            ui.notify("No jobs available yet", type="warning")
            return
        storage["selected_job_id"] = latest["id"]
        jobs_panel.refresh()
        live_processing_panel.refresh()
        ui.notify(f"Focused job {latest['id']}", type="info")

    def save_defaults_from_controls() -> None:
        try:
            parking_zones_text = str(controls["parking_zones_json"].value or "").strip()
            parking_zones = json.loads(parking_zones_text or "[]")
            if not isinstance(parking_zones, list):
                raise ValueError("Parking zones must be a JSON list")
            settings_store.update(
                confidence_threshold=safe_float(controls["confidence_threshold"].value, settings.confidence_threshold),
                frame_skip=safe_int(controls["frame_skip"].value, settings.frame_skip),
                parking_violation_seconds=safe_float(controls["parking_violation_seconds"].value, settings.parking_violation_seconds),
                wrong_side_min_move=safe_int(controls["wrong_side_min_move"].value, settings.wrong_side_min_move),
                triple_overlap_ratio=safe_float(controls["triple_overlap_ratio"].value, settings.triple_overlap_ratio),
                helmet_skin_ratio=safe_float(controls["helmet_skin_ratio"].value, settings.helmet_skin_ratio),
                frame_width=safe_int(controls["frame_width"].value, settings.frame_width),
                frame_height=safe_int(controls["frame_height"].value, settings.frame_height),
                signal_state=str(controls["signal_state"].value or settings.signal_state),
                road_type=str(controls["road_type"].value or settings.road_type),
                allowed_direction=str(controls["allowed_direction"].value or settings.allowed_direction),
                left_allowed_dir=str(controls["left_allowed_dir"].value or settings.left_allowed_dir),
                right_allowed_dir=str(controls["right_allowed_dir"].value or settings.right_allowed_dir),
                stop_line_y=safe_int(controls["stop_line_y"].value, settings.stop_line_y),
                preprocess_profile=str(controls["preprocess_profile"].value or settings.preprocess_profile),
                parking_zones=parking_zones,
            )
            ui.notify("Saved current analysis controls as defaults", type="positive")
            settings_panel.refresh()
        except Exception as exc:
            ui.notify(f"Could not save defaults: {exc}", type="negative")

    def restore_defaults_to_controls() -> None:
        latest = settings_store.get()
        set_control_values_from_settings(controls, latest)
        ui.notify("Restored saved defaults into the analysis controls", type="info")

    def save_camera() -> None:
        camera_id = str(camera_form["camera_id"].value or "").strip()
        location = str(camera_form["location"].value or "").strip()
        latitude = safe_float(camera_form["latitude"].value, 30.300000)
        longitude = safe_float(camera_form["longitude"].value, 78.000000)
        if not camera_id or not location:
            ui.notify("Camera ID and location are required", type="warning")
            return
        try:
            add_camera_if_not_exists(camera_id, location, latitude, longitude)
            ui.notify(f"Saved camera {camera_id}", type="positive")
            scope_toolbar.refresh()
            settings_panel.refresh()
        except Exception as exc:
            ui.notify(f"Could not save camera: {exc}", type="negative")

    def clear_db() -> None:
        confirm_widget = clear_confirm.get("widget")
        if not confirm_widget or not confirm_widget.value:
            ui.notify("Confirm the checkbox first", type="warning")
            return
        try:
            clear_database()
            storage["selected_violation_id"] = ""
            storage["selected_job_id"] = ""
            scope_toolbar.refresh()
            refresh_live_sections(include_evidence=True)
            settings_panel.refresh()
            ui.notify("Database reset and default cameras restored", type="positive")
        except Exception as exc:
            ui.notify(f"Could not clear database: {exc}", type="negative")

    def download_filtered_csv() -> None:
        scope = get_analysis_scope(storage)
        df = search_violations(
            query=str(controls["filter_query"].value or "").strip(),
            violation_type=str(controls["filter_violation_type"].value or "ALL"),
            camera_id=str(controls["filter_camera_id"].value or "ALL"),
            date_prefix=scope,
        )
        if df.empty:
            ui.notify("No data to export", type="warning")
            return
        ui.download(df.to_csv(index=False).encode("utf-8"), filename="gridlock_filtered_violations.csv", media_type="text/csv")

    def download_alerts_csv() -> None:
        scope = get_analysis_scope(storage)
        df = get_repeat_offender_alerts(date_prefix=scope)
        if df.empty:
            ui.notify("No alerts to export", type="warning")
            return
        ui.download(df.to_csv(index=False).encode("utf-8"), filename="gridlock_repeat_offenders.csv", media_type="text/csv")

    async def handle_benchmark_upload(event) -> None:
        try:
            destination = await save_uploaded_media(event.file)
            storage["benchmark_package_path"] = str(destination)
            storage["benchmark_package_name"] = event.file.name
            ground_truth, predictions = load_benchmark_package(destination)
            benchmark_state["ground_truth"] = ground_truth
            benchmark_state["predictions"] = predictions
            benchmark_state["summary"] = summarize_benchmark(ground_truth, predictions)
            benchmark_state["error"] = ""
            benchmark_state["name"] = event.file.name
            ui.notify(f"Loaded benchmark package {event.file.name}", type="positive")
            evaluation_panel.refresh()
        except Exception as exc:
            benchmark_state["summary"] = None
            benchmark_state["ground_truth"] = None
            benchmark_state["predictions"] = None
            benchmark_state["error"] = str(exc)
            ui.notify(f"Could not load benchmark package: {exc}", type="negative")

    def clear_benchmark() -> None:
        storage["benchmark_package_path"] = ""
        storage["benchmark_package_name"] = ""
        benchmark_state["summary"] = None
        benchmark_state["ground_truth"] = None
        benchmark_state["predictions"] = None
        benchmark_state["error"] = ""
        benchmark_state["name"] = ""
        evaluation_panel.refresh()

    @ui.refreshable
    def summary_strip() -> None:
        scope = get_analysis_scope(storage)
        scope_label = storage.get("filter_date_prefix") or "All time"
        summary = get_counts_summary(date_prefix=scope)
        active_jobs = len([job for job in job_manager.list_jobs() if job["status"] in {"queued", "running"}])
        with ui.row().classes("w-full gap-4"):
            render_metric_card("Violations", summary["total"], f"Scope: {scope_label}", accent="metric-blue", icon="report")
            render_metric_card("Repeat alerts", summary["repeat"], "Repeat offenders prioritized", accent="metric-amber", icon="warning")
            render_metric_card("Cameras", summary["cameras"], "Registered locations", accent="metric-teal", icon="camera_alt")
            render_metric_card("Active jobs", active_jobs, "Uploads and webcams in flight", accent="metric-green", icon="task_alt")

    @ui.refreshable
    def evidence_preview() -> None:
        media_path = storage.get("uploaded_media_path") or storage.get("uploaded_video_path", "")
        media_name = storage.get("uploaded_media_name") or storage.get("uploaded_video_name", "")
        media_kind = storage.get("uploaded_media_kind") or detect_media_kind(media_name or media_path)
        with ui.card().classes("glass-card flex-1 min-w-0 preview-stage"):
            ui.label("1. Evidence preview").classes("section-title")
            ui.label("Uploaded image or source video stays here for quick review.").classes("section-copy")
            if media_path and Path(media_path).exists():
                path_obj = Path(media_path)
                if media_kind == "image":
                    ui.image(path_obj).classes("w-full evidence-frame")
                else:
                    ui.video(path_obj).classes("w-full evidence-frame")
                ui.label(
                    f"{media_name or path_obj.name} | {human_size(storage.get('uploaded_video_size', path_obj.stat().st_size))} | "
                    f"{path_obj.suffix.upper().lstrip('.') or media_kind.upper()}"
                ).classes("field-hint")
            else:
                with ui.column().classes("stage-placeholder items-center justify-center w-full gap-3"):
                    ui.icon("image_search", size="4rem").classes("text-slate-400")
                    ui.label("Drop traffic evidence above to start").classes("section-title")
                    ui.label("JPG, PNG, MP4, AVI, MOV, and MKV are supported for upload.").classes("field-hint")

    @ui.refreshable
    def live_processing_panel() -> None:
        active_job = job_manager.get_active_job()
        with ui.card().classes("glass-card flex-1 min-w-0 preview-stage"):
            ui.label("Live annotated scan").classes("section-title")
            ui.label("Exact processed frames with overlays, status, and progress from the current job.").classes("section-copy")

            if active_job:
                latest_snapshot = active_job.snapshot()
                status_label = f"{latest_snapshot['kind'].title()} | {latest_snapshot['status'].title()}"
                ui.label(status_label).classes("status-chip")
                ui.label(f"Source: {latest_snapshot.get('source_label', 'n/a')}").classes("field-hint")
                ui.label(f"Message: {latest_snapshot.get('message', '')}").classes("field-hint")
                ui.linear_progress(
                    value=float(latest_snapshot.get("progress", 0.0)),
                    show_value=True,
                    color="primary",
                ).classes("w-full")

                with ui.row().classes("w-full gap-2 flex-wrap"):
                    ui.label(f"Frames: {latest_snapshot.get('frames_processed', 0)}").classes("status-chip")
                    ui.label(f"Violations: {latest_snapshot.get('violations_logged', 0)}").classes("status-chip")
                    ui.label(f"Total: {latest_snapshot.get('total_frames', 0)}").classes("status-chip")
                    ui.label(f"Updated: {human_time(latest_snapshot.get('updated_at'))}").classes("status-chip")

                preview_b64 = latest_snapshot.get("preview_b64", "")
                if preview_b64:
                    ui.label("Annotated frame").classes("field-label")
                    ui.image(f"data:image/jpeg;base64,{preview_b64}").classes("w-full live-frame")
                    ui.label(
                        latest_snapshot.get("preview_caption") or "Annotated frame with stop line, zones, and OCR overlays."
                    ).classes("field-hint")
                else:
                    with ui.column().classes("stage-placeholder items-center justify-center w-full gap-3"):
                        ui.spinner(size="lg")
                        ui.label("Processing started").classes("section-title")
                        ui.label("The annotated frame will appear here as soon as the first scanned frame is ready.").classes("field-hint")
            else:
                with ui.column().classes("stage-placeholder items-center justify-center w-full gap-3"):
                    ui.icon("motion_photos_auto", size="4rem").classes("text-slate-400")
                    ui.label("Start analysis to see the live annotated frame here").classes("section-title")
                    ui.label("This pane shows detected vehicles, red-line markers, parking boxes, wrong-side lanes, and OCR results.").classes("field-hint")

            with ui.card().classes("glass-card stage-footnote w-full"):
                ui.label("What appears here").classes("field-label")
                ui.label(
                    "Live overlays stay visible while the scan runs, including vehicles, stop-line checks, parking zones, wrong-side motion, and plate OCR."
                ).classes("field-hint")
                with ui.row().classes("scan-chip-row"):
                    for text in ["Vehicles", "Stop line", "Parking zones", "Wrong-side", "OCR plates"]:
                        ui.label(text).classes("scan-chip")

    def preview_panel() -> None:
        with ui.card().classes("glass-card w-full video-shell"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("2. Preview and live scan").classes("section-title")
                    ui.label(
                        "The left pane shows the uploaded evidence, while the right pane shows the live annotated scan output."
                    ).classes("section-copy")
                with ui.row().classes("items-center gap-2"):
                    if media_path := (storage.get("uploaded_media_path") or storage.get("uploaded_video_path", "")):
                        if Path(media_path).exists():
                            ui.button(
                                "Download",
                                icon="download",
                                on_click=lambda: ui.download(Path(media_path), filename=Path(media_path).name),
                            ).props("flat")
                            ui.button("Clear", icon="close", on_click=clear_upload).props("flat")

            with ui.element("div").classes("w-full preview-dual-grid"):
                evidence_preview()
                live_processing_panel()

    @ui.refreshable
    def scope_toolbar() -> None:
        cameras = camera_options()
        storage.setdefault("filter_camera_id", "ALL")
        storage.setdefault("filter_violation_type", "ALL")
        storage.setdefault("filter_query", "")
        storage.setdefault("filter_date_prefix", today_prefix())
        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("3. Scope and search").classes("section-title")
                    ui.label("Filter the dashboard without using a sidebar.").classes("section-copy")
                ui.label(f"Current scope: {storage.get('filter_date_prefix') or 'All time'}").classes("status-chip")

            with ui.row().classes("w-full gap-3 items-end"):
                controls["filter_query"] = ui.input("Search plate or location", value=storage.get("filter_query", "")).classes("flex-1")
                controls["filter_violation_type"] = ui.select(
                    ["ALL"] + list(VIOLATION_TYPES.values()),
                    value=storage.get("filter_violation_type", "ALL"),
                    label="Violation type",
                ).classes("min-w-[220px]")
                controls["filter_camera_id"] = ui.select(
                    cameras,
                    value=storage.get("filter_camera_id", "ALL"),
                    label="Camera",
                ).classes("min-w-[180px]")
                controls["filter_date_prefix"] = ui.date_input(
                    "Date scope",
                    value=storage.get("filter_date_prefix", today_prefix()) or today_prefix(),
                ).classes("min-w-[210px]")

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Apply filters", icon="filter_alt", on_click=apply_filters).props("unelevated color=primary")
                ui.button("Reset", icon="restart_alt", on_click=reset_filters).props("flat")
                ui.button("Export CSV", icon="download", on_click=download_filtered_csv).props("flat")

    @ui.refreshable
    def overview_panel() -> None:
        scope = get_analysis_scope(storage)
        recent_df = get_recent_violations(limit=8, date_prefix=scope)
        type_df = get_violations_by_type(date_prefix=scope)
        trend_df = get_violations_over_time(date_prefix=scope)
        camera_df = get_camera_wise_violations(date_prefix=scope)
        with ui.column().classes("w-full gap-4"):
            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("glass-card flex-1 min-w-[300px]"):
                    ui.label("Violation mix").classes("section-title")
                    ui.label("Breakdown of the current scope by violation category.").classes("section-copy")
                    fig = build_pie_fig(type_df, "Violation mix", "Violation Type", "Count", ["#4d96ff", "#2dd4bf", "#f4b860", "#fb7185", "#8b5cf6"])
                    if fig is not None:
                        ui.plotly(fig).classes("w-full").style("height: 340px")
                    else:
                        ui.label("No violation mix data for this scope.").classes("field-hint")
                with ui.card().classes("glass-card flex-1 min-w-[300px]"):
                    ui.label("Violation trend").classes("section-title")
                    ui.label("How incidents move across the selected days.").classes("section-copy")
                    fig = build_line_fig(trend_df, "Violations over time", "Date", "Count")
                    if fig is not None:
                        ui.plotly(fig).classes("w-full").style("height: 340px")
                    else:
                        ui.label("No timeline data for this scope.").classes("field-hint")
            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("glass-card flex-1 min-w-[300px]"):
                    ui.label("Camera load").classes("section-title")
                    ui.label("Which locations are generating the most activity.").classes("section-copy")
                    fig = build_bar_fig(camera_df.sort_values("Count", ascending=True) if not camera_df.empty else camera_df, "Camera-wise violations", "Location", "Count", horizontal=True)
                    if fig is not None:
                        ui.plotly(fig).classes("w-full").style("height: 340px")
                    else:
                        ui.label("No camera data for this scope.").classes("field-hint")
                with ui.card().classes("glass-card flex-1 min-w-[300px]"):
                    ui.label("Recent violations").classes("section-title")
                    ui.label("The latest evidence entries for this scope.").classes("section-copy")
                    render_data_table(recent_df, pagination=8, row_key="Time")

    @ui.refreshable
    def evidence_panel() -> None:
        scope = get_analysis_scope(storage)
        df = search_violations(
            query=str(controls["filter_query"].value or "").strip(),
            violation_type=str(controls["filter_violation_type"].value or "ALL"),
            camera_id=str(controls["filter_camera_id"].value or "ALL"),
            date_prefix=scope,
            limit=500,
        )
        rows = dataframe_to_rows(df)
        selected_id = storage.get("selected_violation_id", "")
        if rows and (not selected_id or all(str(row.get("id")) != str(selected_id) for row in rows)):
            selected_id = rows[0].get("id", "")
            storage["selected_violation_id"] = selected_id

        selected_row = next((row for row in rows if str(row.get("id")) == str(selected_id)), rows[0] if rows else None)

        def on_select(event: Any) -> None:
            if event.selection:
                storage["selected_violation_id"] = event.selection[0]["id"]
                evidence_panel.refresh()

        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Evidence browser").classes("section-title")
                    ui.label("Click a violation row to open the exact evidence frame.").classes("section-copy")
                ui.button("Export CSV", icon="download", on_click=download_filtered_csv).props("flat")

            if df.empty:
                ui.label("No evidence found for the current filters.").classes("field-hint")
                return

            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-[1.4] min-w-[540px] gap-3"):
                    render_data_table(df, pagination=10, row_key="id", selection="single", on_select=on_select)
                with ui.column().classes("flex-[0.9] min-w-[340px] gap-3"):
                    ui.label("Selected violation").classes("section-title")
                    if selected_row:
                        ui.label(f"{selected_row.get('Violation Type', '')}").classes("status-chip")
                        ui.label(f"Plate: {selected_row.get('Plate Number', 'UNKNOWN')}").classes("field-hint")
                        ui.label(f"Camera: {selected_row.get('Camera ID', '')}").classes("field-hint")
                        ui.label(f"Location: {selected_row.get('Location', '')}").classes("field-hint")
                        ui.label(f"Time: {selected_row.get('Time', '')}").classes("field-hint")
                        ui.label(f"Confidence: {selected_row.get('Confidence', '')}").classes("field-hint")

                        evidence_value = str(selected_row.get("Evidence", "") or "").strip()
                        evidence_path = Path(evidence_value) if evidence_value else None
                        if evidence_path and evidence_path.exists():
                            try:
                                ui.image(str(evidence_path)).classes("w-full evidence-frame")
                                ui.button(
                                    "Download evidence",
                                    icon="download",
                                    on_click=lambda: ui.download(evidence_path, filename=evidence_path.name),
                                ).props("flat")
                            except Exception:
                                ui.label("Evidence image available, but the preview could not be rendered.").classes("field-hint")
                        else:
                            ui.label("No evidence image available for this row.").classes("field-hint")
                    else:
                        ui.label("Select a row to inspect evidence.").classes("field-hint")

    @ui.refreshable
    def hotspots_panel() -> None:
        scope = get_analysis_scope(storage)
        cameras_df = get_cameras_with_density(date_prefix=scope)
        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Hotspots and location intelligence").classes("section-title")
                    ui.label("Violation density across camera locations.").classes("section-copy")
                ui.button("Refresh map", icon="refresh", on_click=hotspots_panel.refresh).props("flat")

            if cameras_df.empty:
                ui.label("No camera coordinates found. Add locations in Settings.").classes("field-hint")
                return

            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("glass-card flex-[1.3] min-w-[540px] map-shell"):
                    render_hotspot_map(cameras_df, scope)
                with ui.card().classes("glass-card flex-[0.8] min-w-[360px]"):
                    ui.label("Camera density").classes("section-title")
                    ui.label("Count of violations per registered camera.").classes("section-copy")
                    render_data_table(cameras_df[["camera_id", "location", "count"]].sort_values("count", ascending=False), pagination=8, row_key="camera_id")

    @ui.refreshable
    def alerts_panel() -> None:
        scope = get_analysis_scope(storage)
        df = get_repeat_offender_alerts(date_prefix=scope)
        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Repeat offender alerts").classes("section-title")
                    ui.label("Prioritize vehicles that keep violating the rules.").classes("section-copy")
                ui.button("Export alerts", icon="download", on_click=download_alerts_csv).props("flat")

            if df.empty:
                ui.label("No repeat offenders in the current scope.").classes("field-hint")
                return

            top_cards = df.head(3).to_dict("records")
            with ui.row().classes("w-full gap-4"):
                for item in top_cards:
                    with ui.card().classes("metric-card metric-amber flex-1"):
                        ui.label("Top offender").classes("metric-label")
                        ui.label(str(item.get("Plate Number", "UNKNOWN"))).classes("metric-value")
                        ui.label(f"{item.get('Offense Count', 0)} offenses | {item.get('Latest Violation', '')}").classes("metric-note")
            render_data_table(df, pagination=8, row_key="Plate Number")

    @ui.refreshable
    def jobs_panel() -> None:
        jobs = job_manager.list_jobs()
        selected_id = storage.get("selected_job_id", "")
        if jobs and (not selected_id or all(job["id"] != selected_id for job in jobs)):
            selected_id = jobs[0]["id"]
            storage["selected_job_id"] = selected_id

        selected_job = next((job for job in jobs if job["id"] == selected_id), jobs[0] if jobs else None)

        def on_select(event: Any) -> None:
            if event.selection:
                storage["selected_job_id"] = event.selection[0]["id"]
                jobs_panel.refresh()

        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Background jobs").classes("section-title")
                    ui.label("Track image, video, and webcam analysis jobs from one place.").classes("section-copy")
                with ui.row().classes("items-center gap-2"):
                    ui.button("Refresh", icon="refresh", on_click=jobs_panel.refresh).props("flat")
                    ui.button("Focus latest", icon="visibility", on_click=focus_latest_job).props("flat")

            if not jobs:
                ui.label("No jobs have been created yet.").classes("field-hint")
                return

            job_rows = format_job_rows(jobs)
            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-[1.1] min-w-[520px] gap-3"):
                    render_data_table(pd.DataFrame(job_rows), pagination=8, row_key="id", selection="single", on_select=on_select)
                with ui.column().classes("flex-[0.9] min-w-[360px] gap-3"):
                    ui.label("Selected job").classes("section-title")
                    if selected_job:
                        ui.label(f"{selected_job['kind'].title()} | {selected_job['status'].title()}").classes("status-chip")
                        ui.label(f"Source: {selected_job['source_label']}").classes("field-hint")
                        ui.label(f"Message: {selected_job['message']}").classes("field-hint")
                        ui.linear_progress(
                            value=float(selected_job.get("progress", 0.0)),
                            show_value=True,
                            color="primary",
                        ).classes("w-full")
                        ui.label(f"Frames processed: {selected_job['frames_processed']}").classes("field-hint")
                        ui.label(f"Violations logged: {selected_job['violations_logged']}").classes("field-hint")
                        ui.label(f"Updated: {human_time(selected_job['updated_at'])}").classes("field-hint")

                        preview_b64 = selected_job.get("preview_b64", "")
                        if preview_b64:
                            ui.image(f"data:image/jpeg;base64,{preview_b64}").classes("w-full live-frame")
                            ui.label(selected_job.get("preview_caption") or "Latest annotated frame").classes("field-hint")
                    else:
                        ui.label("Select a job to inspect details.").classes("field-hint")

            buttons = ui.row().classes("w-full gap-2 mt-2")
            with buttons:
                cancel_btn = ui.button("Cancel active job", icon="cancel", on_click=cancel_active_job).props("flat")
                if not job_manager.get_active_job():
                    cancel_btn.disable()
                ui.button("Start webcam", icon="videocam", on_click=start_webcam_analysis).props("flat")
                ui.button("Queue analysis", icon="play_arrow", on_click=start_upload_analysis).props("flat")

    @ui.refreshable
    def settings_panel() -> None:
        settings = settings_store.get()
        dependency_state_local = job_manager.processor.dependency_status()
        saved_summary = pd.DataFrame(
            [
                {"Field": "Confidence threshold", "Value": settings.confidence_threshold},
                {"Field": "Frame skip", "Value": settings.frame_skip},
                {"Field": "Parking dwell seconds", "Value": settings.parking_violation_seconds},
                {"Field": "Wrong-side min move", "Value": settings.wrong_side_min_move},
                {"Field": "Triple overlap ratio", "Value": settings.triple_overlap_ratio},
                {"Field": "Helmet skin ratio", "Value": settings.helmet_skin_ratio},
                {"Field": "Frame width", "Value": settings.frame_width},
                {"Field": "Frame height", "Value": settings.frame_height},
                {"Field": "Signal state", "Value": settings.signal_state},
                {"Field": "Road type", "Value": settings.road_type},
                {"Field": "Allowed direction", "Value": settings.allowed_direction},
                {"Field": "Left lane direction", "Value": settings.left_allowed_dir},
                {"Field": "Right lane direction", "Value": settings.right_allowed_dir},
                {"Field": "Stop line Y", "Value": settings.stop_line_y},
                {"Field": "Preprocessing profile", "Value": settings.preprocess_profile},
            ]
        )

        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Saved defaults").classes("section-title")
                    ui.label("Persist the analysis controls you want to reuse in future sessions.").classes("section-copy")
                with ui.row().classes("items-center gap-2"):
                    ui.button("Restore into controls", icon="restore", on_click=restore_defaults_to_controls).props("flat")
                    ui.button("Save current controls", icon="save", on_click=save_defaults_from_controls).props("flat color=primary")

            render_data_table(saved_summary, pagination=20, row_key="Field", dense=False)

        with ui.card().classes("glass-card w-full mt-4"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Camera registry").classes("section-title")
                    ui.label("Register or update camera locations used by the heatmap and summaries.").classes("section-copy")
                ui.label("No sidebar. Everything is in the main workspace.").classes("status-chip")

            camera_form["camera_id"] = ui.input("Camera ID", value=DEFAULT_CAMERAS[0]["id"]).classes("w-full")
            camera_form["location"] = ui.input("Location", value=DEFAULT_CAMERAS[0]["location"]).classes("w-full")
            with ui.row().classes("w-full gap-4"):
                camera_form["latitude"] = ui.number("Latitude", value=DEFAULT_CAMERAS[0]["latitude"], format="%.6f").classes("flex-1")
                camera_form["longitude"] = ui.number("Longitude", value=DEFAULT_CAMERAS[0]["longitude"], format="%.6f").classes("flex-1")
            ui.button("Add or update camera", icon="add_location_alt", on_click=save_camera).props("unelevated color=primary")

            cameras_df = get_cameras_with_density()
            if cameras_df.empty:
                ui.label("No registered cameras yet.").classes("field-hint")
            else:
                render_data_table(
                    cameras_df[["camera_id", "location", "latitude", "longitude", "count"]].sort_values("camera_id"),
                    pagination=8,
                    row_key="camera_id",
                )

        with ui.card().classes("glass-card w-full mt-4"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Data reset").classes("section-title")
                    ui.label("Clear the database and restore the default camera list.").classes("section-copy")
                ui.label(f"OpenCV: {dependency_state_local['opencv']}  |  Detector: {dependency_state_local['detector_available']}  |  OCR: {dependency_state_local['ocr_available']}").classes("field-hint")
            clear_confirm["widget"] = ui.checkbox("I understand this will clear violations, alerts, and cameras.")
            ui.button("Clear database", icon="delete_forever", on_click=clear_db).props("unelevated color=negative")

    @ui.refreshable
    def evaluation_panel() -> None:
        benchmark_name = storage.get("benchmark_package_name") or benchmark_state.get("name") or "none"
        summary = benchmark_state.get("summary")

        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Benchmark evaluation").classes("section-title")
                    ui.label("Load a JSON benchmark package with `ground_truth` and `predictions` to score the model outputs.").classes("section-copy")
                ui.label(f"Loaded: {benchmark_name}").classes("status-chip")

            with ui.row().classes("w-full gap-3 items-end"):
                ui.upload(
                    label="Drop benchmark JSON here or click to browse",
                    multiple=False,
                    auto_upload=True,
                    on_upload=handle_benchmark_upload,
                ).classes("flex-[1.4]")
                ui.button("Clear benchmark", icon="close", on_click=clear_benchmark).props("flat")

            ui.label(
                "Expected format: {\"ground_truth\": [...], \"predictions\": [...]} with labels, and optional boxes/confidence for mAP."
            ).classes("field-hint")

            if benchmark_state.get("error"):
                ui.label(f"Benchmark error: {benchmark_state['error']}").classes("status-chip bad")

            if not summary:
                ui.label("No benchmark package loaded yet. Use a JSON file to show Accuracy, Precision, Recall, F1, and mAP50.").classes("field-hint")
                return

            with ui.row().classes("w-full gap-4"):
                render_metric_card("Samples", summary["classification"]["samples"], "Benchmark rows evaluated", accent="metric-blue", icon="dataset")
                render_metric_card("Accuracy", f"{summary['classification']['accuracy']:.3f}", "Label agreement", accent="metric-green", icon="check_circle")
                render_metric_card("Precision", f"{summary['classification']['precision']:.3f}", "False positive control", accent="metric-amber", icon="precision_manufacturing")
                render_metric_card("Recall", f"{summary['classification']['recall']:.3f}", "Missed violation control", accent="metric-teal", icon="visibility")
                render_metric_card("F1", f"{summary['classification']['f1']:.3f}", "Balanced score", accent="metric-red", icon="score")
                render_metric_card("mAP50", f"{summary['detection']['map50']:.3f}", "Detection quality", accent="metric-blue", icon="image_search")

            with ui.row().classes("w-full gap-4 mt-4"):
                with ui.card().classes("glass-card flex-1 min-w-[360px]"):
                    ui.label("Benchmark summary").classes("section-title")
                    ui.label(summary["detection"].get("note", "Detection benchmark computed successfully.")).classes("field-hint")
                    ui.label(f"Matched pairs: {summary['classification']['matched_pairs']}").classes("field-hint")
                    ui.label(f"False positives: {summary['classification']['false_positives']}").classes("field-hint")
                    ui.label(f"False negatives: {summary['classification']['false_negatives']}").classes("field-hint")
                    ui.label(f"Mean AP: {summary['detection']['mean_ap']:.3f}").classes("field-hint")

                with ui.card().classes("glass-card flex-[1.2] min-w-[420px]"):
                    ui.label("AP by class").classes("section-title")
                    ap_rows = summary["detection"].get("ap_by_class") or []
                    if ap_rows:
                        render_data_table(pd.DataFrame(ap_rows), pagination=8, row_key="class", dense=False)
                    else:
                        ui.label("Bounding boxes were not present in the benchmark package, so class AP is not available.").classes("field-hint")

    with ui.column().classes("gridlock-shell gap-4"):
        with ui.card().classes("hero-card w-full"):
            with ui.row().classes("w-full items-start justify-between gap-6"):
                with ui.column().classes("gap-1 flex-1"):
                    ui.label("AI Traffic Violation Intelligence Dashboard").classes("hero-eyebrow")
                    ui.label("Upload, preview, and tune the traffic rules").classes("hero-title")
                    ui.label(
                        "A modern single-tier NiceGUI dashboard for traffic-camera uploads, rule tuning, detection jobs, "
                        "evidence review, benchmark evaluation, hotspot maps, and repeat offender tracking."
                    ).classes("hero-copy")
                    with ui.row().classes("pill-row"):
                        ui.label("Upload-first workflow").classes("pill info")
                        ui.label("No sidebar").classes("pill good")
                        ui.label("Modern NiceGUI UI").classes("pill info")
                        ui.label("Optional CV stack").classes("pill warn")
                    with ui.row().classes("pill-row"):
                        ui.label("Photo evidence").classes("pill good")
                        ui.label("Video evidence").classes("pill info")
                        ui.label("Preprocessing").classes("pill warn")
                        ui.label("OCR + evaluation").classes("pill good")
                with ui.column().classes("items-end gap-2"):
                    ui.label("Status").classes("hero-eyebrow")
                    ui.label("NiceGUI").classes("status-chip")
                    ui.label("Single Python tier").classes("status-chip")
                    ui.label("Port 8501").classes("status-chip")
                    if analysis_ready:
                        ui.label("Analysis ready").classes("status-chip good")
                    else:
                        ui.label("CV stack partial").classes("status-chip warn")

        if not analysis_ready:
            with ui.card().classes("glass-card w-full"):
                ui.label("Analysis is partially disabled").classes("section-title")
                ui.label(
                    "Install the optional CV stack to enable image upload, video upload, and webcam analysis: "
                    "`pip install -r requirements-cv.txt`"
                ).classes("section-copy")
                ui.label(
                    f"OpenCV: {dependency_state['opencv']} | Detector: {dependency_state['detector_available']} | OCR: {dependency_state['ocr_available']}"
                ).classes("field-hint")

        summary_strip()

        with ui.card().classes("glass-card w-full"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("1. Upload evidence").classes("section-title")
                    ui.label(
                        "Drop a traffic photo or recording here. The file is cached locally in data/uploads so analysis can reuse it."
                    ).classes("section-copy")
                ui.label("Preview works with image and video evidence.").classes("status-chip")
            ui.upload(
                label="Drop a traffic image or recording here or click to browse",
                multiple=False,
                auto_upload=True,
                on_upload=handle_upload,
            ).classes("w-full")
            with ui.row().classes("w-full items-center justify-between gap-3"):
                ui.label(f"Current file: {storage.get('uploaded_media_name') or storage.get('uploaded_video_name') or 'none'}").classes("field-hint")
                if storage.get("uploaded_media_path") or storage.get("uploaded_video_path"):
                    ui.button("Clear upload", icon="close", on_click=clear_upload).props("flat")

        preview_panel()

        with ui.card().classes("glass-card w-full control-shell"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("2. Traffic rule controls").classes("section-title")
                    ui.label("All controls live directly under the player, including red light, traffic-way, and lane rules.").classes("section-copy")
                ui.label("Tuned for uploaded footage and webcam runs.").classes("status-chip")

            with ui.element("div").classes("w-full control-grid"):
                with ui.card().classes("control-group"):
                    ui.label("Camera details").classes("control-group-title")
                    ui.label("Stamp the evidence with the camera and location used for analysis.").classes("control-group-copy")
                    controls["camera_id"] = ui.input("Camera ID", value=storage["camera_id"]).classes("w-full")
                    controls["location_name"] = ui.input("Location name", value=storage["location_name"]).classes("w-full")
                    with ui.row().classes("w-full gap-4"):
                        controls["latitude"] = ui.number("Latitude", value=storage["latitude"], format="%.6f").classes("flex-1")
                        controls["longitude"] = ui.number("Longitude", value=storage["longitude"], format="%.6f").classes("flex-1")
                    with ui.row().classes("w-full gap-4"):
                        controls["selected_date"] = ui.date_input("Date", value=storage["selected_date"]).classes("flex-1")
                        controls["selected_time"] = ui.time_input("Start time", value=storage["selected_time"]).classes("flex-1")

                with ui.card().classes("control-group"):
                    ui.label("Signal and lane logic").classes("control-group-title")
                    ui.label("Set the stop line, road direction, and lane permissions for the processor.").classes("control-group-copy")
                    controls["signal_state"] = render_toggle(
                        "Signal state",
                        {"Red light": "RED", "Green light": "GREEN"},
                        storage["signal_state"],
                    )
                    controls["road_type"] = render_toggle(
                        "Traffic way",
                        {"One-way road": "One-Way Road", "Two-way split": "Two-Way Road (Split Left/Right)"},
                        storage["road_type"],
                    )
                    controls["allowed_direction"] = render_toggle(
                        "Allowed direction",
                        {"Down": "down", "Up": "up"},
                        storage["allowed_direction"],
                    )
                    with ui.row().classes("w-full gap-3"):
                        with ui.column().classes("flex-1 gap-1"):
                            controls["left_allowed_dir"] = render_toggle(
                                "Left lane direction",
                                {"Up": "up", "Down": "down"},
                                storage["left_allowed_dir"],
                            )
                        with ui.column().classes("flex-1 gap-1"):
                            controls["right_allowed_dir"] = render_toggle(
                                "Right lane direction",
                                {"Down": "down", "Up": "up"},
                                storage["right_allowed_dir"],
                            )
                    controls["enable_signal_check"] = ui.switch("Enable signal checks", value=storage["enable_signal_check"]).classes("w-full")
                    controls["enable_wrong_side"] = ui.switch("Enable wrong-side checks", value=storage["enable_wrong_side"]).classes("w-full")
                    controls["enable_parking_check"] = ui.switch("Enable parking checks", value=storage["enable_parking_check"]).classes("w-full")

                with ui.card().classes("control-group"):
                    ui.label("Detection tuning").classes("control-group-title")
                    ui.label("Tune preprocessing, thresholds, and the live frame size used by the analyzer.").classes("control-group-copy")
                    controls["preprocess_profile"] = ui.select(
                        {
                            "off": "Off",
                            "auto": "Auto enhance",
                            "low_light": "Low light boost",
                            "motion_blur": "Motion blur reduction",
                            "enhanced": "Contrast + sharpen",
                        },
                        value=storage.get("preprocess_profile", "auto"),
                        label="Preprocessing profile",
                    ).classes("w-full")
                    ui.label("Enhances noisy, dark, and blurred evidence before detection.").classes("field-hint")

                    with ui.column().classes("w-full gap-2"):
                        controls["frame_skip"] = render_slider("Frame skip", min_value=1, max_value=15, step=1, value=float(storage["frame_skip"]))
                        controls["confidence_threshold"] = render_slider(
                            "Detection confidence",
                            min_value=0.10,
                            max_value=0.90,
                            step=0.01,
                            value=float(storage["confidence_threshold"]),
                        )
                        controls["parking_violation_seconds"] = render_slider(
                            "Parking dwell seconds",
                            min_value=1.0,
                            max_value=15.0,
                            step=0.5,
                            value=float(storage["parking_violation_seconds"]),
                        )
                        controls["wrong_side_min_move"] = render_slider(
                            "Wrong-side min move",
                            min_value=5,
                            max_value=50,
                            step=1,
                            value=float(storage["wrong_side_min_move"]),
                        )
                        controls["triple_overlap_ratio"] = render_slider(
                            "Triple-riding overlap",
                            min_value=0.10,
                            max_value=0.80,
                            step=0.01,
                            value=float(storage["triple_overlap_ratio"]),
                        )
                        controls["helmet_skin_ratio"] = render_slider(
                            "Helmet skin ratio",
                            min_value=0.05,
                            max_value=0.50,
                            step=0.01,
                            value=float(storage["helmet_skin_ratio"]),
                        )
                        controls["stop_line_y"] = render_slider(
                            "Stop line Y (red line)",
                            min_value=50,
                            max_value=500,
                            step=5,
                            value=float(storage["stop_line_y"]),
                        )

                    with ui.row().classes("w-full gap-4"):
                        controls["frame_width"] = ui.number("Frame width", value=storage["frame_width"], min=480, max=1280, step=20).classes("flex-1")
                        controls["frame_height"] = ui.number("Frame height", value=storage["frame_height"], min=320, max=900, step=20).classes("flex-1")

            with ui.card().classes("control-group w-full mt-4"):
                ui.label("Parking zones JSON").classes("control-group-title")
                ui.label("Define the parking boxes used to detect illegal parking inside the frame.").classes("control-group-copy")
                controls["parking_zones_json"] = ui.textarea(
                    "Parking zones JSON",
                    value=storage["parking_zones_json"],
                    placeholder="[[36, 295, 169, 346], ...]",
                ).classes("w-full")

            with ui.row().classes("w-full items-center gap-2 mt-4 control-actions"):
                ui.button("Queue analysis", icon="play_arrow", on_click=start_upload_analysis).props("unelevated color=primary")
                ui.button("Start webcam analysis", icon="videocam", on_click=start_webcam_analysis).props("flat")
                ui.button("Cancel active job", icon="cancel", on_click=cancel_active_job).props("flat")
                ui.button("Focus latest job", icon="visibility", on_click=focus_latest_job).props("flat")

            if not analysis_ready:
                ui.label("Install the optional CV packages to enable analysis buttons.").classes("field-hint")

        scope_toolbar()

        with ui.tabs().classes("w-full") as tabs:
            ui.tab("overview", label="Overview", icon="dashboard")
            ui.tab("evidence", label="Evidence", icon="photo")
            ui.tab("evaluation", label="Evaluation", icon="analytics")
            ui.tab("hotspots", label="Hotspots", icon="public")
            ui.tab("alerts", label="Alerts", icon="warning")
            ui.tab("jobs", label="Jobs", icon="task_alt")
            ui.tab("settings", label="Settings", icon="tune")

        with ui.tab_panels(tabs, value="overview").classes("w-full"):
            with ui.tab_panel("overview"):
                overview_panel()
            with ui.tab_panel("evidence"):
                evidence_panel()
            with ui.tab_panel("evaluation"):
                evaluation_panel()
            with ui.tab_panel("hotspots"):
                hotspots_panel()
            with ui.tab_panel("alerts"):
                alerts_panel()
            with ui.tab_panel("jobs"):
                jobs_panel()
            with ui.tab_panel("settings"):
                settings_panel()

        ui.timer(1.2, live_processing_panel.refresh)

        live_panels = [overview_panel, hotspots_panel, alerts_panel, jobs_panel, summary_strip]
        ui.timer(4.0, lambda: [panel.refresh() for panel in live_panels])



