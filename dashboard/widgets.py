from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
from nicegui import ui

from .helpers import dataframe_to_rows, table_columns

VIBRANT_DARK_PIE_COLORS = [
    "#991b1b",
    "#ef4444",
    "#f97316",
    "#f59e0b",
    "#84cc16",
    "#22c55e",
    "#10b981",
    "#06b6d4",
    "#8b5cf6",
]
VIBRANT_DARK_CONTINUOUS = [
    "#7f1d1d",
    "#b91c1c",
    "#ef4444",
    "#f97316",
    "#facc15",
    "#84cc16",
    "#22c55e",
]
DARK_PANEL_BG = "rgba(5, 10, 18, 0.96)"
DARK_GRID = "rgba(148, 163, 184, 0.18)"
DARK_TEXT = "#f8fafc"


def _apply_dark_layout(fig: Any, title: str) -> Any:
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor=DARK_PANEL_BG,
        plot_bgcolor=DARK_PANEL_BG,
        font=dict(color=DARK_TEXT),
        colorway=VIBRANT_DARK_PIE_COLORS,
        legend=dict(font=dict(color=DARK_TEXT)),
    )
    fig.update_xaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, tickfont=dict(color=DARK_TEXT))
    fig.update_yaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, tickfont=dict(color=DARK_TEXT))
    return fig


def render_data_table(
    df,
    *,
    pagination: int = 8,
    row_key: str | None = None,
    selection: str | None = None,
    on_select: Any = None,
    dense: bool = True,
) -> None:
    if df.empty:
        ui.label("No records for the current scope.").classes("field-hint")
        return

    key = row_key or ("id" if "id" in df.columns else str(df.columns[0]))
    props = "flat bordered"
    if dense:
        props += " dense"
    table = ui.table(
        columns=table_columns(df),
        rows=dataframe_to_rows(df),
        row_key=key,
        selection=selection,
        on_select=on_select,
        pagination=pagination,
    ).classes("w-full table-shell")
    table.props(props)


def build_pie_fig(df: pd.DataFrame, title: str, label_col: str, value_col: str, color_sequence: list[str]) -> Any:
    if df.empty:
        return None
    fig = px.pie(df, names=label_col, values=value_col, hole=0.58, color_discrete_sequence=color_sequence)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(legend=dict(orientation="h", y=-0.1, font=dict(color=DARK_TEXT)))
    _apply_dark_layout(fig, title)
    return fig


def build_bar_fig(df: pd.DataFrame, title: str, x_col: str, y_col: str, *, horizontal: bool = False) -> Any:
    if df.empty:
        return None
    if horizontal:
        fig = px.bar(df, x=y_col, y=x_col, orientation="h", color=y_col, color_continuous_scale=VIBRANT_DARK_CONTINUOUS)
    else:
        fig = px.bar(df, x=x_col, y=y_col, color=y_col, color_continuous_scale=VIBRANT_DARK_CONTINUOUS)
    fig.update_layout(coloraxis_showscale=False)
    _apply_dark_layout(fig, title)
    return fig


def build_line_fig(df: pd.DataFrame, title: str, x_col: str, y_col: str) -> Any:
    if df.empty:
        return None
    fig = px.line(df, x=x_col, y=y_col, markers=True, line_shape="spline")
    fig.update_traces(
        line=dict(color="#22c55e", width=3.5),
        marker=dict(size=8, color="#ef4444", line=dict(color="#f8fafc", width=1)),
    )
    _apply_dark_layout(fig, title)
    return fig


def render_hotspot_map(cameras_df: pd.DataFrame, scope_prefix: str | None = None) -> None:
    if cameras_df.empty:
        ui.label("No camera coordinates available yet.").classes("field-hint")
        return

    avg_lat = float(cameras_df["latitude"].mean())
    avg_lon = float(cameras_df["longitude"].mean())

    ui.label(f"Scope: {scope_prefix or 'All time'}").classes("field-hint")

    ui.label("Map provider: Leaflet").classes("status-chip good")

    leaflet_map = ui.leaflet(center=(avg_lat, avg_lon), zoom=12).classes("w-full")
    leaflet_map.style("height: 540px")
    leaflet_map.tile_layer(
        url_template="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        options={
            "attribution": "&copy; OpenStreetMap &copy; CartoDB",
            "subdomains": "abcd",
            "maxZoom": 19,
        },
    )

    with leaflet_map:
        for _, row in cameras_df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            count = int(row["count"])
            location = str(row["location"])
            camera_id = str(row["camera_id"])
            if count > 10:
                color = "#fb7185"
            elif count > 3:
                color = "#f4b860"
            else:
                color = "#2dd4bf"

            leaflet_map.generic_layer(
                name="circleMarker",
                args=[
                    [lat, lon],
                    {
                        "radius": max(8, min(30, 8 + count * 2)),
                        "color": color,
                        "fillColor": color,
                        "fillOpacity": 0.55,
                        "weight": 2,
                    },
                ],
            )
            leaflet_map.marker(
                latlng=(lat, lon),
                options={
                    "title": f"{camera_id} | {location} | {count} violations",
                },
            )


def render_metric_card(label: str, value: Any, note: str, accent: str = "metric-blue", icon: str = "insights") -> None:
    with ui.card().classes(f"metric-card {accent}"):
        with ui.row().classes("items-start justify-between w-full"):
            ui.label(label).classes("metric-label")
            ui.icon(icon, size="1.2rem").classes("text-slate-300")
        ui.label(str(value)).classes("metric-value")
        ui.label(note).classes("metric-note")


def render_slider(label: str, *, min_value: float, max_value: float, step: float, value: float) -> ui.slider:
    def format_value(current: float) -> str:
        try:
            return f"{float(current):g}"
        except Exception:
            return str(current)

    with ui.column().classes("w-full gap-1"):
        with ui.row().classes("w-full items-center justify-between gap-3"):
            ui.label(label).classes("field-label")
            value_label = ui.label(f"Selected: {format_value(value)}").classes("field-hint")

        slider = ui.slider(
            min=min_value,
            max=max_value,
            step=step,
            value=value,
        ).classes("w-full")

        def sync_value(event) -> None:
            current = getattr(event, "value", slider.value)
            value_label.set_text(f"Selected: {format_value(current)}")

        slider.on("update:model-value", sync_value)
        return slider


def render_toggle(label: str, options: dict[str, Any], value: Any) -> ui.toggle:
    ui.label(label).classes("field-label")
    toggle_options: list | dict = options
    if isinstance(options, dict):
        toggle_options = {stored_value: display_label for display_label, stored_value in options.items()}
        if value not in toggle_options:
            legacy_lookup = {display_label: stored_value for display_label, stored_value in options.items()}
            value = legacy_lookup.get(value, next(iter(toggle_options), None))
    return ui.toggle(toggle_options, value=value).classes("w-full")
