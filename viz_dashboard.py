#!/usr/bin/env python3
"""Interactive dashboard for one-pip stop sensitivity simulations."""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from one_pip_stop_sensitivity import format_p_value, mcnemar_exact


DATA_DIR = Path(".")
RESULTS_PATTERN = "one_pip_results_*.csv"
SUMMARY_PATH = Path("one_pip_results_summary.json")
REFRESH_INTERVAL_MS = 60_000


def _load_results_files() -> Tuple[
    Dict[str, pd.DataFrame], Optional[pd.Timestamp], Optional[pd.Timestamp]
]:
    results: Dict[str, pd.DataFrame] = {}
    global_min: Optional[pd.Timestamp] = None
    global_max: Optional[pd.Timestamp] = None
    for path in DATA_DIR.glob(RESULTS_PATTERN):
        instrument = path.stem.replace("one_pip_results_", "", 1).upper()
        if not instrument:
            continue
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if "entry_time" not in df.columns:
            continue
        if df.empty:
            df = df.copy()
            df["instrument"] = instrument
        else:
            df = df.copy()
            df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
            df = df.dropna(subset=["entry_time"])
            df["instrument"] = instrument
            if not df.empty:
                inst_min = df["entry_time"].min()
                inst_max = df["entry_time"].max()
                global_min = inst_min if global_min is None else min(global_min, inst_min)
                global_max = inst_max if global_max is None else max(global_max, inst_max)
        results[instrument] = df
    return results, global_min, global_max


def _load_summary_blob() -> Dict[str, object]:
    if SUMMARY_PATH.exists():
        try:
            payload = json.loads(SUMMARY_PATH.read_text())
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {"instruments": {}, "overall": {}}


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    serialised = df.copy()
    if "entry_time" in serialised.columns:
        serialised["entry_time"] = serialised["entry_time"].astype(str)
    return serialised.to_dict(orient="records")


def _build_empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
        ],
    )
    return fig


def _filter_data(
    store: Dict[str, object],
    instruments: Iterable[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Dict[str, pd.DataFrame]:
    filtered: Dict[str, pd.DataFrame] = {}
    for instrument in instruments:
        instrument_store = store.get(instrument)
        if not instrument_store:
            continue
        df = pd.DataFrame(instrument_store.get("records", []))
        if df.empty:
            filtered[instrument] = df
            continue
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
            df = df.dropna(subset=["entry_time"])
        bool_columns = [col for col in ("won_swing", "won_plus1") if col in df.columns]
        for col in bool_columns:
            df[col] = df[col].astype(bool)
        if start is not None:
            df = df[df["entry_time"] >= start]
        if end is not None:
            df = df[df["entry_time"] < end]
        filtered[instrument] = df
    return filtered


def _build_winrate_figure(filtered: Dict[str, pd.DataFrame]) -> go.Figure:
    rows: List[Dict[str, object]] = []
    for instrument, df in filtered.items():
        swing_wr = float(df["won_swing"].mean()) if not df.empty else 0.0
        plus_wr = float(df["won_plus1"].mean()) if not df.empty else 0.0
        rows.extend(
            [
                {"instrument": instrument, "variant": "Swing", "win_rate": swing_wr},
                {"instrument": instrument, "variant": "Swing + 1 pip", "win_rate": plus_wr},
            ]
        )
    if not rows:
        return _build_empty_figure("Win Rate Comparison", "Нет данных для выбранных инструментов")
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="instrument",
        y="win_rate",
        color="variant",
        barmode="group",
        text_auto=".2%",
        title="Win Rate Comparison (Swing vs Swing + 1 pip)",
    )
    fig.update_yaxes(tickformat=".0%", title="Win rate")
    fig.update_layout(legend_title="Variant")
    return fig


def _build_distribution_figure(filtered: Dict[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Stop size (pips)", "Take profit size (pips)"),
        horizontal_spacing=0.12,
    )
    has_data = False
    for instrument, df in filtered.items():
        if df.empty:
            continue
        has_data = True
        fig.add_trace(
            go.Histogram(
                x=df["stop_pips_swing"],
                name=f"{instrument} swing stop",
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=df["stop_pips_plus1"],
                name=f"{instrument} swing+1 stop",
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=df["tp_pips"],
                name=f"{instrument} take profit",
                opacity=0.6,
            ),
            row=1,
            col=2,
        )
    if not has_data:
        return _build_empty_figure(
            "Distribution of stop/TP sizes", "Нет данных для построения распределений"
        )
    fig.update_layout(
        title="Distribution of stop and take-profit sizes",
        barmode="overlay",
        bargap=0.1,
    )
    fig.update_xaxes(title="Pips", row=1, col=1)
    fig.update_xaxes(title="Pips", row=1, col=2)
    fig.update_yaxes(title="Count", row=1, col=1)
    fig.update_yaxes(title="Count", row=1, col=2)
    return fig


def _build_timeline_figure(filtered: Dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    has_data = False
    for instrument, df in filtered.items():
        if df.empty:
            continue
        has_data = True
        ordered = df.sort_values("entry_time").reset_index(drop=True)
        ordered["trade_index"] = range(1, len(ordered) + 1)
        fig.add_trace(
            go.Scatter(
                x=ordered["entry_time"],
                y=ordered["won_swing"].astype(int),
                mode="lines+markers",
                name=f"{instrument} swing",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ordered["entry_time"],
                y=ordered["won_plus1"].astype(int),
                mode="lines+markers",
                name=f"{instrument} swing + 1 pip",
            )
        )
    if not has_data:
        return _build_empty_figure(
            "Trade outcome timeline", "Нет данных для построения таймлайна"
        )
    fig.update_layout(title="Trade outcome timeline", legend_title="Series")
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Loss", "Win"], title="Outcome")
    fig.update_xaxes(title="Entry time")
    return fig


def _build_mcnemar_table(
    filtered: Dict[str, pd.DataFrame],
    summaries: Dict[str, object],
) -> html.Div:
    rows: List[Dict[str, object]] = []
    combined_frames = [df for df in filtered.values() if not df.empty]
    for instrument, df in filtered.items():
        if df.empty:
            rows.append({"instrument": instrument, "b": 0, "c": 0, "p_value": "1.000000"})
            continue
        b = int(((~df["won_swing"]) & df["won_plus1"]).sum())
        c = int((df["won_swing"] & (~df["won_plus1"])).sum())
        p_value = format_p_value(mcnemar_exact(b, c))
        rows.append({"instrument": instrument, "b": b, "c": c, "p_value": p_value})
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        b_combined = int(((~combined["won_swing"]) & combined["won_plus1"]).sum())
        c_combined = int((combined["won_swing"] & (~combined["won_plus1"])).sum())
        combined_row = {
            "instrument": "Combined",
            "b": b_combined,
            "c": c_combined,
            "p_value": format_p_value(mcnemar_exact(b_combined, c_combined)),
        }
        rows.append(combined_row)
    if not rows:
        return html.Div("McNemar statistics unavailable for the current selection.")
    header = html.Thead(
        html.Tr(
            [
                html.Th("Instrument"),
                html.Th("b (swing loss, +1 win)"),
                html.Th("c (swing win, +1 loss)"),
                html.Th("Exact p-value"),
            ]
        )
    )
    body_rows: List[html.Tr] = []
    for row in rows:
        meta = summaries.get("instruments", {}).get(row["instrument"], {}) if summaries else {}
        last_run = meta.get("last_run")
        instrument_cell: List[object] = [html.Span(row["instrument"])]
        if last_run:
            instrument_cell.extend([html.Br(), html.Small(f"Last run: {last_run}")])
        body_rows.append(
            html.Tr(
                [
                    html.Td(instrument_cell),
                    html.Td(str(row["b"])),
                    html.Td(str(row["c"])),
                    html.Td(row["p_value"]),
                ]
            )
        )
    body = html.Tbody(body_rows)
    return html.Div(
        [
            html.H3("McNemar exact test"),
            html.Table([header, body], className="mcnemar-table"),
        ]
    )


app = Dash(__name__)
app.title = "One-pip Stop Sensitivity Dashboard"

app.layout = html.Div(
    [
        html.H1("One-pip Stop Sensitivity Dashboard"),
        dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dcc.Store(id="results-store"),
        dcc.Store(id="summary-store"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Instruments"),
                        dcc.Dropdown(id="instrument-dropdown", multi=True, placeholder="Select instruments"),
                    ],
                    className="control-block",
                ),
                html.Div(
                    [
                        html.Label("Date range"),
                        dcc.DatePickerRange(id="date-range-picker"),
                    ],
                    className="control-block",
                ),
            ],
            className="controls",
        ),
        dcc.Graph(id="winrate-graph"),
        dcc.Graph(id="distribution-graph"),
        dcc.Graph(id="timeline-graph"),
        html.Div(id="mcnemar-table"),
    ],
    className="container",
)


@app.callback(
    Output("results-store", "data"),
    Output("summary-store", "data"),
    Output("instrument-dropdown", "options"),
    Output("instrument-dropdown", "value"),
    Output("date-range-picker", "min_date_allowed"),
    Output("date-range-picker", "max_date_allowed"),
    Output("date-range-picker", "start_date"),
    Output("date-range-picker", "end_date"),
    Input("refresh-interval", "n_intervals"),
    State("instrument-dropdown", "value"),
    State("date-range-picker", "start_date"),
    State("date-range-picker", "end_date"),
)
def refresh_data(_, current_selection, start_date, end_date):
    results_map, global_min, global_max = _load_results_files()
    summary_blob = _load_summary_blob()

    options = [{"label": inst, "value": inst} for inst in sorted(results_map.keys())]
    selection = list(current_selection) if isinstance(current_selection, list) else []
    if selection:
        selection = [inst for inst in selection if inst in results_map]
    if not selection and options:
        selection = [options[0]["value"]]

    min_date_allowed = global_min.date().isoformat() if global_min is not None else None
    max_date_allowed = global_max.date().isoformat() if global_max is not None else None

    if global_min is not None and global_max is not None:
        start_dt = pd.to_datetime(start_date).date() if start_date else global_min.date()
        end_dt = pd.to_datetime(end_date).date() if end_date else global_max.date()
        if start_dt < global_min.date():
            start_dt = global_min.date()
        if end_dt > global_max.date():
            end_dt = global_max.date()
        if start_dt > end_dt:
            start_dt, end_dt = global_min.date(), global_max.date()
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
    else:
        start_iso = start_date
        end_iso = end_date

    results_payload = {
        inst: {"records": _dataframe_to_records(df), "columns": df.columns.tolist()}
        for inst, df in results_map.items()
    }
    return (
        results_payload,
        summary_blob,
        options,
        selection,
        min_date_allowed,
        max_date_allowed,
        start_iso,
        end_iso,
    )


@app.callback(
    Output("winrate-graph", "figure"),
    Output("distribution-graph", "figure"),
    Output("timeline-graph", "figure"),
    Output("mcnemar-table", "children"),
    Input("instrument-dropdown", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date"),
    Input("results-store", "data"),
    Input("summary-store", "data"),
)
def update_visuals(selected_instruments, start_date, end_date, results_store, summary_store):
    if not results_store:
        empty_fig = _build_empty_figure("No data", "Запустите симуляцию для отображения результатов")
        return empty_fig, empty_fig, empty_fig, html.Div("Нет данных для отображения")

    if not selected_instruments:
        empty_fig = _build_empty_figure("No instruments selected", "Выберите хотя бы один инструмент")
        return empty_fig, empty_fig, empty_fig, html.Div("Не выбран ни один инструмент")

    start_ts = pd.to_datetime(start_date, utc=True) if start_date else None
    end_ts = pd.to_datetime(end_date, utc=True) if end_date else None
    if end_ts is not None:
        end_ts = end_ts + pd.Timedelta(days=1)

    filtered = _filter_data(results_store, selected_instruments, start_ts, end_ts)

    winrate_fig = _build_winrate_figure(filtered)
    distribution_fig = _build_distribution_figure(filtered)
    timeline_fig = _build_timeline_figure(filtered)
    mcnemar_component = _build_mcnemar_table(filtered, summary_store or {})

    return winrate_fig, distribution_fig, timeline_fig, mcnemar_component


if __name__ == "__main__":
    try:
        app.run_server(debug=False)
    except Exception as exc:
        # Dash 3 deprecated run_server in favour of run(); fall back if available.
        if exc.__class__.__name__ == "ObsoleteAttributeException" and hasattr(app, "run"):
            app.run(debug=False)
        else:
            raise
