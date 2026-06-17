import os
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv("/opt/private/env.txt")

ssl_cert = os.environ.get("BM_SSL_CERT_FILE")
ssl_key = os.environ.get("BM_SSL_KEY_FILE")

# ── Constants & OpenAI client ────────────────────────────────────────────────
MODEL = "gpt-5.4"
MAX_WORKERS = 10

_api_key = os.environ.get("OPENAI_TEAM_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None


# ── Helpers ──────────────────────────────────────────────────────────────────

def call_openai_row(row_dict: dict, prompt: str) -> str:
    if client is None:
        return "[ERROR: No API key configured]"
    row_text = "\n".join(f"{k}: {v}" for k, v in row_dict.items())
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal AI assistant analyzing newsletter entries. "
                        "Provide concise, accurate answers."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nRow data:\n{row_text}",
                },
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def build_column_defs(df: pd.DataFrame) -> list:
    defs = []
    for col in df.columns:
        defs.append(
            {
                "field": col,
                "headerName": col,
                "resizable": True,
                "sortable": True,
                "filter": True,
                "wrapText": True,
                "autoHeight": True,
                "minWidth": 150,
                "tooltipField": col,
            }
        )
    return defs


def decode_upload(contents: str) -> bytes:
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def unique_col_name(df: pd.DataFrame, name: str) -> str:
    if name not in df.columns:
        return name
    i = 2
    while f"{name}_{i}" in df.columns:
        i += 1
    return f"{name}_{i}"


# ── App initialisation ───────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    url_base_pathname = '/zw_test_1/',
    external_stylesheets=[dbc.themes.FLATLY],
    title="Legal AI Newsletter Analyzer",
    suppress_callback_exceptions=True,
)
server = app.server


# ── Layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        # Stores
        dcc.Store(id="store-df", storage_type="memory"),
        dcc.Store(id="store-api-key-ok", data=bool(_api_key)),
        dcc.Download(id="download-excel"),

        # Navbar
        dbc.Navbar(
            dbc.Container(
                [
                    html.Div(
                        [
                            html.Span(
                                "Legal AI Newsletter Analyzer",
                                className="fs-4 fw-bold text-white",
                            ),
                            dbc.Badge(
                                "GPT-5.4 Powered",
                                color="success",
                                className="ms-3 align-middle",
                            ),
                        ],
                        className="d-flex align-items-center",
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4 px-3",
        ),

        # API key warning
        dbc.Alert(
            id="api-key-warning",
            color="danger",
            is_open=False,
            dismissable=False,
            className="mb-3",
        ),

        # Upload area
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                [
                    html.I(className="me-2"),
                    "Drag & Drop or ",
                    html.A("Select an Excel File", className="fw-bold"),
                    html.Div(
                        "Accepts .xlsx / .xls",
                        className="text-muted small mt-1",
                    ),
                ],
                className="text-center py-2",
            ),
            style={
                "width": "100%",
                "border": "2px dashed #adb5bd",
                "borderRadius": "8px",
                "padding": "20px",
                "textAlign": "center",
                "cursor": "pointer",
                "backgroundColor": "#f8f9fa",
            },
            accept=".xlsx,.xls",
            className="mb-2",
        ),
        dbc.Alert(id="upload-status-alert", is_open=False, className="mb-3"),

        # Controls card
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Tabs(
                        [
                            # ── Tab 1: Generate Column ───────────────────
                            dbc.Tab(
                                label="Generate Column",
                                tab_id="tab-generate",
                                children=[
                                    html.P(
                                        "Ask a question and GPT-5.4 will answer it for every row, "
                                        "adding the results as a new column. You can run this "
                                        "multiple times to add multiple columns.",
                                        className="text-muted small mt-3",
                                    ),
                                    dbc.Textarea(
                                        id="generate-prompt",
                                        placeholder="e.g. What is the key legal risk in this update?",
                                        rows=3,
                                        className="mb-2",
                                    ),
                                    dcc.Loading(
                                        id="loading-generate",
                                        type="circle",
                                        children=dbc.Button(
                                            "Generate Column",
                                            id="btn-generate",
                                            color="primary",
                                            className="me-2",
                                        ),
                                    ),
                                    html.Div(
                                        id="generate-status",
                                        className="text-muted small mt-2",
                                    ),
                                ],
                            ),

                            # ── Tab 2: Consolidated Output ────────────────
                            dbc.Tab(
                                label="Consolidated Output",
                                tab_id="tab-consolidated",
                                children=[
                                    html.P(
                                        "Select rows in the table below, then ask a question to "
                                        "get a single consolidated GPT-5.4 analysis across all "
                                        "selected rows.",
                                        className="text-muted small mt-3",
                                    ),
                                    dbc.Textarea(
                                        id="consolidated-prompt",
                                        placeholder="e.g. Summarise the common themes across these updates.",
                                        rows=3,
                                        className="mb-2",
                                    ),
                                    dcc.Loading(
                                        id="loading-consolidated",
                                        type="circle",
                                        children=dbc.Button(
                                            "Run Consolidated Analysis",
                                            id="btn-consolidated",
                                            color="success",
                                        ),
                                    ),
                                    html.Div(
                                        id="consolidated-status",
                                        className="text-muted small mt-2",
                                    ),
                                ],
                            ),
                        ],
                        id="control-tabs",
                        active_tab="tab-generate",
                    ),
                ]
            ),
            className="mb-3 shadow-sm",
        ),

        # Download button
        dbc.Button(
            [html.I(className="me-1"), "Download Excel"],
            id="btn-download",
            color="secondary",
            outline=True,
            className="mb-3",
        ),

        # Data table
        dag.AgGrid(
            id="data-table",
            columnDefs=[],
            rowData=[],
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 50,
                "animateRows": True,
                "tooltipShowDelay": 300,
                "rowSelection": {
                    "mode": "multiRow",
                    "checkboxes": True,
                    "headerCheckbox": True,
                },
            },
            defaultColDef={
                "resizable": True,
                "sortable": True,
                "filter": True,
                "wrapText": True,
                "autoHeight": True,
            },
            style={"height": "550px", "width": "100%"},
            className="ag-theme-alpine mb-4",
        ),

        # Consolidated result modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Consolidated Analysis Result")),
                dbc.ModalBody(
                    dcc.Markdown(
                        id="consolidated-result-text",
                        style={"whiteSpace": "pre-wrap"},
                    )
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="modal-close",
                        color="secondary",
                        className="ms-auto",
                    )
                ),
            ],
            id="consolidated-modal",
            size="lg",
            scrollable=True,
            is_open=False,
        ),
    ],
    fluid=True,
)


# ── Callbacks ────────────────────────────────────────────────────────────────

# CB2 — API key warning (runs on load)
@app.callback(
    Output("api-key-warning", "is_open"),
    Output("api-key-warning", "children"),
    Input("store-api-key-ok", "data"),
)
def show_api_warning(key_ok):
    if not key_ok:
        return True, (
            "⚠ OPENAI_API_KEY environment variable is not set. "
            "Generate features will return errors until it is configured."
        )
    return False, ""


# CB1 — Upload & parse
@app.callback(
    Output("store-df", "data"),
    Output("data-table", "columnDefs"),
    Output("data-table", "rowData"),
    Output("upload-status-alert", "children"),
    Output("upload-status-alert", "is_open"),
    Output("upload-status-alert", "color"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def parse_upload(contents, filename):
    if contents is None:
        return no_update, no_update, no_update, no_update, no_update, no_update
    try:
        decoded = decode_upload(contents)
        df = pd.read_excel(io.BytesIO(decoded))
        # Convert all columns to string for safe JSON serialisation
        df = df.astype(str).replace("nan", "")
        store = df.to_json(orient="records")
        col_defs = build_column_defs(df)
        row_data = df.to_dict("records")
        msg = f"✓ Loaded '{filename}' — {len(df):,} rows × {len(df.columns)} columns."
        return store, col_defs, row_data, msg, True, "success"
    except Exception as e:
        return (
            no_update,
            no_update,
            no_update,
            f"Error reading file: {e}",
            True,
            "danger",
        )


# CB3 — Generate column (concurrent API calls)
@app.callback(
    Output("store-df", "data", allow_duplicate=True),
    Output("data-table", "columnDefs", allow_duplicate=True),
    Output("data-table", "rowData", allow_duplicate=True),
    Output("generate-status", "children"),
    Input("btn-generate", "n_clicks"),
    State("generate-prompt", "value"),
    State("store-df", "data"),
    prevent_initial_call=True,
)
def generate_column(n_clicks, prompt, store_data):
    if store_data is None:
        return no_update, no_update, no_update, "⚠ Please upload a file first."
    if not prompt or not prompt.strip():
        return no_update, no_update, no_update, "⚠ Please enter a prompt."

    df = pd.read_json(io.StringIO(store_data), orient="records", dtype=str).fillna("")
    records = df.to_dict("records")
    n = len(records)

    results = [None] * n
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(call_openai_row, row, prompt): i
            for i, row in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"[ERROR: {e}]"

    col_name = unique_col_name(df, prompt[:60].strip().replace("\n", " "))
    df[col_name] = results

    store = df.to_json(orient="records")
    col_defs = build_column_defs(df)
    row_data = df.to_dict("records")
    status = f"✓ Column '{col_name}' added ({n} rows processed)."
    return store, col_defs, row_data, status


# CB4 — Consolidated analysis + modal close
@app.callback(
    Output("consolidated-modal", "is_open"),
    Output("consolidated-result-text", "children"),
    Output("consolidated-status", "children"),
    Input("btn-consolidated", "n_clicks"),
    Input("modal-close", "n_clicks"),
    State("consolidated-prompt", "value"),
    State("store-df", "data"),
    State("data-table", "selectedRows"),
    prevent_initial_call=True,
)
def handle_consolidated(btn, close, prompt, store_data, selected_rows):
    triggered = ctx.triggered_id
    if triggered == "modal-close":
        return False, no_update, no_update

    if store_data is None:
        return False, no_update, "⚠ Please upload a file first."
    if not selected_rows:
        return False, no_update, "⚠ Please select at least one row in the table."
    if not prompt or not prompt.strip():
        return False, no_update, "⚠ Please enter a prompt."
    if client is None:
        return False, no_update, "⚠ OPENAI_API_KEY is not set."

    context = "\n\n---\n\n".join(
        f"Row {i + 1}:\n" + "\n".join(f"{k}: {v}" for k, v in row.items())
        for i, row in enumerate(selected_rows)
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal AI assistant. Analyse the provided legal "
                        "newsletter rows and answer the user's question with a "
                        "well-structured, consolidated response."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nSelected rows:\n{context}",
                },
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        result = resp.choices[0].message.content.strip()
        return True, result, f"✓ Analysis completed across {len(selected_rows)} row(s)."
    except Exception as e:
        return False, no_update, f"[ERROR: {e}]"


# CB5 — Download Excel
@app.callback(
    Output("download-excel", "data"),
    Input("btn-download", "n_clicks"),
    State("store-df", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, store_data):
    if store_data is None:
        return no_update
    df = pd.read_json(io.StringIO(store_data), orient="records", dtype=str).fillna("")
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return dcc.send_bytes(buffer.read(), "newsletter_analyzed.xlsx")



@app.callback(
    Output("output", "children"),
    Input("text-input", "value"),
)
def update_output(value):
    return f"You typed: {value}"

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=50001, debug=False, ssl_context=(ssl_cert, ssl_key)) 
