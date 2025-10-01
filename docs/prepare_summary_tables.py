import glob
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy.stats import gaussian_kde
from table_tag_utils import wrap_tags

DOCS_DIR = Path(__file__).resolve().parent
STATIC_DATASET_DIR = DOCS_DIR / "source" / "_static" / "dataset_generated"


MODALITY_CANONICAL = {
    "visual": "Visual",
    "auditory": "Auditory",
    "tactile": "Tactile",
    "somatosensory": "Tactile",
    "multisensory": "Multisensory",
    "motor": "Motor",
    "rest": "Resting State",
    "resting state": "Resting State",
    "resting-state": "Resting State",
    "sleep": "Sleep",
    "other": "Other",
}

MODALITY_COLOR_MAP = {
    "Visual": "#2563eb",
    "Auditory": "#0ea5e9",
    "Tactile": "#10b981",
    "Multisensory": "#ec4899",
    "Motor": "#f59e0b",
    "Resting State": "#6366f1",
    "Sleep": "#7c3aed",
    "Other": "#14b8a6",
    "Unknown": "#94a3b8",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(99, 102, 241, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _primary_modality(value: object) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, float) and pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    for sep in ("/", "|", ";"):
        text = text.replace(sep, ",")
    tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not tokens:
        return "Unknown"
    raw = tokens[0].lower()
    canonical = MODALITY_CANONICAL.get(raw)
    if canonical:
        return canonical
    candidate = tokens[0].strip()
    title_candidate = candidate.title()
    if title_candidate in MODALITY_COLOR_MAP:
        return title_candidate
    return "Other"


def _to_numeric_median_list(val) -> float | None:
    """Return a numeric value from possible list-like strings.

    Examples
    --------
    - "64" -> 64
    - "6,129" -> median -> 67.5 -> 68
    - "128, 512" -> 320
    - 500.0 -> 500

    """
    if pd.isna(val):
        return None
    try:
        # already numeric
        return float(val)
    except Exception:
        pass
    s = str(val).strip().strip("[]")
    if not s:
        return None
    try:
        nums = [float(x) for x in s.split(",") if str(x).strip()]
        if not nums:
            return None
        return float(np.median(nums))
    except Exception:
        return None


def _safe_int(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def gen_datasets_bubble(
    df: pd.DataFrame,
    out_html: str = "_static/dataset/dataset_bubble.html",
    x_var: str = "records",  # one of: 'records', 'duration_h', 'size_gb', 'tasks'
):
    """Generate an interactive bubble chart for datasets.

    - x: total duration (hours)
    - y: number of subjects
    - size: on-disk size (GB)
    - color: dataset modality
    """
    d = df.copy()
    d = d[d["dataset"].str.lower() != "test"]

    # numeric columns
    d["duration_h"] = pd.to_numeric(d.get("duration_hours_total"), errors="coerce")
    d["subjects"] = pd.to_numeric(d.get("n_subjects"), errors="coerce")
    d["records"] = pd.to_numeric(d.get("n_records"), errors="coerce")
    d["tasks"] = pd.to_numeric(d.get("n_tasks"), errors="coerce")
    d["size_bytes"] = pd.to_numeric(d.get("size_bytes"), errors="coerce")

    # parse sampling and channels into representative numeric values
    d["sfreq"] = d["sampling_freqs"].map(_to_numeric_median_list)
    d["nchans"] = d["nchans_set"].map(_to_numeric_median_list)

    d["modality_label"] = d.get("modality of exp").apply(_primary_modality)

    # disk size in GB for sizing
    GB = 1024**3
    d["size_gb"] = d["size_bytes"] / GB

    # hover content
    def _fmt_size(bytes_):
        return human_readable_size(_safe_int(bytes_, 0))

    # choose x axis field and labels
    x_field = (
        x_var
        if x_var in {"records", "duration_h", "size_gb", "tasks", "subjects"}
        else "records"
    )

    axis_base_labels = {
        "records": "#Records",
        "duration_h": "Duration (hours)",
        "size_gb": "Size (GB)",
        "tasks": "#Tasks",
        "subjects": "#Subjects",
    }

    x_label = f"{axis_base_labels[x_field]} (log scale)"
    y_field = "subjects"
    if x_field == "subjects":
        y_field = "records"
    y_label = f"{axis_base_labels[y_field]} (log scale)"

    # hover text adapts to axis choices
    if x_field == "duration_h":
        x_hover = "Duration (x): %{x:.2f} h"
    elif x_field == "size_gb":
        x_hover = "Size (x): %{x:.2f} GB"
    elif x_field == "tasks":
        x_hover = "Tasks (x): %{x:,}"
    elif x_field == "subjects":
        x_hover = "Subjects (x): %{x:,}"
    else:
        x_hover = "Records (x): %{x:,}"

    if y_field == "subjects":
        y_hover = "Subjects (y): %{y:,}"
    else:
        y_hover = "Records (y): %{y:,}"

    hover = (
        "<b>%{customdata[0]}</b>"  # dataset id
        f"<br>{x_hover}"
        f"<br>{y_hover}"
        "<br>Subjects (total): %{customdata[1]:,}"
        "<br>Records (total): %{customdata[2]:,}"
        "<br>Tasks: %{customdata[3]:,}"
        "<br>Channels: %{customdata[4]}"
        "<br>Sampling: %{customdata[5]} Hz"
        "<br>Size: %{customdata[6]}"
        "<br>Modality: %{customdata[7]}"
        "<br><i>Click bubble to open dataset page</i>"
        "<extra></extra>"
    )

    required_columns = {x_field, y_field, "size_gb"}
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=list(required_columns))
    d = d[(d[x_field] > 0) & (d[y_field] > 0)]

    d["dataset_url"] = d["dataset"].apply(get_dataset_url)

    if d.empty:
        out_path = Path(out_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        no_data_html = """
<div class="dataset-loading" id="dataset-loading">No dataset records available for plotting.</div>
"""
        with open(str(out_path), "w", encoding="utf-8") as f:
            f.write(no_data_html)
        return str(out_path)

    # Marker sizing: scale into a good visual range
    size_max = d["size_gb"].max()
    if not np.isfinite(size_max) or size_max <= 0:
        size_max = 1.0
    sizeref = (2.0 * size_max) / (40.0**2)  # target ~40px max marker

    # Prepare prettified strings for hover
    def _fmt_int(v):
        if v is None or pd.isna(v):
            return ""
        try:
            return str(int(round(float(v))))
        except Exception:
            return str(v)

    sfreq_str = d["sfreq"].map(_fmt_int)
    nchans_str = d["nchans"].map(_fmt_int)

    fig = px.scatter(
        d,
        x=x_field,
        y=y_field,
        size="size_gb",
        color="modality_label",
        hover_name="dataset",
        custom_data=[
            d["dataset"],
            d["subjects"],
            d["records"],
            d["tasks"],
            nchans_str,
            sfreq_str,
            d["size_bytes"].map(_fmt_size),
            d["modality_label"],
            d["dataset_url"],
        ],
        size_max=40,
        labels={
            y_field: y_label,
            "modality_label": "Modality",
            x_field: x_label,
        },
        color_discrete_map=MODALITY_COLOR_MAP,
        title="",
        category_orders={
            "modality_label": [
                label
                for label in MODALITY_COLOR_MAP.keys()
                if label in d["modality_label"].unique()
            ]
        },
        log_x=True,
        log_y=True,
    )

    # Add a log-log regression fit line and R² annotation when data permits
    fit_annotation_text = None
    numeric_x = pd.to_numeric(d[x_field], errors="coerce")
    numeric_y = pd.to_numeric(d[y_field], errors="coerce")
    mask = (
        np.isfinite(numeric_x)
        & np.isfinite(numeric_y)
        & (numeric_x > 0)
        & (numeric_y > 0)
    )

    if mask.sum() >= 2:
        log_x = np.log10(numeric_x[mask])
        log_y = np.log10(numeric_y[mask])
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)
        if np.ptp(log_x) > 0 and np.ptp(log_y) > 0 and ss_tot > 0:
            slope, intercept = np.polyfit(log_x, log_y, 1)
            line_log_x = np.linspace(log_x.min(), log_x.max(), 200)
            line_x = 10**line_log_x
            line_y = 10 ** (slope * line_log_x + intercept)
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="log-log fit",
                    line=dict(color="#111827", width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            residuals = log_y - (slope * log_x + intercept)
            r_squared = 1 - np.sum(residuals**2) / ss_tot
            fit_annotation_text = f"log-log OLS fit R² = {r_squared:.3f}"

    # tune marker sizing explicitly for better control
    for tr in fig.data:
        mode = getattr(tr, "mode", "") or ""
        if "markers" not in mode:
            continue
        tr.marker.update(
            sizemin=6,
            sizemode="area",
            sizeref=sizeref,
            line=dict(width=0.6, color="rgba(0,0,0,0.3)"),
            opacity=0.75,
        )
        tr.hovertemplate = hover

    plot_width = 1280
    plot_height = 720

    fig.update_layout(
        height=plot_height,
        width=plot_width,  # Landscape orientation
        margin=dict(l=60, r=40, t=80, b=60),
        template="plotly_white",
        legend=dict(
            title="Modality",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.99,
        ),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
        ),
        title=dict(
            text="",
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad=dict(t=10, b=8),
        ),
        autosize=True,  # Enable auto-sizing to fill container
    )

    if fit_annotation_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=fit_annotation_text,
            showarrow=False,
            font=dict(size=15, color="#111827"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(17,24,39,0.25)",
            borderwidth=1,
            borderpad=6,
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
        type="log",
        dtick=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
        type="log",
        dtick=1,
    )

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Add CSS and loading indicator for immediate proper sizing
    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="dataset-bubble",
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "dataset_landscape",
                "height": plot_height,
                "width": plot_width,
                "scale": 2,
            },
        },
    )

    # Wrap with styling to ensure proper initial sizing
    styled_html = f"""
<style>
#dataset-bubble {{
    width: 100% !important;
    max-width: {plot_width}px;
    height: {plot_height}px !important;
    min-height: {plot_height}px;
    margin: 0 auto;
}}
#dataset-bubble .plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
}}
.dataset-loading {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: {plot_height}px;
    font-family: Inter, system-ui, sans-serif;
    color: #6b7280;
}}
</style>
<div class="dataset-loading" id="dataset-loading">Loading dataset landscape...</div>
{html_content}
<script>
// Hide loading indicator once plot is rendered and make bubbles clickable
document.addEventListener('DOMContentLoaded', function() {{
    const loading = document.getElementById('dataset-loading');
    const plot = document.getElementById('dataset-bubble');

    function showPlot() {{
        if (loading) {{
            loading.style.display = 'none';
        }}
        if (plot) {{
            plot.style.display = 'block';
        }}
    }}

    function hookPlotlyClick(attempts) {{
        if (!plot || typeof plot.on !== 'function') {{
            if (attempts < 40) {{
                window.setTimeout(function() {{ hookPlotlyClick(attempts + 1); }}, 60);
            }}
            return;
        }}
        plot.on('plotly_click', function(evt) {{
            const point = evt && evt.points && evt.points[0];
            const url = point && point.customdata && point.customdata[8];
            if (url) {{
                window.open(url, '_blank', 'noopener');
            }}
        }});
        showPlot();
    }}

    hookPlotlyClick(0);
    showPlot();
}});
</script>
"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(styled_html)
    return str(out_path)


def human_readable_size(num_bytes: int) -> str:
    """Format bytes using the closest unit among MB, GB, TB (fallback to KB/B).

    Chooses the largest unit such that the value is >= 1. Uses base 1024.
    """
    if num_bytes is None:
        return "0 B"
    size = float(num_bytes)
    units = [
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024**1, "KB"),
        (1, "B"),
    ]
    for factor, unit in units:
        if size >= factor:
            value = size / factor
            # Use no decimals for B/KB; two decimals otherwise
            if unit in ("B", "KB"):
                return f"{int(round(value))} {unit}"
            return f"{value:.2f} {unit}"
    return "0 B"


def get_dataset_url(name: str) -> str:
    """Generate dataset URL for plots (relative to dataset summary page)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    return f"../../api/dataset/eegdash.dataset.{text.upper()}.html"


def wrap_dataset_name(name: str):
    # Remove any surrounding whitespace
    name = name.strip()
    # Link to the individual dataset API page
    # Updated structure: api/dataset/eegdash.dataset.<CLASS>.html
    url = get_dataset_url(name)
    if not url:
        return name.upper()
    return f'<a href="{url}">{name.upper()}</a>'


DATASET_CANONICAL_MAP = {
    "pathology": {
        "healthy controls": "Healthy",
        "healthy": "Healthy",
        "control": "Healthy",
        "clinical": "Clinical",
        "patient": "Clinical",
    },
    "modality": {
        "auditory": "Auditory",
        "visual": "Visual",
        "somatosensory": "Somatosensory",
        "multisensory": "Multisensory",
    },
    "type": {
        "perception": "Perception",
        "decision making": "Decision-making",
        "decision-making": "Decision-making",
        "rest": "Rest",
        "resting state": "Resting-state",
        "sleep": "Sleep",
    },
}


def _tag_normalizer(kind: str):
    canonical = {k.lower(): v for k, v in DATASET_CANONICAL_MAP.get(kind, {}).items()}

    def _normalise(token: str) -> str:
        text = " ".join(token.replace("_", " ").split())
        lowered = text.lower()
        if lowered in canonical:
            return canonical[lowered]
        return text

    return _normalise


def prepare_table(df: pd.DataFrame):
    # drop test dataset and create a copy to avoid SettingWithCopyWarning
    df = df[df["dataset"] != "test"].copy()

    df["dataset"] = df["dataset"].apply(wrap_dataset_name)
    # changing the column order
    df = df[
        [
            "dataset",
            "n_records",
            "n_subjects",
            "n_tasks",
            "nchans_set",
            "sampling_freqs",
            "size",
            "size_bytes",
            "Type Subject",
            "modality of exp",
            "type of exp",
        ]
    ]

    # renaming time for something small
    df = df.rename(
        columns={
            "modality of exp": "modality",
            "type of exp": "type",
            "Type Subject": "pathology",
        }
    )
    # number of subject are always int
    df["n_subjects"] = df["n_subjects"].astype(int)
    # number of tasks are always int
    df["n_tasks"] = df["n_tasks"].astype(int)
    # number of records are always int
    df["n_records"] = df["n_records"].astype(int)

    # from the sample frequency list, I will apply str
    df["sampling_freqs"] = df["sampling_freqs"].apply(parse_freqs)
    # from the channels set, I will follow the same logic of freq
    df["nchans_set"] = df["nchans_set"].apply(parse_freqs)
    # Wrap categorical columns with styled tags for downstream rendering
    pathology_normalizer = _tag_normalizer("pathology")
    modality_normalizer = _tag_normalizer("modality")
    type_normalizer = _tag_normalizer("type")

    df["pathology"] = df["pathology"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-pathology",
            normalizer=pathology_normalizer,
        )
    )
    df["modality"] = df["modality"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-modality",
            normalizer=modality_normalizer,
        )
    )
    df["type"] = df["type"].apply(
        lambda value: wrap_tags(
            value,
            kind="dataset-type",
            normalizer=type_normalizer,
        )
    )

    # Creating the total line
    df.loc["Total"] = df.sum(numeric_only=True)
    df.loc["Total", "dataset"] = f"Total {len(df) - 1} datasets"
    df.loc["Total", "nchans_set"] = ""
    df.loc["Total", "sampling_freqs"] = ""
    df.loc["Total", "pathology"] = ""
    df.loc["Total", "modality"] = ""
    df.loc["Total", "type"] = ""
    df.loc["Total", "size"] = human_readable_size(df.loc["Total", "size_bytes"])
    df = df.drop(columns=["size_bytes"])
    # arrounding the hours

    df.index = df.index.astype(str)

    return df


def main(source_dir: str, target_dir: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(Path(source_dir) / "dataset" / "*.csv"))
    for f in files:
        target_file = target_dir / Path(f).name
        print(f"Processing {f} -> {target_file}")
        df_raw = pd.read_csv(
            f, index_col=False, header=0, skipinitialspace=True
        )  # , sep=";")
        # Generate bubble chart from the raw data to have access to size_bytes
        # Use x-axis as number of subjects so participant counts lead the story
        bubble_path = target_dir / "dataset_bubble.html"
        gen_datasets_bubble(df_raw, str(bubble_path), x_var="subjects")
        copyfile(bubble_path, STATIC_DATASET_DIR / bubble_path.name)

        df = prepare_table(df_raw)
        # preserve int values
        df["n_subjects"] = df["n_subjects"].astype(int)
        df["n_tasks"] = df["n_tasks"].astype(int)
        df["n_records"] = df["n_records"].astype(int)
        int_cols = ["n_subjects", "n_tasks", "n_records"]

        # Coerce to numeric, allow NAs, and keep integer display
        df[int_cols] = (
            df[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
        )
        df = df.rename(
            columns={
                "dataset": "Dataset",
                "nchans_set": "# of channels",
                "sampling_freqs": "sampling (Hz)",
                "size": "size",
                "n_records": "# of records",
                "n_subjects": "# of subjects",
                "n_tasks": "# of tasks",
                "pathology": "Pathology",
                "modality": "Modality",
                "type": "Type",
            }
        )
        df = df[
            [
                "Dataset",
                "Pathology",
                "Modality",
                "Type",
                "# of records",
                "# of subjects",
                "# of tasks",
                "# of channels",
                "sampling (Hz)",
                "size",
            ]
        ]
        # (If you add a 'Total' row after this, cast again or build it as Int64.)
        html_table = df.to_html(
            classes=["sd-table", "sortable"],
            index=False,
            escape=False,
            table_id="datasets-table",
        )
        table_path = target_dir / "dataset_summary_table.html"
        with open(table_path, "+w", encoding="utf-8") as f:
            f.write(html_table)
        copyfile(table_path, STATIC_DATASET_DIR / table_path.name)

        # Generate KDE ridgeline plot for modality participant distributions
        try:
            d_modal = df_raw[df_raw["dataset"].str.lower() != "test"].copy()
            d_modal["modality_label"] = d_modal["modality of exp"].apply(
                _primary_modality
            )
            d_modal["n_subjects"] = pd.to_numeric(
                d_modal["n_subjects"], errors="coerce"
            )
            d_modal = d_modal.dropna(subset=["n_subjects"])

            # Filter out "Other" modality
            d_modal = d_modal[d_modal["modality_label"] != "Other"]

            # Calculate median participants per modality and reorder ascending
            median_participants = (
                d_modal.groupby("modality_label")["n_subjects"].median().sort_values()
            )
            order = [
                label
                for label in median_participants.index
                if label in d_modal["modality_label"].unique()
            ]

            fig_kde = go.Figure()
            rng = np.random.default_rng(42)

            for idx, label in enumerate(order):
                subset = d_modal[d_modal["modality_label"] == label].copy()
                vals = subset["n_subjects"].astype(float).dropna()
                if len(vals) < 3:
                    continue
                # Generate URLs for datasets in this modality
                subset["dataset_url"] = subset["dataset"].apply(get_dataset_url)
                log_vals = np.log10(vals)
                grid = np.linspace(log_vals.min() - 0.25, log_vals.max() + 0.25, 240)
                kde = gaussian_kde(log_vals)
                density = kde(grid)
                if density.max() <= 0:
                    continue
                density_norm = density / density.max()
                amplitude = 0.6
                baseline = idx * 1.1
                y_curve = baseline + density_norm * amplitude
                x_curve = 10**grid

                color = MODALITY_COLOR_MAP.get(label, "#6b7280")
                fill = _hex_to_rgba(color, 0.28)

                fig_kde.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_curve, x_curve[::-1]]),
                        y=np.concatenate([y_curve, np.full_like(y_curve, baseline)]),
                        name=label,
                        fill="toself",
                        fillcolor=fill,
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                fig_kde.add_trace(
                    go.Scatter(
                        x=x_curve,
                        y=y_curve,
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{label}</b><br>#Participants: %{{x:.0f}}<extra></extra>",
                        showlegend=False,
                    )
                )

                jitter = rng.uniform(0.02, amplitude * 0.5, size=len(vals))
                # Prepare custom data with dataset names and URLs
                custom_data = np.column_stack(
                    [subset["dataset"].to_numpy(), subset["dataset_url"].to_numpy()]
                )
                fig_kde.add_trace(
                    go.Scatter(
                        x=vals,
                        y=np.full_like(vals, baseline) + jitter,
                        mode="markers",
                        name=label,
                        marker=dict(color=color, size=8, opacity=0.6),
                        customdata=custom_data,
                        hovertemplate="<b><a href='%{customdata[1]}' target='_parent'>%{customdata[0]}</a></b><br>#Participants: %{x}<br><i>Click to view dataset details</i><extra></extra>",
                        showlegend=False,
                    )
                )

            if fig_kde.data:
                fig_kde.update_layout(
                    height=max(650, 150 * len(order)),
                    width=1200,  # Set explicit width for consistent sizing
                    template="plotly_white",
                    xaxis=dict(
                        type="log",
                        title="Number of Participants (Log Scale)",
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.08)",
                        zeroline=False,
                        dtick=1,
                        minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
                    ),
                    yaxis=dict(
                        title="Modality",
                        tickmode="array",
                        tickvals=[idx * 1.1 for idx in range(len(order))],
                        ticktext=order,
                        showgrid=False,
                        range=[-0.3, max(0.3, (len(order) - 1) * 1.1 + 0.9)],
                    ),
                    showlegend=False,
                    margin=dict(l=120, r=40, t=100, b=80),
                    title=dict(
                        text=f"<br><sub>Based on a EEG-Dash Datasets avaliables at {datetime.now().strftime('%d/%m/%Y')}.</sub>",
                        x=0.5,
                        xanchor="center",
                        y=0.98,
                        yanchor="top",
                    ),
                    autosize=True,  # Enable auto-sizing to fill container
                )

                # Add annotation highlighting Visual distribution
                fig_kde.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.98,
                    y=0.02,
                    text="Visual studies consistently use the<br>largest sample sizes, typically 20-30 participants",
                    showarrow=False,
                    font=dict(size=12, color="#111827"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(17,24,39,0.3)",
                    borderwidth=1,
                    borderpad=8,
                    xanchor="right",
                    yanchor="bottom",
                )
                # Add CSS and loading indicator for immediate proper sizing
                kde_height = max(650, 150 * len(order))
                plot_config = {
                    "responsive": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "participant_kde",
                        "height": kde_height,
                        "width": 1200,
                        "scale": 2,
                    },
                }
                fig_spec = fig_kde.to_plotly_json()
                data_json = json.dumps(fig_spec.get("data", []), cls=PlotlyJSONEncoder)
                layout_json = json.dumps(
                    fig_spec.get("layout", {}), cls=PlotlyJSONEncoder
                )
                config_json = json.dumps(plot_config, cls=PlotlyJSONEncoder)

                # Wrap with styling to ensure proper initial sizing and defer Plotly rendering
                styled_html = f"""
<style>
#dataset-kde-modalities {{
    width: 100% !important;
    max-width: 1200px;
    height: {kde_height}px !important;
    min-height: {kde_height}px;
    margin: 0 auto;
    display: none;
}}
#dataset-kde-modalities.plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
}}
.kde-loading {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: {kde_height}px;
    font-family: Inter, system-ui, sans-serif;
    color: #6b7280;
}}
</style>
<div class="kde-loading" id="kde-loading">Loading participant distribution...</div>
<div id="dataset-kde-modalities" class="plotly-graph-div"></div>
<script>
(function() {{
  const TARGET_ID = 'dataset-kde-modalities';
  const FIG_DATA = {data_json};
  const FIG_LAYOUT = {layout_json};
  const FIG_CONFIG = {config_json};

  function onReady(callback) {{
    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', callback, {{ once: true }});
    }} else {{
      callback();
    }}
  }}

  function renderPlot() {{
    const container = document.getElementById(TARGET_ID);
    if (!container) {{
      return;
    }}

    const draw = () => {{
      if (!window.Plotly) {{
        window.requestAnimationFrame(draw);
        return;
      }}

      window.Plotly.newPlot(TARGET_ID, FIG_DATA, FIG_LAYOUT, FIG_CONFIG).then((plot) => {{
        const loading = document.getElementById('kde-loading');
        if (loading) {{
          loading.style.display = 'none';
        }}
        container.style.display = 'block';

        plot.on('plotly_click', (event) => {{
          const point = event.points && event.points[0];
          if (!point || !point.customdata) {{
            return;
          }}
          const url = point.customdata[1];
          if (url) {{
            const resolved = new URL(url, window.location.href);
            window.open(resolved.href, '_self');
          }}
        }});
      }});
    }};

    draw();
  }}

  onReady(renderPlot);
}})();
</script>
"""

                kde_path = Path(target_dir) / "dataset_kde_modalities.html"
                with open(kde_path, "w", encoding="utf-8") as f:
                    f.write(styled_html)
                copyfile(kde_path, STATIC_DATASET_DIR / kde_path.name)
        except Exception as exc:
            print(f"[dataset KDE] Skipped due to error: {exc}")


def parse_freqs(value) -> str:
    if isinstance(value, str):
        freq = [int(float(f)) for f in value.strip("[]").split(",")]
        if len(freq) == 1:
            return f"{int(freq[0])}"
        else:
            return f"{int(np.median(freq))}*"

    elif isinstance(value, (int, float)) and not pd.isna(value):
        return f"{int(value)}"
    return ""  # for other types like nan


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()
    main(args.source_dir, args.target_dir)
    print(args.target_dir)
