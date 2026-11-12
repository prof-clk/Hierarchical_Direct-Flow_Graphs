from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from hdfg import (
    EventLogConfig,
    HierLevel,
    RenderSettings,
    run_analysis, 
    share_change, 
    build_pipeline, 
    diagnose_flow, 
    load_event_log, 
    render_process_map,
    )


st.set_page_config(page_title="H-DFG", layout="wide")
st.title("Hierarchical Direct-Flow Graph\nfor Process Mining")

# ============================================================================
# Sidebar: column mapping UI (Streamlit counterpart of the notebook's
# "Column mapping" settings cell).
# ============================================================================
_DEFAULT_LABELS = ("Workcenter", "Workstation")
_CODE_DEFAULTS = ("WorkCenterCode", "WorkStationCode")
_NAME_DEFAULTS = ("WorkCenterName", "WorkStationName")


def _pick(cols, *names, optional=False):
    """Index of the first candidate name found in cols (pre-selects sensible defaults)."""
    options = (["(none)"] if optional else []) + list(cols)
    for n in names:
        if n in options:
            return options.index(n)
    return 0


def build_config_ui(df: pd.DataFrame) -> EventLogConfig:
    cols = list(df.columns)

    st.sidebar.subheader("2. Columns")
    case_id_col = st.sidebar.selectbox("Order / Case ID", cols, index=_pick(cols, "OrderId"))
    activity_code_col = st.sidebar.selectbox("Operation code", cols, index=_pick(cols, "OperationCode"))
    activity_name_col = st.sidebar.selectbox("Operation name (optional)", ["(none)"] + cols, index=_pick(cols, "OperationName", optional=True))

    st.sidebar.subheader("3. Time model")
    st.sidebar.caption("Interval logs use Start/Finish; for flow events a single timestamp may also be given.")
    use_interval = st.sidebar.checkbox("Interval event log (Start/Finish)", value="StartTime" in cols)
    if use_interval:
        start_col = st.sidebar.selectbox("Start time", cols, index=_pick(cols, "StartTime"))
        end_col = st.sidebar.selectbox("Finish time", cols, index=_pick(cols, "FinishTime"))
        timestamp_col = None
    else:
        timestamp_col = st.sidebar.selectbox("Timestamp", cols, index=_pick(cols, "timestamp"))
        start_col = end_col = None

    st.sidebar.subheader("4. Resource / other dimensions")
    resource_cols = st.sidebar.multiselect("Resource columns (shown inside operation boxes)", cols, default=[c for c in ("User",) if c in cols])
    other_cols = st.sidebar.multiselect("Other dimensions", cols)

    st.sidebar.subheader("5. Hierarchy (outer to inner)")
    n_levels = int(st.sidebar.number_input("Number of levels", min_value=1, max_value=5, value=2))
    levels: list[HierLevel] = []
    for i in range(n_levels):
        st.sidebar.markdown(f"**Level {i}**")
        code_col = st.sidebar.selectbox(f"Code column #{i}", cols, key=f"lvl_code_{i}", index=_pick(cols, *(_CODE_DEFAULTS[i:i + 1])))
        name_col = st.sidebar.selectbox(f"Name column #{i} (optional)", ["(none)"] + cols, key=f"lvl_name_{i}", index=_pick(cols, *(_NAME_DEFAULTS[i:i + 1]), optional=True))
        label = st.sidebar.text_input(f"Label #{i}", value=_DEFAULT_LABELS[i] if i < 2 else f"Level {i}", key=f"lvl_label_{i}")
        levels.append(HierLevel(code_col, None if name_col == "(none)" else name_col, label))

    st.sidebar.subheader("6. Flow events (optional)")
    use_flow = st.sidebar.checkbox("Dataset contains flow events", value="IsFlowEvent" in cols)
    is_flow_col = flow_lvl_col = flow_code_col = flow_ts_col = None
    if use_flow:
        is_flow_col = st.sidebar.selectbox("IsFlowEvent column", cols, index=_pick(cols, "IsFlowEvent"))
        flow_lvl_col = st.sidebar.selectbox("FlowAttachLevel column (0 = outermost)", cols, 
                                            index=_pick(cols, "FlowAttachLevel"))
        flow_code_col = st.sidebar.selectbox("FlowAttachCode column", cols, 
                                             index=_pick(cols, "FlowAttachCode"))
        pick = st.sidebar.selectbox("Flow event time column", ["(Same as Start / Timestamp)"] + cols,
                                    index=_pick(cols, "timestamp", optional=True) if "timestamp" in cols else 0)
        flow_ts_col = None if pick.startswith("(") else pick

    return EventLogConfig(
        case_id_col=case_id_col,
        activity_code_col=activity_code_col,
        activity_name_col=None if activity_name_col == "(none)" else activity_name_col,
        timestamp_col=timestamp_col,
        start_timestamp_col=start_col,
        end_timestamp_col=end_col,
        resource_cols=tuple(resource_cols),
        other_cols=tuple(other_cols),
        hierarchy_levels=tuple(levels),
        is_flow_event_col=is_flow_col,
        flow_attach_level_col=flow_lvl_col,
        flow_attach_code_col=flow_code_col,
        flow_timestamp_col=flow_ts_col,
    )


def _show_svg(svg: str, height: int = 800):
    """Show SVG in an iframe (so zoom/pan/hover JS works)."""
    st.iframe("<html><body style='margin:0;background:#fff'>" + svg + "</body></html>", height=height)


def _arrow_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type (e.g., "All" + years) object columns to strings; prevents Arrow serialization warnings."""
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(type).nunique() > 1:
            out[c] = out[c].astype(str)
    return out


def _table(frame: pd.DataFrame, _file_name: str):
    st.download_button(f"Download {_file_name} CSV", frame.to_csv(index=False), file_name=_file_name + ".csv", mime="text/csv")
    st.dataframe(_arrow_safe(frame), width="stretch")


def _render_svg_to_string(pipeline, cfg, agg_level, settings, only_codes, mode) -> str:
    fd, path = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        render_process_map(pipeline, cfg, agg_level, settings, only_codes, mode, out_path=path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


# ============================================================================
# 1) FILE
# ============================================================================
st.sidebar.subheader("1. File")
if "log" not in st.session_state:
    st.session_state.log = None

log_file = st.sidebar.file_uploader("Event Log CSV or Excel File (required)", type=["csv", "xlsx", "xls"])

if log_file is not None:
    try:
        st.session_state.log = load_event_log(log_file.getvalue(), log_file.name)
        st.sidebar.success("Event Log File\n Successfully Uploaded!")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"File could not be read: {e}")
        st.stop()

if st.sidebar.button("Load Sample Data", type="primary"):
    try:
        st.session_state.log = pd.read_csv("log.csv.gz", dtype=str)
        st.sidebar.info("Sample data loaded.")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"Could not read sample data: {e}")

if st.session_state.log is None:
    st.info("Upload the Event Log from the left sidebar to begin.")
    st.stop()

raw_df = st.session_state.log.copy()

# ============================================================================
# 2) SETTINGS
# ============================================================================
cfg = build_config_ui(raw_df)

st.sidebar.subheader("7. View")
level_options: list[tuple[int | str, str]] = [(i, l.label) for i, l in enumerate(cfg.hierarchy_levels)]
level_options.append(("activity", "Activity / Operation (most detailed)"))
agg_choice = st.sidebar.selectbox("Draw at level", level_options, format_func=lambda x: x[1], index=0)
agg_level = agg_choice[0]
path_filter = st.sidebar.slider("Path filter (lower = remove more rare paths)", 0.0, 1.0, 1.0, 0.05)
only_codes_raw = st.sidebar.text_input("Only these outermost codes (comma-separated, empty = all)", "")
only_codes = {c.strip() for c in only_codes_raw.split(",") if c.strip()} or None

with st.sidebar.expander("Advanced"):
    order_by = st.selectbox("Box ordering", ["flow", "code", "volume", "degree"], index=0)
    rankdir = st.selectbox("Flow direction", ["LR", "TB"], index=0)
    edge_min_count = st.number_input("Minimum edge frequency", min_value=1, value=1)
    label_top_n = st.number_input("Always label the top N edges", min_value=0, value=40)
    top_child_rows = st.number_input("Sub-rows inside a box (0 = all)", min_value=0, value=0)

settings = RenderSettings(order_by=order_by, rankdir=rankdir, edge_min_count=int(edge_min_count),
                          label_top_n=int(label_top_n), top_child_rows=int(top_child_rows))

# ============================================================================
# 3) GENERATE
# ============================================================================
if st.sidebar.button("Generate", type="primary"):
    res: dict = {}
    try:
        pipeline = build_pipeline(raw_df, cfg, agg_level, path_filter)
    except ValueError as e:
        st.error(f"Data preparation error: {e}")
        st.stop()

    res["counts"] = (len(pipeline["activities_filtered"]), len(pipeline["dfg_filtered"]))
    
    if cfg.is_flow_event_col:
        res["flow_diag"] = diagnose_flow(pipeline["df"], cfg, pipeline)

    for mode in ("freq", "perf"):
        try:
            res[mode] = _render_svg_to_string(pipeline, cfg, agg_level, settings, only_codes, mode)
        except ValueError as e:
            res[mode] = None
            st.warning(f"{'Frequency' if mode == 'freq' else 'Performance'} map could not be rendered: {e}")
    
    try:
        res["analysis"] = run_analysis(pipeline, cfg)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        st.warning(f"Analysis tables could not be generated: {e}")
    
    st.session_state["res"] = res


# ============================================================================
# 4) VIEW
# ============================================================================
res = st.session_state.get("res") # type: ignore

if not res:
    st.info("Map the columns and click **Generate**.")
    st.stop()

st.caption(f"Filtered nodes: {res['counts'][0]:,} | Filtered edges: {res['counts'][1]:,}")
fd = res.get("flow_diag")
if fd:
    with st.expander("Flow event diagnosis", expanded=bool(fd["notes"])):
        st.dataframe(pd.DataFrame(fd["steps"], columns=["Step", "Count"]), hide_index=True, width="stretch")
        for n in fd["notes"]:
            st.warning(n)

level_name = level_options[int(agg_level)][1] if isinstance(agg_level, int) else "Activity / Operation"
order_name = order_by.capitalize() if order_by in ("flow", "code", "volume", "degree") else order_by
st.header(f"{level_name} | Order by: {order_name}")
labels = ["📊 Frequency", "⏱️ Performance (duration)"]
if res.get("analysis"):
    labels.append("📈 Analysis")
tabs = dict(zip(labels, st.tabs(labels)))
for key, label in (("freq", labels[0]), ("perf", labels[1])):
    with tabs[label]:
        svg = res.get(key)
        if svg:
            st.download_button("Download SVG", svg, 
                               file_name=f"{level_name}_H-DFG_{key}_{order_name}.svg",
                               mime="image/svg+xml", 
                               key=f"dl_{key}"
                               )
            _show_svg(svg)
        else:
            st.info("No edges found for this view.")

if "📈 Analysis" in tabs:
    with tabs["📈 Analysis"]:
        an = res["analysis"]

        st.subheader("Event Log Summary")
        _table(an["summary"], f"{level_name} Event Log Summary")

        st.subheader("Overall Yearly Overview")
        _table(an["yearly"], "Overall Yearly Overview")

        st.subheader("Order Lead Time (first to last event, flow events included)")
        _table(an["case_durations"], "Overall Order Lead Time First to Last Event Including Flow Events")
        
        st.subheader("Top Trace Variants")
        _table(an["variants"], f"{level_name} Top Trace Variants")
        
        st.subheader("Start / End Nodes of Orders")
        _table(an["start_end"], f"{level_name} Start to End Nodes of Orders")
        
        st.subheader("Overall Flow Events")
        _table(an["flow_events"], "Overall Flow Events")
        
        st.subheader("Most Frequent Node → Node Flows")
        _table(an["flows"], f"{level_name} Most Frequent Node to Node Flows")
        
        st.subheader("Most Busiest Node → Node Flows")
        _table(an["busiest_flows"], f"{level_name} Most Busiest Node to Node Flows")

        st.subheader("Bottlenecks: waiting / transfer / flow-event Edges by Total Time")
        _table(an["bottlenecks"], f"{level_name} Bottlenecks Edges by Total Time")
        
        st.subheader("Nodes")
        _table(an["nodes"], f"{level_name} Nodes")
        
        if not an["transfer"].empty:
            st.subheader("Transfer (waiting) time — by separation level")
            _table(an["transfer"], "Overall Transfer Waiting Time by Separation Level")

        if level_name != "activity":
            st.subheader("**Ordered by Operation Events (descending)**")
            _table(an["levels"][int(agg_level)]["ranking"], f"{level_name} Descending Ordered by Operation Events")

        if level_name != "activity":
            st.subheader("**The Slowest Records Includes More Than 30 Events Descending by AVG Operation Duration**")
            slow = an["levels"][int(agg_level)]["ranking"]
            slow = slow[slow["Operation events"] >= 30]
            slow = slow.iloc[np.argsort(-slow["Avg op. duration (sec)"].to_numpy(dtype=float), kind="stable")]
            _table(slow, f"{level_name} Slowest Records Include More Than 30 Events Descending by AVG Operation Duration")

        if level_name != "activity":
            years = [int(y) for y in an["yearly"].index]
            if len(years) >= 2:
                sc = share_change(an["levels"][int(agg_level)]["year_share"], years[0], years[-1])
                st.subheader(f"**Share Change ({years[0]} → {years[-1]})**")
                _table(sc, f"{level_name} Share Change")

