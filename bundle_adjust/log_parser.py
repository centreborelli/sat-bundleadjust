"""
this script can be used to extract a summary of key data from a bundle adjustment log
data of interest:
    - running time per block (feature tracking vs optimization time vs total)
    - final reprojection error per camera
    - number of cameras corrected
    - number of cameras not corrected
"""

from pathlib import Path
import re
import pandas
import numpy as np

def parse_time_to_seconds(s: str) -> float:
    """
    Parse:
      - '138.28 seconds'
      - '78.75 seconds'
      - '00:07:18.07'
      - '00:12:32.16'
      - '12:32.16'   (optional support)
    Returns seconds as float.
    """
    s = s.strip()

    # e.g. "138.28 seconds"
    m = re.fullmatch(r"(?P<sec>\d+(?:\.\d+)?)\s*seconds?", s, flags=re.IGNORECASE)
    if m:
        return float(m.group("sec"))

    # e.g. "00:07:18.07" or "12:32.16"
    parts = s.split(":")
    if len(parts) == 3:
        h, m_, sec = parts
        return int(h) * 3600 + int(m_) * 60 + float(sec)
    if len(parts) == 2:
        m_, sec = parts
        return int(m_) * 60 + float(sec)

    raise ValueError(f"Unsupported time format: {s!r}")


def seconds_to_hms(seconds):
    if seconds is None:
        return None
    h = int(seconds // 3600)
    rem = seconds - 3600 * h
    m = int(rem // 60)
    s = rem - 60 * m
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ---------- robust extractors ----------

def _extract_block_done_in(text: str, block_header: str):
    """
    Find:
        <block_header>
        ...
        ...done in X seconds

    Uses a non-greedy match up to the first "...done in".
    """
    pattern = re.compile(
        rf"{re.escape(block_header)}\s*(.*?)\.\.\.done in\s+(\d+(?:\.\d+)?)\s+seconds",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return None
    return float(m.group(2))


def _extract_named_duration(text: str, label_regex: str):
    """
    Generic extractor for lines like:
      'Feature tracks computed in 00:07:18.07'
      'Optimization problem solved in 00:05:04.99 (330 iterations)'
      'Bundle adjustment pipeline completed in 00:12:32.16'
      'TOTAL TIME: 00:12:34.85'
    """
    pattern = re.compile(
        rf"{label_regex}\s*(?P<dur>\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    return parse_time_to_seconds(m.group("dur"))


def parse_ba_log_text(text: str):
    """
    Returns durations in seconds plus camera-level BA summary data.
    Missing values are None.
    """
    result = {
        "sift": _extract_block_done_in(text, "Running feature detection..."),
        "matching": _extract_block_done_in(text, "Matching..."),
        "tracks_total": _extract_named_duration(
            text, r"Feature tracks computed in"
        ),
        "optimization": _extract_named_duration(
            text, r"Optimization problem solved in"
        ),
        "total_time": _extract_named_duration(
            text, r"TOTAL TIME:\s*"
        ),
    }

    # Fallback for optimization if the HH:MM:SS line is missing
    if result["optimization"] is None:
        opt_took = re.findall(
            r"Optimization took\s+(\d+(?:\.\d+)?)\s+seconds",
            text,
            flags=re.IGNORECASE,
        )
        if opt_took:
            result["optimization"] = sum(float(x) for x in opt_took)

    # ---- new fields ----
    final_reproj = _extract_final_reprojection_error_per_camera(text)
    final_obs = _extract_final_observations_per_camera(text)
    n_corrected, n_total = _extract_corrected_camera_counts(text)
    uncorrected_paths = _extract_uncorrected_camera_paths(text)
    #print(uncorrected_paths)

    result["n_pairs"] = _extract_n_matching_pairs(text)
    result["n_tracks"] = _extract_n_feature_tracks(text)
    result["n_cc"] = _extract_n_connected_components(text)

    result["final_reprojection_error_per_camera"] = final_reproj # dict
    result["obs_per_cam"] = final_obs # dict
    if len(final_obs) > 0:
        result["total_obs"] = sum(final_obs.values())
        result["avg_obs_per_cam"] = np.array(
            list(final_obs.values())
        ).mean()
    else:
        result["total_obs"] = 0
        result["avg_obs_per_cam"] = [0]*n_total

    result["cams_corrected"] = n_corrected
    # Prefer explicit list length when available; otherwise infer from total-corrected
    if len(uncorrected_paths) > 0:
        result["cams_fail"] = len(uncorrected_paths)
    elif n_corrected is not None and n_total is not None:
        result["cams_fail"] = n_total - n_corrected
    else:
        result["cams_fail"] = None

    # Optional but useful
    result["cams_fail_paths"] = uncorrected_paths
    result["cams_total"] = n_total

    # useful for WorldView3 data: check how many cameras ended up with a mean reprojection error below 1 px 
    n_cams_below_1px = sum([1 for k, v in result['final_reprojection_error_per_camera'].items() if v < 1])
    n_cams_above_1px = n_total - n_cams_below_1px - result["cams_fail"]
    result['cams_<1px'] = n_cams_below_1px
    result['cams_>1px'] = n_cams_above_1px

    # average overall reprojection error
    result['err'] = np.array([v for k, v in result['final_reprojection_error_per_camera'].items()]).mean()

    return result

def _extract_n_connected_components(text):
    """
    Extract the number of connected components from the last line like:
        Connectivity graph: 1 connected components (CCs)

    Returns
    -------
    int or None
        Number of connected components, or None if not found.
    """
    matches = re.findall(
        r"Connectivity\s+graph\s*:\s*(\d+)\s+connected\s+components\s*\(CCs\)",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    # Use the last occurrence in the log.
    return int(matches[-1])

def _extract_n_matching_pairs(text):
    """
    Extract number of matching pairs from lines like:
        Total pairs to match : 205
    """
    m = re.search(
        r"Total\s+pairs\s+to\s+match\s*:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return int(m.group(1))

def _extract_n_feature_tracks(text):
    """
    Extract number of feature tracks from lines like:
        Found 17481 tracks in total
    """
    m = re.search(
        r"Found\s+(\d+)\s+tracks\s+in\s+total",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return int(m.group(1))

def _extract_final_reprojection_error_per_camera(text):
    """
    Extract per-camera final reprojection error from lines like:
        - cam   0 -  1948 obs - (mean before / mean after): 6.31 / 0.68

    Returns
    -------
    dict
        {camera_id: mean_after_pixels}
    """
    pattern = re.compile(
        r"^\s*-\s*cam\s+(\d+)\s*-\s*\d+\s+obs\s*-\s*"
        r"\(mean before / mean after\):\s*"
        r"[-+]?\d+(?:\.\d+)?\s*/\s*([-+]?\d+(?:\.\d+)?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    out = {}
    for cam_idx, mean_after in pattern.findall(text):
        out[int(cam_idx)] = float(mean_after)

    return out

def _extract_final_observations_per_camera(text):
    """
    Extract per-camera final observation counts from lines like:
        - cam   0 -  1948 obs - (mean before / mean after): 6.31 / 0.68

    Returns
    -------
    dict
        {camera_id: n_observations}
    """
    pattern = re.compile(
        r"^\s*-\s*cam\s+(\d+)\s*-\s*(\d+)\s+obs\s*-\s*"
        r"\(mean before / mean after\):",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    out = {}
    for cam_idx, n_obs in pattern.findall(text):
        out[int(cam_idx)] = int(n_obs)

    return out

def _extract_corrected_camera_counts(text):
    """
    Extract counts from lines like:
        Successfully corrected 40/42 cameras.

    Returns
    -------
    tuple
        (n_corrected, n_total) or (None, None) if not found
    """
    m = re.search(
        r"Successfully corrected\s+(\d+)\s*/\s*(\d+)\s+cameras",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None

    n_corrected = int(m.group(1))
    n_total = int(m.group(2))
    return n_corrected, n_total


def _extract_uncorrected_camera_paths(text):
    """
    Extract the list of RPC paths under the block:

        RPCs that could not be corrected:
        /path/one.rpc
        /path/two.rpc

    Stops at the first blank line after the block.

    Returns
    -------
    list[str]
    """
    m = re.search(
        r"RPCs that could not be corrected:\s*\n(?P<block>.*?)(?:\n\s*\n|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []

    block = m.group("block")
    paths = []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        camera_id = line.split("/")[-1]
        paths.append(camera_id)

    return paths

def compare_logs_table(
    log_files,
    tags=None,
    metrics=None,
    parser_func=None,
    round_digits=3,
    aggregation="mean",
):
    """
    Parse two or more BA log files and display a comparison table where:
      - each row = one tag, if tags are provided
      - rows with the same tag are aggregated together
      - each column = one selected metric

    Parameters
    ----------
    log_files : list
        List of log file paths.

    tags : list or None
        Optional display names, same length as log_files.
        Logs with the same tag are grouped and aggregated.
        Example: ["baseline", "baseline", "dinov2", "dinov2"]

    metrics : list or None
        Metrics to include as columns.
        Example: ["feature_extraction", "matching", "optimization", "total_time"]
        If None, a default subset is used.

    parser_func : callable or None
        Function that parses one log file and returns a dict of metrics.
        If None, assumes you already have a function named:
            parse_ba_log_file(log_file)

    round_digits : int
        Number of decimals for float rounding in the displayed table.

    aggregation : str or dict
        Aggregation method to use when multiple logs share the same tag.

        Options:
            "mean" : average numeric values by tag
            "sum"  : sum numeric values by tag
            "list" : concatenate all values into a list by tag

        You may also pass a dict to choose per-metric aggregation:
            {
                "feature_extraction": "mean",
                "matching": "mean",
                "feature_tracks_total": "sum",
                "optimization": "mean",
                "total_time": "sum",
            }

    Returns
    -------
    df : pandas.DataFrame
        Comparison table. If tags are provided, metrics are aggregated by tag.
    """
    import pandas as pd
    import json

    if parser_func is None:
        parser_func = parse_ba_log_file

    if metrics is None:
        metrics = [
            "feature_extraction",
            "matching",
            "feature_tracks_total",
            "optimization",
            "total_time",
        ]

    allowed_aggregations = {"mean", "sum", "list"}

    if isinstance(aggregation, str):
        if aggregation not in allowed_aggregations:
            raise ValueError(
                f"aggregation must be one of {sorted(allowed_aggregations)}"
            )
        aggregation_by_metric = {metric: aggregation for metric in metrics}

    elif isinstance(aggregation, dict):
        aggregation_by_metric = {}

        for metric in metrics:
            method = aggregation.get(metric, "mean")

            if method not in allowed_aggregations:
                raise ValueError(
                    f"Invalid aggregation for metric '{metric}': {method}. "
                    f"Must be one of {sorted(allowed_aggregations)}"
                )

            aggregation_by_metric[metric] = method

    else:
        raise TypeError("aggregation must be a string or a dict")

    if tags is not None and len(tags) != len(log_files):
        raise ValueError("tags must have the same length as log_files")

    rows = []

    for i, log_file in enumerate(log_files):
        parsed = parser_func(log_file)

        row = {
            "log_file": log_file,
            "tag": tags[i] if tags is not None else log_file,
        }

        for metric in metrics:
            value = parsed.get(metric, None)

            # Make nested structures display nicely in the table.
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)

            row[metric] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    if tags is not None:
        grouped = df.groupby("tag", sort=False)

        # New column: number of logs per tag.
        df_agg = grouped.size().reset_index(name="tag_count")

        for metric in metrics:
            if metric not in df.columns:
                continue

            method = aggregation_by_metric[metric]

            if method in {"mean", "sum"}:
                if pd.api.types.is_numeric_dtype(df[metric]):
                    values_by_tag = grouped[metric].agg(method).reset_index()
                else:
                    # Non-numeric values cannot be meaningfully averaged or summed.
                    # Fall back to compact unique-value lists.
                    values_by_tag = (
                        grouped[metric]
                        .apply(lambda s: list(s.dropna()))
                        .reset_index()
                    )

            elif method == "list":
                values_by_tag = (
                    grouped[metric]
                    .apply(lambda s: list(s.dropna()))
                    .reset_index()
                )

            df_agg = df_agg.merge(values_by_tag, on="tag", how="left")

        df = df_agg

        # Round numeric columns for readability.
        for metric in metrics:
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                if metric in ["sift", "matching", "tracks_total", "optimization", "total_time", "n_tracks"]:
                    df[metric] = df[metric].round(3) # no decimals needed
                else:
                    df[metric] = df[metric].round(round_digits)

        ordered_cols = ["tag", "tag_count"] + metrics

    else:
        # No tags: preserve original behavior, one row per log file.
        for metric in metrics:
            if metric in df.columns and pd.api.types.is_float_dtype(df[metric]):
                df[metric] = df[metric].round(round_digits)

        ordered_cols = ["log_file"] + metrics

    existing_cols = [c for c in ordered_cols if c in df.columns]
    df = df[existing_cols]

    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))

    return df


def plot_metric_list_by_tag(
    df,
    metric,
    tag_col="tag",
    xtick_labels=None,
    ymax=None,
    figsize=(8, 5),
    marker="o",
    title=None,
    xlabel="Log",
    ylabel=None,
    rotate_xticks=45,
):
    """
    Plot one metric from a comparison DataFrame where the metric values
    were aggregated as lists.

    Parameters
    ----------
    df : pandas.DataFrame
        Output DataFrame from compare_logs_table(..., aggregation="list").

    metric : str
        Metric column to plot.

    tag_col : str
        Column containing the tag/group name. Default is "tag".

    xtick_labels : list of str or None
        Optional labels for the x axis.

        Each label is expected to be a string in the format:
            XXX_YYY

        If provided, these labels replace the numeric log/list indices
        on the x axis.

    ymax : int, float, or None
        Optional upper limit for the y axis.

        Example:
            ymax=100

        If None, matplotlib chooses the y-axis limit automatically.

    figsize : tuple
        Matplotlib figure size.

    marker : str
        Marker style for each data point.

    title : str or None
        Optional plot title. If None, one is generated automatically.

    xlabel : str
        Label for the x axis.

    ylabel : str or None
        Label for the y axis. If None, uses the metric name.

    rotate_xticks : int or float
        Rotation angle for x tick labels.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    import ast
    import json
    import pandas as pd
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' is not a column in the DataFrame.")

    if tag_col not in df.columns:
        raise ValueError(f"Tag column '{tag_col}' is not a column in the DataFrame.")

    if xtick_labels is not None:
        if not isinstance(xtick_labels, list):
            raise TypeError("xtick_labels must be a list of strings.")

        if not all(isinstance(label, str) for label in xtick_labels):
            raise TypeError("All xtick_labels must be strings.")

        invalid_labels = [
            label for label in xtick_labels
            if "_" not in label or len(label.split("_")) != 2
        ]

        if invalid_labels:
            raise ValueError(
                "All xtick_labels must be in the format 'XXX_YYY'. "
                f"Invalid labels: {invalid_labels}"
            )

    if ymax is not None:
        if not isinstance(ymax, (int, float)):
            raise TypeError("ymax must be an int, float, or None.")

    def to_list(value):
        """
        Convert stored list-like values back into Python lists when needed.
        Handles actual lists, JSON strings, and Python-list strings.
        """
        if isinstance(value, list):
            return value

        if pd.isna(value):
            return []

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            return [value]

        return [value]

    fig, ax = plt.subplots(figsize=figsize)

    max_list_len = 0

    for _, row in df.iterrows():
        tag = row[tag_col]
        values = to_list(row[metric])

        max_list_len = max(max_list_len, len(values))

        numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
        valid_mask = numeric_values.notna()

        x = list(range(len(values)))
        y = numeric_values.tolist()

        # Keep only numeric values for plotting.
        x = [idx for idx, is_valid in zip(x, valid_mask) if is_valid]
        y = [val for val, is_valid in zip(y, valid_mask) if is_valid]

        if len(y) == 0:
            continue

        ax.plot(x, y, marker=marker, label=str(tag))

    if xtick_labels is not None:
        if len(xtick_labels) < max_list_len:
            raise ValueError(
                f"xtick_labels has length {len(xtick_labels)}, but the longest "
                f"metric list has length {max_list_len}."
            )

        tick_positions = list(range(max_list_len))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            xtick_labels[:max_list_len],
            rotation=rotate_xticks,
            ha="right",
        )
    else:
        ax.set_xticks(list(range(max_list_len)))

    if ymax is not None:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    ax.set_title(title if title is not None else f"{metric} by log/list index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax


def parse_ba_log_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    out = parse_ba_log_text(text)
    out["log_path"] = str(path)
    return out