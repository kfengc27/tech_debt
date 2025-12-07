# plots.py
import io, base64
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.dates as mdates  # 确保有这行
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from typing import Optional
from typing import Iterable, List, Tuple, Optional, Dict, Any, Union

def compute_datetime_upper_envelope(
    x,
    y,
    bins: int = 300,
    q: float = 0.97,
    roll: int = 5,
    min_bins: int = 10,
):
    """
    Build a time-based upper envelope for (x, y) where x are datetimes (any tz) and y are numeric.
    Returns both detailed arrays and a highlight_df (x_mid, y_envelope) for plotting or export.
    """
    # --- 1) Clean & normalize types ---
    x = pd.to_datetime(x, errors="coerce")
    x = pd.DatetimeIndex(x)
    if x.tz is not None:
        x = x.tz_convert("UTC").tz_localize(None)

    y = np.asarray(pd.to_numeric(y, errors="coerce"))

    # --- 2) Filter valid rows ---
    m = (~x.isna()) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        raise ValueError("No valid data after cleaning.")

    # --- 3) Convert datetime → int64 ns ---
    xn = x.asi8

    # --- 4) Sort ---
    idx = np.argsort(xn)
    xs, ys = xn[idx], y[idx]
    xs_time = pd.to_datetime(xs)

    # --- 5) Bin + envelope ---
    B = int(min(bins, max(min_bins, xs.size // 5)))
    edges = np.linspace(xs.min(), xs.max(), B + 1)
    which = np.clip(np.digitize(xs, edges, right=True) - 1, 0, B - 1)

    yq = np.full(B, np.nan)
    for i in range(B):
        yi = ys[which == i]
        if yi.size:
            yq[i] = np.quantile(yi, q)

    x_mid = pd.to_datetime(((edges[:-1] + edges[1:]) / 2).astype("int64"), unit="ns")
    line = pd.Series(yq).rolling(roll, center=True, min_periods=1).median().to_numpy()
    mask = np.isfinite(line)

    # --- 6) Build highlight_df ---
    highlight_df = pd.DataFrame({
        "x_mid": x_mid[mask],
        "y_envelope": line[mask]
    })

    return {
        "xs_time": xs_time,    # cleaned and sorted timestamps
        "ys": ys,              # numeric series
        "x_mid": x_mid,        # bin midpoints
        "y_env": line,         # smoothed envelope
        "mask": mask,          # finite mask
        "highlight_df": highlight_df  # DataFrame ready for plotting/export
    }

def compute_adjR2(n, sx, sy, sxx, sxy, syy, p=2, eps=1e-12):
    """
    使用累积量直接计算一阶线性回归的 Adjusted R^2.

    参数
    ----
    n   : 样本数
    sx  : sum(x)
    sy  : sum(y)
    sxx : sum(x^2)
    sxy : sum(x*y)
    syy : sum(y^2)
    p   : 模型参数个数 (一元线性回归通常为 2: β0, β1)
    eps : 数值稳定用的极小量
    """

    # 样本太少，无法拟合
    if n <= p:
        return 0.0

    # 计算回归系数 β1, β0
    denom = n * sxx - sx * sx
    if abs(denom) < eps:
        # x 没有变化，无法回归，返回 0
        return 0.0

    beta1 = (n * sxy - sx * sy) / denom
    beta0 = (sy - beta1 * sx) / n

    # SSE: Residual Sum of Squares
    # 利用闭式：SSE = syy - β0 * sy - β1 * sxy
    SSE = syy - beta0 * sy - beta1 * sxy

    # SST: Total Sum of Squares
    # SST = Σ(y - ȳ)^2 = syy - sy^2 / n
    SST = syy - (sy * sy) / n

    if SST <= eps:
        # y 基本不变，R² 没意义，视为 0
        return 0.0

    # R²
    R2 = 1.0 - SSE / SST

    # Adjusted R²
    # adjR2 = 1 - (1 - R2) * (n - 1) / (n - p)
    denom_adj = (n - p)
    if denom_adj <= 0:
        return R2

    adjR2 = 1.0 - (1.0 - R2) * (n - 1) / denom_adj

    # 防止数值误差超界，夹到 [-1, 1]
    if adjR2 > 1.0:
        adjR2 = 1.0
    elif adjR2 < -1.0:
        adjR2 = -1.0

    return float(adjR2)


def compute_MSE(n, sx, sy, sxx, sxy, syy, p=2, eps=1e-12):
    """
    使用累积量直接计算一阶回归的 MSE (Mean Squared Error)
    n   = 样本数
    sx  = sum(x)
    sy  = sum(y)
    sxx = sum(x^2)
    sxy = sum(x*y)
    syy = sum(y^2)
    p   = 参数数量(线性回归为2: β0, β1)
    """

    if n < p:
        return float("inf")  # 不足以拟合，视为极差

    denom = (n * sxx - sx * sx)

    if abs(denom) < eps:  # 避免除0
        return float("inf")

    # 回归系数
    beta1 = (n * sxy - sx * sy) / denom
    beta0 = (sy - beta1 * sx) / n

    # 残差平方和 SSE
    SSE = syy - beta0 * sy - beta1 * sxy

    # **最终 MSE**
    return max(SSE / n, 0)  # 防数值负误差


def split_by_metric(points, drop_thre=0.01, w=2, min_len=8,
                    rel_drop=False, rel_drop_thre=0.02, k=1,
                    metric='mse', eps=1e-12):
    """
    通用分段：可用 AdjR² 或 MSE 作为监控指标。
    points: list of (x, y) sorted by x
    metric: 'adjr2' 或 'mse'
    """

    # 你已有的两个指标函数（请确保这两个函数已定义在同一作用域中）
    # compute_adjR2(n, sx, sy, sxx, sxy, syy, p, eps=1e-12) -> float
    # compute_MSE(n, sx, sy, sxx, sxy, syy, p, eps=1e-12)   -> float

    if metric.lower() == 'adjr2':
        metric_fn = lambda n, sx, sy, sxx, sxy, syy, p: compute_adjR2(n, sx, sy, sxx, sxy, syy, p, eps)
        better_is_higher = True
        small = 1e-6  # 用于相对比值的分母保护

    elif metric.lower() == 'mse':
        metric_fn = lambda n, sx, sy, sxx, sxy, syy, p: compute_MSE(n, sx, sy, sxx, sxy, syy, p, eps)
        better_is_higher = False
        small = 1e-12
    else:
        raise ValueError("metric must be 'adjr2' or 'mse'")

    p = k + 1

    segments = []

    breakpoints = []

    def reset_state():
        # 返回：n, sx, sy, sxx, sxy, syy, last_metric, bad_streak
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0
    
    # 初始化
    start = 0
    n = sx = sy = sxx = sxy = syy = 0
    last_metric = None
    bad_streak = 0

    # 为了在切分时记录当时的指标值，我们同时跟踪“上一次用于比较的指标值”
    metric_at_prev = None
    for t, (x, y) in enumerate(points):
        # 更新累计量
        n += 1
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y

        # 起始长度不足则跳过
        if n < max(min_len, p + 1):
            continue

        # 当前区段的指标
        cur_metric = metric_fn(n, sx, sy, sxx, sxy, syy, p)
 
        # 检查是否“变坏”
        bad = False
        if last_metric is not None:
            delta = cur_metric - last_metric
            if better_is_higher:
                # AdjR²：下降为坏（delta < -阈值）
                if delta < -drop_thre:
                    bad = True
                elif rel_drop:
                    denom = max(abs(last_metric), small)
                    if (delta / denom) < -rel_drop_thre:
                        bad = True
            else:
                # MSE：上升为坏（delta > +阈值）
                if delta > +drop_thre:
                    bad = True
                elif rel_drop:
                    denom = max(abs(last_metric), small)
                    if (delta / denom) > +rel_drop_thre:
                        bad = True

        # 递推坏计数
        if bad:
            bad_streak += 1
        else:
            bad_streak = 0


        # 更新“上一时刻的指标”
        last_metric = cur_metric
        metric_at_prev = cur_metric  # 记录最近一次计算值

        # 连续坏满 w 次 → 切分
        if bad_streak >= w:
            cut = t - w  # 回看窗口的起点作为切点
            left_len = cut - start + 1

            if left_len >= max(min_len, p + 1):
                # 记录左段（为了更稳妥，可在 cut 位置重算一次指标，但这里沿用最近值）
                segments.append({
                    "start": start,
                    "end": cut,
                    "metric": metric,
                    "value": last_metric
                })
                breakpoints.append(cut)

                # 重置从 cut+1 到当前 t 的累积
                start = cut + 1
                n, sx, sy, sxx, sxy, syy, last_metric, bad_streak = reset_state()
                # 把右侧残余点重新累积起来（含 t-w+1...t）
                for u in range(start, t + 1):
                    x_u, y_u = points[u]
                    n += 1
                    sx += x_u
                    sy += y_u
                    sxx += x_u * x_u
                    syy += y_u * y_u
                    sxy += x_u * y_u
                # 重置后，last_metric 置空，让下一轮重新建立基线
                last_metric = None
                metric_at_prev = None

    # 收尾段

    segments.append({
        "start": start,
        "end": len(points) - 1,
        "metric": metric,
        "value": metric_at_prev
    })
    
    return segments, breakpoints

def plot_accumulative_complexity(df: pd.DataFrame, value_col: str, title="Accumulative Complexity", freq="W"):
    """
    Build a cumulative-sum chart over time for a numeric column.
    If no real date column exists, uses a synthetic timeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    value_col : str
        Column containing complexity or numeric values.
    title : str
        Plot title.
    freq : str
        Frequency for resampling ('D', 'W', or 'M').

    Returns
    -------
    b64 : str
        Base64-encoded PNG chart.
    """

    # --- Generate synthetic timeline if no explicit date col ---
    # --- Generate REAL timeline instead of fake one ---
    df = df.dropna(subset=[value_col]).copy()
    if "x_mid" in df.columns:
        df["x_mid"] = pd.to_datetime(df["x_mid"], errors="coerce")
        df = df.dropna(subset=["x_mid"])
        df = df.sort_values("x_mid")
        time_col = "x_mid"  # ← 关键：用真实时间
    else:
        # ⚠️ fallback：无日期才造假
        start = pd.Timestamp("2021-01-01")
        end   = pd.Timestamp("2024-12-31")
        df["_fake_date"] = pd.date_range(start, end, periods=len(df))
        time_col = "_fake_date"
        df = df.sort_values(time_col)
 
 

    # --- Aggregate & accumulate ---
    agg_df = (
        df.set_index(time_col)[value_col]
        .resample(freq)
        .mean()
        .dropna()
        .to_frame(name=value_col)
        .reset_index()
    )

    agg_df["accumulative_complexity"] = agg_df[value_col].cumsum()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(
        agg_df[time_col],
        agg_df["accumulative_complexity"],
        color="steelblue",
        linewidth=2,
        marker="o",
        label="Accumulative Complexity"
    )

    ax.set_title(f"{title} ({'Weekly' if freq=='W' else 'Monthly'})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Accumulative Complexity")
    ax.grid(alpha=0.8)
    ax.legend(loc="best")

    # --- Convert to base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---- helpers ----
def detect_date_col(df):
    # 1) names that hint at date/time
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "time", "timestamp", "created"]):
            return c
    # 2) try parsing any object column; keep the one with best success rate
    best_c, best_ratio = None, 0
    for c in df.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        ratio = parsed.notna().mean()
        if ratio > best_ratio and ratio >= 0.7:
            best_c, best_ratio = c, ratio
    return best_c

def detect_value_col(df, exclude=None):
    exclude = set(exclude or [])
    # prefer columns with complexity-ish names
    for c in df.columns:
        if c in exclude:
            continue
        if ("complex" in c.lower() or "variation" in c.lower() or "delta" in c.lower()) \
            and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # else first numeric that isn't obviously an id/index
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not any(t in c.lower() for t in ["id","index","idx"]):
            return c
    return None


def plot_time_vs_complexity(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    prefer_line: bool = True,   # set False to use bars
    freq: str,    # 'D' | 'W' | 'M' | None (no resample)
    title: str = "Corresponding Complexity Change over Time",
) -> tuple[str, str, str]:
    """
    Returns (b64_png, resolved_time_col, resolved_value_col).
    - Detects/validates value_col: tries numeric-friendly columns; uses your 70% rule
    - Detects/validates date_col: if parsing fails, uses index-based time axis
    - Optional resample for nicer ticks
    """
    df = df.copy()

    # --- Detect/confirm date col ---
    if date_col is None:
        # try obvious date/time-like columns first
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ["date", "time", "timestamp", "created", "dt"]):
                try:
                    parsed = pd.to_datetime(df[c], errors="raise")
                    df[c] = parsed
                    date_col = c
                    break
                except Exception:
                    pass
        if date_col is None:
            # try any column that parses reasonably well
            best, score = None, 0.0
            for c in df.columns:
                try:
                    p = pd.to_datetime(df[c], errors="coerce")
                    ok = p.notna().mean()
                    if ok > score and ok >= 0.7:
                        best, score = c, ok
                except Exception:
                    continue
            date_col = best

    # --- Detect/confirm value col (your 70% numeric rule) ---
    if value_col is None:
        for c in df.columns:
            if c == date_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.7:
                df[c] = s
                value_col = c
                break
    if value_col is None:
        raise RuntimeError(
            "Could not find a numeric 'complexity' column. Columns: " + ", ".join(map(str, df.columns))
        )

    # --- Build a usable time axis ---
    if date_col is None:
        df["_time"] = np.arange(len(df))
        time_col = "_time"
    else:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if parsed.notna().mean() < 0.7:
            df["_time"] = np.arange(len(df))
            time_col = "_time"
        else:
            df[date_col] = parsed
            time_col = date_col

    # --- Clean & sort ---
    vals = pd.to_numeric(df[value_col], errors="coerce")
    m = vals.replace([np.inf, -np.inf], np.nan).notna() & df[time_col].notna()
    plot_df = df.loc[m, [time_col, value_col]].sort_values(by=time_col).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No valid rows to plot after cleaning.")

    # --- Optional resample for nicer timeline (only if time is datetime-like) ---
    if freq in {"D", "W", "M"} and np.issubdtype(plot_df[time_col].dtype, np.datetime64):
        plot_df = (
            plot_df.set_index(time_col)[value_col]
                   .resample(freq).mean()
                   .dropna()
                   .to_frame(name=value_col)
                   .reset_index()
        )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    if prefer_line or not np.issubdtype(plot_df[time_col].dtype, np.datetime64):
        ax.plot(plot_df[time_col], plot_df[value_col], linewidth=1.8)
    else:
        ax.bar(plot_df[time_col], plot_df[value_col])

    ax.set_title(title)
    ax.set_ylabel("Complexity")
    ax.set_xlabel("Time")
    if np.issubdtype(plot_df[time_col].dtype, np.datetime64):
        fig.autofmt_xdate(rotation=15)
    else:
        # many points? reduce tick crowding
        n = len(plot_df)
        if n > 20:
            step = max(1, n // 12)
            ax.set_xticks(plot_df[time_col][::step])
    ax.grid(alpha=0.85)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return b64, time_col, value_col


# extract segments
def extract_segments(splits):
    """
    将多种形态的 segments 容器，统一解析成：
    [{'start': int, 'end': int, 'AdjR2': float}, ...]
    可能的形态包括：
      - [dict, dict, ...]
      - ( [dict, dict, ...], [boundaries...] )
      - {'segments': [dict, ...]}
      - {0:{...},1:{...}}  # dict-of-dicts
    """
    # list/tuple
    if isinstance(splits, (list, tuple)):
        if len(splits) > 0 and isinstance(splits[0], dict):
            return list(splits)
        if len(splits) > 0 and isinstance(splits[0], (list, tuple)):
            cand = splits[0]
            if len(cand) > 0 and isinstance(cand[0], dict):
                return list(cand)
        if all(isinstance(s, dict) for s in splits):
            return list(splits)
    # dict
    if isinstance(splits, dict):
        if "segments" in splits and isinstance(splits["segments"], (list, tuple)):
            return list(splits["segments"])
        vals = list(splits.values())
        if vals and isinstance(vals[0], dict) and "start" in vals[0] and "end" in vals[0]:
            return vals
    raise ValueError("Cannot parse `segs` into a list of segment dicts.")


import matplotlib.dates as mdates

def plot_segments(df: pd.DataFrame, date_col: str, value_col: str):
    # -------- 参数：end 是否为闭区间（默认 False：右开区间） --------
    END_IS_INCLUSIVE = False

    # 把日期列变成 datetime → 数值轴 (seconds)
    dt = pd.to_datetime(df[date_col], errors="coerce")
    x_num = dt.view("int64") / 1e9  # ns → seconds
    y_num = df[value_col].to_numpy(dtype=float)

    points = np.column_stack([x_num, y_num])

    segs, cuts = split_by_metric(
        points,
        drop_thre=0.001,
        w=1,
        min_len=8,
        rel_drop=True,
        rel_drop_thre=0.05,
        k=1,
        metric='mse'
    )

    seg_list = extract_segments(segs)


    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(14, 8))

    # 全部 envelope 背景点
    ax.scatter(
        df[date_col],
        df[value_col],
        s=3,
        color="red",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        zorder=3,
        label="envelope points"
    )

    prev_end_x_num = None
    prev_end_y = None

    # 按分段画 piecewise OLS
    for i, seg in enumerate(seg_list, start=1):
        s = int(seg["start"])
        e = int(seg["end"])
        # 如果你逻辑里 end 是闭区间，可以这么调：
        if END_IS_INCLUSIVE:
            e = e + 1

        g = df.iloc[s:e]
        if g.empty:
            continue



        # segment 点（稍大一点，便于分段观察）
        ax.scatter(
            g[date_col],
            g[value_col],
            s=4,
            zorder=4,
            label=f"Seg {i}"
        )

        if len(g) >= 2:
            # 1) datetime → matplotlib 数值日期
            xd = mdates.date2num(g[date_col])
            yd = g[value_col].to_numpy()

            # 2) 一阶线性拟合
            m, b = np.polyfit(xd, yd, 1)


            # 3) 该段的拟合线 x 范围
            xfit = np.linspace(xd.min(), xd.max(), 200)
            yfit = m * xfit + b

            # 当前段的“首尾”点（数值坐标）
            seg_start_x_num = xfit[0]
            seg_start_y = yfit[0]
            seg_end_x_num = xfit[-1]
            seg_end_y = yfit[-1]

            # 4) 画该段回归线
            ax.plot(
                mdates.num2date(xfit),
                yfit,
                linewidth=2,
                color="red",
                zorder=5
            )

            # 5) 和上一段“首尾相连”
            if prev_end_x_num is not None:
                ax.plot(
                    [mdates.num2date(prev_end_x_num), mdates.num2date(seg_start_x_num)],
                    [prev_end_y, seg_start_y],
                    linewidth=2,
                    color="red",
                    zorder=5
                )

            # 更新上一段的末尾点
            prev_end_x_num = seg_end_x_num
            prev_end_y = seg_end_y

    ax.set_title("Envelope points with segments (piecewise OLS, end-to-start linked)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Complexity")
    ax.legend(ncol=2)

    # --- Convert figure to base64 PNG ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    return b64


def plot_weekly_change(
    df,
    date_col: str,
    value_col: str,
    agg="mean",
    title="Weekly Change in Complexity",
):
    """
    按周聚合 upper envelope（y_envelope），画出一张每周复杂度水平/变化的图。

    Parameters
    ----------
    df : DataFrame
        形如 [x_mid, y_envelope] 的表（或有其他列也可以）。
    date_col : str
        时间列名，默认 "x_mid"。
    value_col : str
        复杂度数值列名，默认 "y_envelope"。
    agg : {"mean", "sum", "median"}
        每周聚合方式，默认取 mean。
    title : str
        图表标题。

    Returns
    -------
    img_b64 : str or None
        base64 编码后的 PNG 字符串；如果数据不足返回 None。
    """

    if df is None or len(df) == 0:
        return None

    # 只取需要的列
    data = df[[date_col, value_col]].copy()

    # 确保时间列是 datetime
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])

    if data.empty:
        return None

    # 按时间排序并设为索引
    data = data.sort_values(date_col).set_index(date_col)

    # === 按周聚合 ===
    if agg == "sum":
        weekly = data[value_col].resample("W").sum()
    elif agg == "median":
        weekly = data[value_col].resample("W").median()
    else:  # 默认 mean
        weekly = data[value_col].resample("W").mean()

    if weekly.empty:
        return None

    # 如果你更想看“变化量”，可以改成：
    # weekly_change = weekly.diff()
    # 然后下面把 weekly 换成 weekly_change
    # 这里先画每周的平均水平

    # === 画图 ===
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.bar(weekly.index, weekly.values, width=5)  # 简单柱状图

    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel(f"Weekly {agg} of {value_col}")

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    plt.close(fig)
    buf.close()

    return img_b64


# Todo 
# def plot_temporal_variation_change(
#     df,
#     date_col="x_mid",
#     value_col="y_envelope",
#     freq="W",
#     title="Temporal Variation in Complexity (Δ Change)"
# ):
#     """ Show red if complexity increases, green if decreases """
#     if df is None or len(df)==0:
#         return None

#     data = df[[date_col, value_col]].copy().reset_index(drop=True)

#     data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
#     data = data.dropna(subset=[date_col, value_col])
    
#     data = data.sort_values(date_col)

#     # 🔥 Weekly mean — 可改 median/sum
#     weekly = data.set_index(date_col)[value_col].resample(freq).mean()
#     weekly = weekly.dropna() 

#     change = weekly.diff()  # --> 差分曲线 Δy

#     if change.empty or change.isna().all():
#         return None

#     # 🎨 红=上升（变复杂），绿=下降（变简单）
#     colors = ["red" if x > 0 else "green" for x in change]

#     # === 绘图 ===
#     fig, ax = plt.subplots(figsize=(16,8))

#     ax.bar(change.index, change.values, color=colors, width=6)

#     ax.axhline(0, color="black", linewidth=1.2)   # 基准线
#     ax.set_title(title)
#     ax.set_ylabel("Δ Complexity (week over week)")
#     ax.set_xlabel("Time (Weekly)")
#     fig.autofmt_xdate(rotation=25)
#     plt.tight_layout()

#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     img_b64 = base64.b64encode(buf.read()).decode("ascii")

#     plt.close(fig)
#     buf.close()
#     return img_b64
def plot_temporal_variation_change(
    df,
    date_col="x_mid",
    value_col="y_envelope",
    freq="W",
    title="Tech Debt Over Time",
    window=None,   # "3M","6M","1Y","3Y","5Y"... 或 None
):
    """
    左轴：复杂度变化量 |Δ|（红=变复杂，绿=变简单，所有柱子向上，且视觉更大）
    右轴：复杂度水平（weekly mean）折线
    window:
        None  -> 全部历史
        "3M"  -> 过去 3 个月
        "6M"  -> 过去 6 个月
        "1Y"  -> 过去 1 年
        "3Y"  -> 过去 3 年
        "5Y"  -> 过去 5 年
        以及任意类似格式的 "数字+M/Y"
    """
    if df is None or len(df) == 0:
        return None

    data = df[[date_col, value_col]].copy().reset_index(drop=True)

    # --- 清洗 & 排序 ---
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])
    data = data.sort_values(date_col)

    # 🔥 按周聚合得到“复杂度水平”（全量）
    weekly_full = (
        data.set_index(date_col)[value_col]
        .resample(freq)
        .mean()
        .dropna()
    )
    if weekly_full.empty or len(weekly_full) < 2:
        return None

    # ===== 通用 window 解析逻辑 =====
    weekly = weekly_full
    if window:
        w = str(window).strip().upper()  # 例如 "3Y" / "5Y" / "6M"
        last_date = weekly_full.index.max()
        cutoff = None

        try:
            if w.endswith("M"):
                n_months = int(w[:-1])
                cutoff = last_date - pd.DateOffset(months=n_months)
            elif w.endswith("Y"):
                n_years = int(w[:-1])
                cutoff = last_date - pd.DateOffset(years=n_years)
        except ValueError:
            cutoff = None  # 解析失败就当没填 window

        if cutoff is not None:
            weekly_window = weekly_full[weekly_full.index >= cutoff]
            # 如果数据太少（比如不足 5 周），就自动 fallback 回全量
            if len(weekly_window) >= 5:
                weekly = weekly_window
            else:
                weekly = weekly_full  # 回退到 ALL

    # Δ complexity（周对周变化）
    change = weekly.diff().dropna()
    if change.empty:
        return None

    # 高度用绝对值，颜色用正负
    abs_change = change.abs()
    colors = ["red" if x > 0 else "green" for x in change.values]

    abs_vals = abs_change.values

    # ====== 用分位数控制 y 轴上限，让大部分柱子「长高」 ======
    cap = np.percentile(abs_vals, 98)
    cap = max(cap, np.max(abs_vals) * 0.25, 1e-6)

    # 真正画出来的高度（超过 cap 的直接截断到 cap）
    bar_heights = np.minimum(abs_vals, cap)

    # ========= 绘图部分 =========
    fig, ax = plt.subplots(figsize=(36, 16))

    # --- 左轴：变化量柱子（全部向上，放大+加粗）---
    ax.bar(
        change.index,
        bar_heights,
        color=colors,
        width=10,
        alpha=0.9,
        label="|Δ Tech Debt| (Week over Week)",
    )

    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_lim = cap * 1.15
    ax.set_ylim(0, y_lim)
    ax.set_ylabel("|Δ Complexity| (Weekly Change)")

    # --- 对于超过 cap 的极端值，单独用竖线 + 标注表示 ---
    outlier_mask = abs_vals > cap
    if outlier_mask.any():
        for x, real_y, sign in zip(
            change.index[outlier_mask],
            change.values[outlier_mask],
            np.sign(change.values[outlier_mask]),
        ):
            ax.vlines(
                x,
                0,
                y_lim,  # 拉到顶
                color="green" if sign < 0 else "red",
                linewidth=2.0,
                alpha=0.9,
            )
            ax.annotate(
                f"{real_y:.0f}",
                xy=(x, y_lim),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.8,
            )

    # --- 右轴：复杂度水平折线 ---
    ax2 = ax.twinx()

    aligned_weekly = weekly.loc[change.index]
    ax2.plot(
        aligned_weekly.index,
        aligned_weekly.values,
        color="#1f77b4",
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        alpha=0.92,
        label="Weekly Complexity Level",
    )
    ax2.set_ylabel("Weekly Complexity Level")
    ax2.set_ylim(0, aligned_weekly.max() * 1.25)

    # --- 时间轴格式 ---
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    # --- 合并图例 ---
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    # 标题 & 轴标签
    if window:
        ax.set_title(f"{title} · Range: {window}", fontsize=14, fontweight="bold")
    else:
        ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xlabel("Time (Weekly)")

    plt.tight_layout()

    # ========= 输出为 base64 =========
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    plt.close(fig)
    buf.close()
    return img_b64



def plot_accumulative_complexity_multi(
    series_dict,
    date_col: str = "x_mid",
    value_col: str = "y_envelope",
    title: str = "Project Comparison – Accumulative Complexity",
    freq: str = "W",
):
    """
    series_dict: { label -> highlight_df }，每个 df 至少包含 [date_col, value_col]
    用于多个项目的累积复杂度对比。
    """
    if not series_dict:
        return None

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    for label, df in series_dict.items():
        if df is None or df.empty:
            continue

        local = df[[date_col, value_col]].dropna().copy()
        local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
        local = local.dropna(subset=[date_col, value_col]).sort_values(date_col)

        if local.empty:
            continue

        agg = (
            local.set_index(date_col)[value_col]
            .resample(freq)
            .mean()
            .dropna()
            .to_frame()
            .reset_index()
        )
        if agg.empty:
            continue

        agg["accumulative_complexity"] = agg[value_col].cumsum()

        ax.plot(
            agg[date_col],
            agg["accumulative_complexity"],
            linewidth=2,
            marker="o",
            label=str(label),
        )

    if not ax.lines:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Accumulative Complexity")
    ax.grid(alpha=0.6)
    ax.legend(title="Project", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _prepare_time_series(df, date_col="x_mid", value_col="y_envelope"):
    """清理 + 按时间排序的工具函数"""
    local = df[[date_col, value_col]].dropna().copy()
    local[date_col] = pd.to_datetime(local[date_col], errors="coerce")
    local = local.dropna(subset=[date_col, value_col]).sort_values(date_col)
    return local
def plot_envelope_multi(
    series_dict,
    date_col: str = "x_mid",
    value_col: str = "y_envelope",
    title: str = "Project Comparison – Complexity Envelope Segments",
    normalize: bool = False,
    align_start: bool = False,
):
    if not series_dict:
        return None

    fig, ax = plt.subplots(figsize=(12, 8), dpi=220)  # 🔥 更大 + 更清晰
    # fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.09)  # 🔥 去白边，放大画面


    for label, df in series_dict.items():
        local = _prepare_time_series(df, date_col, value_col)
        if local.empty:
            continue

        y = local[value_col].astype(float)

        if normalize:
            ymin, ymax = y.min(), y.max()
            if np.isclose(ymax, ymin):
                y_plot = np.zeros_like(y, dtype=float)
            else:
                y_plot = (y - ymin) / (ymax - ymin)
        else:
            y_plot = y

        if align_start:
            x = np.arange(len(local))
        else:
            x = local[date_col]

        ax.plot(
            x,
            y_plot,
            linewidth=1.8,
            marker="o",
            label=str(label),
        )

    if not ax.lines:
        plt.close(fig)
        return None

    if normalize:
        ax.set_ylabel("Normalized Envelope Complexity (0–1)")
    else:
        ax.set_ylabel("Envelope Complexity")

    if align_start:
        ax.set_xlabel("Steps since project start")
        ax.set_title(title + " — Aligned at Start" + ( " [Normalized]" if normalize else "" ))
    else:
        ax.set_xlabel("Time")
        ax.set_title(title)

    ax.grid(alpha=0.6)
    ax.legend(title="Project", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_timeline_multi(
    series_dict,
    date_col: str = "x_mid",
    value_col: str = "y_envelope",
    title: str = "Project Comparison – Complexity Timeline",
    freq: str = "W",
    normalize: bool = False,
    align_start: bool = False,   # ← 新增
):
    if not series_dict:
        return None

    fig, ax = plt.subplots(figsize=(12, 8), dpi=220)  # 🔥 更大 + 更清晰
    # fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.09)  # 🔥 去白边，放大画面


    for label, df in series_dict.items():
        local = _prepare_time_series(df, date_col, value_col)
        if local.empty:
            continue

        ts = (
            local.set_index(date_col)[value_col]
            .resample(freq)
            .mean()
            .dropna()
            .to_frame()
            .reset_index()
        )
        if ts.empty:
            continue

        y = ts[value_col].astype(float)

        # normalize
        if normalize:
            ymin, ymax = y.min(), y.max()
            if np.isclose(ymax, ymin):
                y_plot = np.zeros_like(y, dtype=float)
            else:
                y_plot = (y - ymin) / (ymax - ymin)
        else:
            y_plot = y

        # 🔥 对齐起点：X 改为 0,1,2,...
        if align_start:
            x = np.arange(len(ts))
        else:
            x = ts[date_col]

        ax.plot(
            x,
            y_plot,
            linewidth=1.8,
            marker="o",
            label=str(label),
        )

    if not ax.lines:
        plt.close(fig)
        return None

    if normalize:
        ax.set_ylabel("Normalized Complexity (0–1)")
    else:
        ax.set_ylabel("Average Complexity")

    if align_start:
        ax.set_xlabel(f"Periods since project start ({freq})")
        ax.set_title(title + " — Aligned at Start" + ( " [Normalized]" if normalize else "" ))
    else:
        ax.set_xlabel("Time")
        ax.set_title(title + f" ({freq})")

    ax.grid(alpha=0.6)
    ax.legend(title="Project", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def plot_accumulative_complexity_multi(
    series_dict,
    date_col: str = "x_mid",
    value_col: str = "y_envelope",
    title: str = "Project Comparison – Accumulative Complexity",
    freq: str = "W",
    normalize: bool = False,
    align_start: bool = False,
):
    if not series_dict:
        return None

    fig, ax = plt.subplots(figsize=(12, 8), dpi=220)  # 🔥 更大 + 更清晰
    # fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.09)  # 🔥 去白边，放大画面


    for label, df in series_dict.items():
        local = _prepare_time_series(df, date_col, value_col)
        if local.empty:
            continue

        agg = (
            local.set_index(date_col)[value_col]
            .resample(freq)
            .mean()
            .dropna()
            .to_frame()
            .reset_index()
        )
        if agg.empty:
            continue

        agg["accumulative_complexity"] = agg[value_col].cumsum()
        y = agg["accumulative_complexity"].astype(float)

        if normalize:
            ymin, ymax = float(y.min()), float(y.max())
            if np.isclose(ymax, ymin):
                y_plot = np.zeros_like(y, dtype=float)
            else:
                y_plot = (y - ymin) / (ymax - ymin)
        else:
            y_plot = y

        # 🔥 X 按起点对齐 / 使用真实时间
        if align_start:
            x = np.arange(len(agg))
        else:
            x = agg[date_col]

        ax.plot(
            x,
            y_plot,
            linewidth=2,
            marker="o",
            label=str(label),
        )

    if not ax.lines:
        plt.close(fig)
        return None

    if normalize:
        ax.set_ylabel("Normalized Accumulative Complexity (0–1)")
    else:
        ax.set_ylabel("Accumulative Complexity")

    if align_start:
        ax.set_xlabel(f"Periods since project start ({freq})")
        ax.set_title(title + " — Aligned at Start" + ( " [Normalized]" if normalize else "" ))
    else:
        ax.set_xlabel("Time")
        ax.set_title(title + f" ({freq})")

    ax.grid(alpha=0.6)
    ax.legend(title="Project", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def plot_temporal_variation_multi(
    series_dict,
    date_col: str = "x_mid",
    value_col: str = "y_envelope",
    title: str = "Project Comparison – Temporal Variation in Complexity",
    freq: str = "W",
    normalize: bool = False,
    align_start: bool = False,
):
    if not series_dict:
        return None

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    for label, df in series_dict.items():
        local = _prepare_time_series(df, date_col, value_col)
        if local.empty:
            continue

        weekly = (
            local.set_index(date_col)[value_col]
            .resample(freq)
            .mean()
            .dropna()
        )
        change = weekly.diff().dropna()
        if change.empty:
            continue

        y = change.values.astype(float)

        if normalize:
            ymin, ymax = y.min(), y.max()
            if np.isclose(ymax, ymin):
                y_norm = np.zeros_like(y, dtype=float)
            else:
                # 标准化到 [-1,1] 区间
                denom = max(abs(ymax), abs(ymin))
                y_norm = y / denom
            y_plot = y_norm
        else:
            y_plot = y

        # 🔥 X 对齐
        if align_start:
            x = np.arange(len(change))
        else:
            x = change.index

        ax.plot(
            x,
            y_plot,
            linewidth=1.8,
            marker="o",
            label=str(label),
        )

    if not ax.lines:
        plt.close(fig)
        return None

    if normalize:
        ax.set_ylabel("Normalized Δ Complexity (−1 ~ +1)")
    else:
        ax.set_ylabel("Δ Complexity")

    if align_start:
        ax.set_xlabel(f"Periods since project start ({freq})")
        ax.set_title(title + " — Aligned at Start" + ( " [Normalized]" if normalize else "" ))
    else:
        ax.set_xlabel("Time")
        ax.set_title(title + f" ({freq} Δ)")

    ax.axhline(0, color="black", linewidth=1.0)
    ax.grid(alpha=0.6)
    ax.legend(title="Project", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")



def plot_raw_complexity(
    df,
    date_col: str = "Datetime",
    value_col: str = "complexity_raw",
    title: str = "Raw Complexity over Time",
):
    """
    直接画原始数据：Datetime vs complexity_raw
    不做 resample，适合作为数据样本展示。
    """
    if df is None or df.empty:
        return None

    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    # 用细线 + 小点表现 raw data
    ax.plot(
        data[date_col],
        data[value_col],
        linewidth=0.8,
        marker=".",
        markersize=2,
    )

    ax.set_title(title)
    ax.set_xlabel("Time (raw commit timeline)")
    ax.set_ylabel("Raw Complexity")
    ax.grid(alpha=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")