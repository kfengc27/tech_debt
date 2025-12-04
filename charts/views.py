from django.shortcuts import render

# Create your views here.
import os
from datetime import datetime
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadCSVForm
import glob
from django.utils.html import escape
from .calculate import compute_datetime_upper_envelope, plot_accumulative_complexity, detect_date_col, detect_value_col, plot_time_vs_complexity, plot_weekly_change, plot_temporal_variation_change, plot_accumulative_complexity_multi

from .calculate import plot_segments, plot_temporal_variation_multi,plot_envelope_multi,plot_timeline_multi
UPLOADS_DIR = os.path.join(settings.MEDIA_ROOT, "uploads")


def _read_csv_preview(path):
    """Read top 10 rows with robust encoding fallbacks."""
    try:
        return pd.read_csv(path, nrows=10)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=10, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, nrows=10, encoding="latin-1")

def upload_csv_view(request):
    preview_html = None
    saved_name = None
    saved_url = None
    err = None

    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]
            try:
                # Try utf-8 first, then fallback to latin-1 if needed
                try:
                    # Save to MEDIA_ROOT/uploads/ with automatic unique name if exists
                    uploads_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)

                    fs = FileSystemStorage(location=uploads_dir, base_url=settings.MEDIA_URL + "uploads/")
                    saved_name = fs.save(f.name, f)  # e.g., "mydata.csv" or "mydata_1.csv"
                    saved_path = fs.path(saved_name)  # absolute path on disk
                    saved_url = fs.url(saved_name)  # URL for downloading in dev

                    # Build top-10 preview without loading the whole file
                    df_head = _read_csv_preview(saved_path)
                    preview_html = df_head.to_html(classes="table table-compact", index=False, border=0)

                except UnicodeDecodeError:
                    f.seek(0)
                    df = pd.read_csv(f, encoding="latin-1")
            except Exception as e:
                err = str(e)
        else:
            err = "Invalid form submission."
    else:
        form = UploadCSVForm()

    return render(
        request,
        "upload_plot.html",
        {
            "form": form,
            "preview_html": preview_html,   # render |safe in template
            "saved_name": saved_name,       # file name on disk (may be auto-suffixed)
            "saved_url": saved_url,         # link to download (works when DEBUG=True)
            "error": err,
        }
    )

def _safe_read_csv(path):
    """Read CSV with common encodings and fallback."""
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception:
            continue
    raise ValueError(f"Unable to read CSV: {path}")


def _preview_top5_html(path):
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, nrows=5, encoding=enc) if enc else pd.read_csv(path, nrows=5)
            return df.to_html(classes="table table-compact", index=False, border=0)
        except Exception:
            continue
    return "<div class='error'>Unable to preview this CSV.</div>"
def chart_view(request):
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # ============ 1) 处理 upload / include 参数 ============
    include = request.GET.get("include") or request.session.get("last_upload_name")

    if request.method == "POST" and request.FILES.get("file"):
        f = request.FILES["file"]
        save_path = os.path.join(UPLOADS_DIR, f.name)
        with open(save_path, "wb+") as dest:
            for chunk in f.chunks(): dest.write(chunk)
        include = f.name
        request.session["last_upload_name"] = f.name   # 🔥 记住最新上传文件名

    # ========= 2) 生成 Recent 项目 tiles (≤6) ==============
    all_csv = sorted(
        glob.glob(os.path.join(UPLOADS_DIR, "*.csv")),
        key=os.path.getmtime,
        reverse=True
    )[:6]  # 🔥最多6个，自动铺满UI

    tiles = [{"name": os.path.basename(p)} for p in all_csv]
    for t in tiles: t["shortname"] = t["name"].split("_")[0]

    # ============= 3) 如果未选项目 → 默认第一个 =============
    if not include and tiles:
        include = tiles[0]["name"]

    # ============= 4) 仅渲染 include 文件的 Complexity 图 ============
    segment_64 = time_complexity_b64 = acc_chart_b64 = temporal_change_b64 = None

    if include:
        path = os.path.join(UPLOADS_DIR, include)
        df = _safe_read_csv(path)

        res = compute_datetime_upper_envelope(df["Datetime"].values, df["complexity_raw"].values)
        highlight_df = res["highlight_df"]

        date_col = detect_date_col(highlight_df)
        value_col = detect_value_col(highlight_df, exclude=[date_col])

        segment_64 = plot_segments(highlight_df, date_col="x_mid", value_col=value_col)
        time_complexity_b64,_,_ = plot_time_vs_complexity(highlight_df, date_col, value_col, freq="W")
        acc_chart_b64 = plot_accumulative_complexity(highlight_df, value_col)
        temporal_change_b64 = plot_temporal_variation_change(highlight_df,"x_mid","y_envelope")

    # ============= 5) 传入页面 =============
    return render(request,"chart_view.html",{
        "filename": include,
        "tiles": tiles,
        "segment_64": segment_64,
        "time_complexity_b64": time_complexity_b64,
        "acc_chart_b64": acc_chart_b64,
        "temporal_change_b64": temporal_change_b64,
    })


def compare_view(request):
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # 所有可用项目
    all_csv = sorted(
        glob.glob(os.path.join(UPLOADS_DIR, "*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )
    all_files = [os.path.basename(p) for p in all_csv]

    # 读取用户选中的项目 + 对比类型
    selected_files = request.GET.getlist("files")
    chart_type = request.GET.get("metric", "accumulative")  # envelope / timeline / accumulative / temporal

    if not selected_files:
        selected_files = all_files[:2]  # 默认选最近两个

    series_dict = {}
    errors = []

    for name in selected_files:
        path = os.path.join(UPLOADS_DIR, name)
        if not os.path.isfile(path):
            errors.append({"name": name, "error": "File not found"})
            continue
        try:
            df = _safe_read_csv(path)

            res = compute_datetime_upper_envelope(
                df["Datetime"].values,
                df["complexity_raw"].values,
            )
            highlight_df = res["highlight_df"]

            series_dict[name] = highlight_df

        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    comparison_b64 = None
    if len(series_dict) >= 2:
        if chart_type == "envelope":
            comparison_b64 = plot_envelope_multi(series_dict)
        elif chart_type == "timeline":
            comparison_b64 = plot_timeline_multi(series_dict)
        elif chart_type == "temporal":
            comparison_b64 = plot_temporal_variation_multi(series_dict)
        else:  # 默认用累积复杂度
            comparison_b64 = plot_accumulative_complexity_multi(series_dict)

    context = {
        "all_files": all_files,
        "selected_files": selected_files,
        "comparison_b64": comparison_b64,
        "errors": errors,
        "chart_type": chart_type,
    }
    return render(request, "compare_view.html", context)

def projects_view(request):
    """List all existing uploaded projects (CSV files)."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    all_csv = sorted(
        glob.glob(os.path.join(UPLOADS_DIR, "*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )

    projects = []
    for p in all_csv:
        name = os.path.basename(p)
        size_bytes = os.path.getsize(p)
        mtime = datetime.fromtimestamp(os.path.getmtime(p))

        projects.append({
            "name": name,
            "shortname": name.split("_")[0],
            "size_kb": round(size_bytes / 1024, 1),
            "mtime": mtime,
        })

    context = {
        "projects": projects,
        "total": len(projects),
    }
    return render(request, "projects_list.html", context)