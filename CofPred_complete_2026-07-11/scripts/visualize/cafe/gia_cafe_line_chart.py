# -*- coding: utf-8 -*-
"""
================================================================================
ve_bieu_do.py  -  Ve lai 3 bieu do tu phan tich gia ca phe Robusta noi dia
================================================================================
  1. spread_stats_compare.png          : heatmap P95 chenh lech tung cap tinh
  2. coverage_timeline.png              : timeline do phu - moi tinh 1 dai, dut doan = thieu data
  3. eda_overview.png                   : phan bo gia theo tinh (boxplot)
     eda_target_decision.png            : so quan sat Nam x Tinh (heatmap)
  4. gia_Ca_phe_Robusta_nhan_xo_theo_tinh.png : line chart dien bien gia theo tinh

Chay:
  python3 ve_bieu_do.py --input gia_cafe.csv --outdir charts_out
================================================================================
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Cau hinh chung
# --------------------------------------------------------------------------- #
DAKLAK, LAMDONG = "\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng"
GIALAI, DAKNONG, KONTUM = "Gia Lai", "\u0110\u1eafk N\u00f4ng", "Kon Tum"
TAY_NGUYEN = [DAKLAK, LAMDONG, GIALAI, DAKNONG, KONTUM]
ALL_PROV = TAY_NGUYEN + ["Qu\u1ea3ng Ng\u00e3i", "H\u1ed3 Ch\u00ed Minh"]
ANCHOR = [DAKLAK, LAMDONG]            # 2 tinh tru cot
SPARSE = [GIALAI, DAKNONG, KONTUM]    # 3 tinh thua
PRICE_TYPE = "Th\u01b0\u01a1ng l\u00e1i thu mua"

CMAP = {DAKLAK: "#c0392b", LAMDONG: "#2980b9", GIALAI: "#27ae60",
        DAKNONG: "#e67e22", KONTUM: "#8e44ad",
        "Qu\u1ea3ng Ng\u00e3i": "#16a085", "H\u1ed3 Ch\u00ed Minh": "#7f8c8d"}

plt.rcParams["font.family"] = "DejaVu Sans"   # font ho tro tieng Viet
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def kfmt(x, _):
    """Dinh dang truc tien: 100000 -> 100k"""
    return f"{int(x/1000)}k"


# --------------------------------------------------------------------------- #
# Doc & chuan bi du lieu
# --------------------------------------------------------------------------- #
def load(path):
    df = pd.read_csv(path)
    df["Ngay"] = pd.to_datetime(df["Ngay"], errors="coerce")
    df["Gia"] = pd.to_numeric(df["Gia"], errors="coerce")
    for c in ["Thi_truong", "Loai_gia", "Don_vi"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def build_pivot(df, business_day=True):
    """Pivot gia 'Thuong lai thu mua' theo ngay x tinh (Tay Nguyen)."""
    tl = df[df["Loai_gia"] == PRICE_TYPE]
    g = (tl[tl["Thi_truong"].isin(TAY_NGUYEN)]
         .groupby(["Thi_truong", "Ngay"], as_index=False)["Gia"].mean())
    p = g.pivot(index="Ngay", columns="Thi_truong", values="Gia").sort_index()
    if business_day:
        p = p.reindex(pd.bdate_range(p.index.min(), p.index.max()))
    return tl, p


# =========================================================================== #
# BIEU DO 1 - spread_stats_compare.png
# =========================================================================== #
def chart_spread_stats(p, outdir):
    n = len(TAY_NGUYEN)
    P95 = np.full((n, n), np.nan)
    for i, a in enumerate(TAY_NGUYEN):
        for j, b in enumerate(TAY_NGUYEN):
            d = (p[a] - p[b]).dropna()
            if len(d) == 0:
                continue
            P95[i, j] = d.abs().quantile(0.95)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("ch\u00eanh l\u1ec7ch gi\u1eefa c\u00e1c c\u1eb7p t\u1ec9nh", fontsize=13, fontweight="bold")
    im = ax.imshow(P95, cmap="OrRd", vmin=0, vmax=2500)
    ax.set_xticks(range(n)); ax.set_xticklabels(TAY_NGUYEN, rotation=45, fontsize=8, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(TAY_NGUYEN, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(P95[i, j]):
                ax.text(j, i, f"{P95[i, j]:.0f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/spread_stats_compare.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("[saved] spread_stats_compare.png")


# =========================================================================== #
# BIEU DO 2 - coverage_timeline.png
# =========================================================================== #
def chart_coverage_timeline(df, tl, outdir):
    fig, ax = plt.subplots(figsize=(15, 5.2))
    prov_order = [LAMDONG, DAKLAK, GIALAI, DAKNONG, KONTUM, "Qu\u1ea3ng Ng\u00e3i", "H\u1ed3 Ch\u00ed Minh"]
    for i, m in enumerate(prov_order):
        sub = tl[tl["Thi_truong"] == m].drop_duplicates("Ngay").sort_values("Ngay")
        if len(sub) == 0:
            continue
        days = sub["Ngay"].values
        # moi ngay co gia = 1 vach doc; cho thieu data -> khong co vach (dut doan)
        ax.scatter(days, [i] * len(days), s=6, marker="|", color=CMAP.get(m, "#333"), alpha=0.8)
        ax.text(df["Ngay"].max() + pd.Timedelta(days=25), i, f"{len(sub)} ng",
                va="center", fontsize=9, fontweight="bold", color=CMAP.get(m, "#333"))
    ax.set_yticks(range(len(prov_order)))
    ax.set_yticklabels(prov_order)
    ax.set_ylim(-0.6, len(prov_order) - 0.4)
    ax.set_title("Timeline \u0111\u1ed9 ph\u1ee7 d\u1eef li\u1ec7u theo t\u1ec9nh (m\u1ed7i v\u1ea1ch = 1 ng\u00e0y c\u00f3 gi\u00e1)",
                 fontsize=13, fontweight="bold")
    ax.axhspan(-0.6, 1.5, color="#2ecc71", alpha=0.07)
    ax.text(df["Ngay"].min(), 1.65, "2 t\u1ec9nh tr\u1ee5 c\u1ed9t (anchor) \u2014 ph\u1ee7 ~96%",
            fontsize=9, color="#27ae60", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{outdir}/coverage_timeline.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("[saved] coverage_timeline.png")


# =========================================================================== #
# BIEU DO 3 - EDA gia ca phe (overview + target decision)
# =========================================================================== #
def chart_eda(df, p, outdir):
    # --- 3a. Phan bo gia theo tinh (boxplot) ---
    mk = df.groupby("Thi_truong")["Gia"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("EDA gi\u00e1 c\u00e0 ph\u00ea \u2014 Ph\u00e2n b\u1ed1 gi\u00e1 theo t\u1ec9nh", fontsize=14, fontweight="bold")
    bp = ax.boxplot([df[df["Thi_truong"] == m]["Gia"] for m in mk],
                    vert=False, tick_labels=mk, patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor("#2980b9"); b.set_alpha(0.7)
    ax.set_title("Ph\u00e2n b\u1ed1 gi\u00e1 theo t\u1ec9nh", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/eda_overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[saved] eda_overview.png")

    # --- 3b. So quan sat Nam x Tinh (heatmap) ---
    piv = df.pivot_table(index="Thi_truong", columns=df["Ngay"].dt.year,
                         values="Gia", aggfunc="count", fill_value=0).reindex(ALL_PROV)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("EDA gi\u00e1 c\u00e0 ph\u00ea \u2014 \u0110\u1ed9 ph\u1ee7 d\u1eef li\u1ec7u", fontsize=14, fontweight="bold")
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=9)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, int(piv.values[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("S\u1ed1 quan s\u00e1t N\u0103m x T\u1ec9nh", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/eda_target_decision.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[saved] eda_target_decision.png")


# =========================================================================== #
# BIEU DO 4 - gia_Ca_phe_Robusta_nhan_xo_theo_tinh.png
# =========================================================================== #
def chart_robusta_line(df, outdir):
    """Line chart dien bien gia Ca phe Robusta nhan xo trung binh theo tinh."""
    import matplotlib.dates as mdates

    # Lay tat ca cac tinh co trong du lieu
    all_provs = sorted(df["Thi_truong"].dropna().unique().tolist())

    # Tinh trung binh gia (tat ca loai gia) theo tinh va ngay
    g = (df.groupby(["Thi_truong", "Ngay"], as_index=False)["Gia"].mean())
    pivot = g.pivot(index="Ngay", columns="Thi_truong", values="Gia").sort_index()

    # Mau cho cac tinh (dung CMAP neu co, fallback sang tab10)
    import matplotlib
    tab10 = matplotlib.colormaps["tab10"].colors
    color_cycle = {}
    tab_i = 0
    for prov in all_provs:
        if prov in CMAP:
            color_cycle[prov] = CMAP[prov]
        else:
            color_cycle[prov] = tab10[tab_i % len(tab10)]
            tab_i += 1

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for prov in all_provs:
        if prov not in pivot.columns:
            continue
        series = pivot[prov]          # GIU NAN - matplotlib tu ngat duong tai cho thieu data
        if series.notna().sum() == 0:
            continue
        color = color_cycle[prov]
        ax.plot(series.index, series.values, lw=1.2, label=prov,
                color=color, alpha=0.9)
        # Nhan cuoi duong: dung last_valid_index() de lay diem cuoi co data thuc
        last_date = series.last_valid_index()
        last_val  = series[last_date]
        ax.annotate(
            f"{prov}\n{last_val/1000:.1f}K",
            xy=(last_date, last_val),
            xytext=(6, 0), textcoords="offset points",
            va="center", ha="left", fontsize=7, color=color,
            fontweight="bold"
        )

    ax.set_title(
        "Gi\u00e1 C\u00e0 ph\u00ea Robusta nh\u00e2n x\u00f4 theo t\u1ec9nh (trung b\u00ecnh c\u00e1c ngu\u1ed3n)",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.set_xlabel("Th\u1eddi gian", fontsize=11)
    ax.set_ylabel("Gi\u00e1 (VN\u0110/Kg)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Legend gon o goc tren trai
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8,
              ncol=2, title="T\u1ec9nh / Th\u00e0nh ph\u1ed1", title_fontsize=8)

    # Chu thich nguon
    ax.text(0.01, 0.02,
            "* Gi\u00e1 trung b\u00ecnh c\u1ee7a t\u1ea5t c\u1ea3 ngu\u1ed3n thu mua",
            transform=ax.transAxes, fontsize=8, color="gray", style="italic")

    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout(rect=[0, 0, 0.92, 1])   # danh cho nhan cuoi duong ben phai
    out_name = "gia_Ca_phe_Robusta_nhan_xo_theo_tinh.png"
    fig.savefig(f"{outdir}/{out_name}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Processing/coffe/gia_cafe.csv")
    ap.add_argument("--outdir", default="reports/figures/cafe/cafe_main")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.input)
    tl, p = build_pivot(df)

    chart_spread_stats(p, args.outdir)           # 1
    chart_coverage_timeline(df, tl, args.outdir)  # 2
    chart_eda(df, p, args.outdir)                 # 3
    chart_robusta_line(df, args.outdir)            # 4
    print("\n*** XONG: 5 file PNG trong thu muc", args.outdir, "***")


if __name__ == "__main__":
    main()
