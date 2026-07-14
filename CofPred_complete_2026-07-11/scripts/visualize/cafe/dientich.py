# -*- coding: utf-8 -*-
"""
Xu ly + truc quan hoa DIEN TICH ca phe theo tinh (don vi: nghin ha), nam 1995-2024.
Chay:  python3 20_dientich.py --input dientich_cafe.csv --outdir dientich_out
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TN5 = ["\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng", "Gia Lai", "\u0110\u1eafk N\u00f4ng", "Kon Tum"]
CMAP = {
    "\u0110\u1eafk L\u1eafk": "#c0392b", "L\u00e2m \u0110\u1ed3ng": "#2980b9",
    "Gia Lai": "#27ae60", "\u0110\u1eafk N\u00f4ng": "#e67e22", "Kon Tum": "#8e44ad",
}


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Nam"})
    df["Nam"] = df["Nam"].astype(int)
    return df


def clean(df):
    """Lam sach: sua loi trung cot Lam Dong == Dak Lak (loi nhap lieu)."""
    out = df.copy()
    prov = [c for c in out.columns if c != "Nam"]
    # 1) Lam Dong giong het Dak Lak (va gia tri qua cao so voi LD that ~175) => set NaN
    dup_mask = (out["L\u00e2m \u0110\u1ed3ng"] == out["\u0110\u1eafk L\u1eafk"]) & (out["L\u00e2m \u0110\u1ed3ng"] > 130)
    out.loc[dup_mask, "L\u00e2m \u0110\u1ed3ng"] = np.nan
    # 2) cot tong & ty trong
    out["TN5"] = out[TN5].sum(axis=1, min_count=1)
    out["Total"] = out[prov].sum(axis=1, min_count=1)
    out["TN_share"] = out["TN5"] / out["Total"] * 100
    return out, prov, dup_mask


def chart_overview(df, prov, dup_mask, outdir):
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Tong dien tich ca nuoc theo nam
    a = ax[0, 0]
    a.plot(df["Nam"], df["Total"], color="#2c3e50", lw=2.2, marker="o", ms=3)
    a.fill_between(df["Nam"], df["Total"], color="#2c3e50", alpha=0.08)
    a.set_title("Tong dien tich ca phe ca nuoc (nghin ha)", fontweight="bold")
    a.set_xlabel("Nam"); a.set_ylabel("nghin ha"); a.grid(alpha=0.3)

    # (2) 5 tinh Tay Nguyen
    a = ax[0, 1]
    for p in TN5:
        a.plot(df["Nam"], df[p], label=p, color=CMAP[p], lw=2, marker="o", ms=2.5)
    # danh dau vung Lam Dong bi sua (NaN)
    if dup_mask.any():
        yrs = df.loc[dup_mask, "Nam"]
        a.axvspan(yrs.min(), yrs.max(), color="#bdc3c7", alpha=0.25,
                  label="L\u00e2m \u0110\u1ed3ng loi (da loai)")
    a.set_title("Dien tich 5 tinh Tay Nguyen (nghin ha)", fontweight="bold")
    a.set_xlabel("Nam"); a.set_ylabel("nghin ha"); a.legend(fontsize=8); a.grid(alpha=0.3)

    # (3) Ty trong Tay Nguyen / ca nuoc
    a = ax[1, 0]
    a.plot(df["Nam"], df["TN_share"], color="#16a085", lw=2.2, marker="o", ms=3)
    a.fill_between(df["Nam"], df["TN_share"], color="#16a085", alpha=0.1)
    a.set_title("Ty trong dien tich 5 tinh Tay Nguyen / ca nuoc (%)", fontweight="bold")
    a.set_xlabel("Nam"); a.set_ylabel("%"); a.grid(alpha=0.3)
    a.set_ylim(0, max(60, df["TN_share"].max() + 5))

    # (4) Xep hang top tinh nam moi nhat
    a = ax[1, 1]
    last = df.iloc[-1]
    s = last[prov].dropna().sort_values(ascending=True).tail(10)
    colors = [CMAP.get(p, "#95a5a6") for p in s.index]
    a.barh(range(len(s)), s.values, color=colors)
    a.set_yticks(range(len(s))); a.set_yticklabels(s.index, fontsize=8)
    for i, v in enumerate(s.values):
        a.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
    a.set_title(f"Top 10 tinh theo dien tich nam {int(last['Nam'])} (nghin ha)", fontweight="bold")
    a.set_xlabel("nghin ha"); a.grid(alpha=0.3, axis="x")

    fig.suptitle("EDA DIEN TICH CA PHE THEO TINH (1995-2024)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, "dientich_overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def chart_heatmap(df, outdir):
    """Heatmap do phu du lieu: tinh x nam, to dam = co data."""
    prov_sorted = df[[c for c in df.columns if c not in ("Nam","TN5","Total","TN_share")]]
    order = prov_sorted.notna().sum().sort_values(ascending=False).index
    M = df.set_index("Nam")[order].T
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(M.notna().astype(int), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    ax.set_xticks(range(0, len(M.columns), 2))
    ax.set_xticklabels(M.columns[::2], rotation=90, fontsize=7)
    ax.set_title("Do phu du lieu dien tich theo tinh x nam (xanh = co data)", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "dientich_coverage.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Processing/coffe/dientich_cafe.csv")
    ap.add_argument("--outdir", default="reports/figures/cafe/dientich")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df0 = load(args.input)
    df, prov, dup_mask = clean(df0)

    # Xuat file sach dang LONG (tien dung lam feature)
    long = df.melt(id_vars=["Nam"], value_vars=prov, var_name="Tinh", value_name="DienTich_nghinHa")
    long = long.dropna(subset=["DienTich_nghinHa"]).sort_values(["Tinh", "Nam"])
    long.to_csv(os.path.join(args.outdir, "dientich_long_clean.csv"), index=False)
    # Xuat bang TayNguyen theo nam
    df[["Nam"] + TN5 + ["TN5", "Total", "TN_share"]].to_csv(
        os.path.join(args.outdir, "dientich_taynguyen_yearly.csv"), index=False)

    p1 = chart_overview(df, prov, dup_mask, args.outdir)
    p2 = chart_heatmap(df, args.outdir)

    print("Da sua", int(dup_mask.sum()), "nam Lam Dong bi trung Dak Lak")
    print("TN share 2024: %.1f%%" % df.loc[df.Nam == 2024, "TN_share"].values[0])
    print("Saved:", p1, p2)


if __name__ == "__main__":
    main()
