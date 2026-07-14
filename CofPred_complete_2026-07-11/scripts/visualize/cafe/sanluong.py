# -*- coding: utf-8 -*-
"""
EDA SAN LUONG ca phe theo tinh (don vi: tan), nam 2005-2024.
Chay: python3 24_eda_sanluong.py --input sanluong_cafe.csv --outdir sanluong_out
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
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


def chart_overview(df, prov, outdir):
    df = df.copy()
    df["TN5"] = df[TN5].sum(axis=1, min_count=1)
    df["Total"] = df[prov].sum(axis=1, min_count=1)
    df["share"] = df["TN5"] / df["Total"] * 100

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Tong san luong ca nuoc (quy nghin tan cho de doc)
    a = ax[0, 0]
    a.plot(df["Nam"], df["Total"] / 1000, color="#2c3e50", lw=2.2, marker="o", ms=3)
    a.fill_between(df["Nam"], df["Total"] / 1000, color="#2c3e50", alpha=0.08)
    a.set_title("Tong san luong ca phe ca nuoc (nghin tan)", fontweight="bold")
    a.set_xlabel("Nam"); a.set_ylabel("nghin tan"); a.grid(alpha=0.3)
    years = df["Nam"].values
    a.set_xticks(years[::2]); a.set_xticklabels(years[::2], rotation=45, ha="right", fontsize=8)

    # (2) 5 tinh Tay Nguyen
    a = ax[0, 1]
    for p in TN5:
        a.plot(df["Nam"], df[p] / 1000, label=p, color=CMAP[p], lw=2, marker="o", ms=2.5)
    # danh dau dut gay Kon Tum 2022

    a.set_title("San luong 5 tinh Tay Nguyen (nghin tan)", fontweight="bold")
    a.set_xlabel("Nam"); a.set_ylabel("nghin tan"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_xticks(years[::2]); a.set_xticklabels(years[::2], rotation=45, ha="right", fontsize=8)

    # (3) Ty trong Tay Nguyen / ca nuoc
    a = ax[1, 0]
    a.plot(df["Nam"], df["share"], color="#16a085", lw=2.2, marker="o", ms=3)
    a.fill_between(df["Nam"], df["share"], color="#16a085", alpha=0.1)
    a.set_title("Ty trong san luong 5 tinh Tay Nguyen / ca nuoc (%)", fontweight="bold")
    a.set_xlabel("Nam"); a.set_ylabel("%"); a.grid(alpha=0.3)
    a.set_ylim(80, 100)
    a.set_xticks(years[::2]); a.set_xticklabels(years[::2], rotation=45, ha="right", fontsize=8)

    # (4) Top tinh nam moi nhat
    a = ax[1, 1]
    last = df.iloc[-1]
    s = last[prov].dropna().sort_values(ascending=True).tail(10) / 1000
    colors = [CMAP.get(p, "#95a5a6") for p in s.index]
    a.barh(range(len(s)), s.values, color=colors)
    a.set_yticks(range(len(s))); a.set_yticklabels(s.index, fontsize=8)
    for i, v in enumerate(s.values):
        a.text(v + 3, i, f"{v:.0f}", va="center", fontsize=8)
    a.set_title(f"Top 10 tinh theo san luong nam {int(last['Nam'])} (nghin tan)", fontweight="bold")
    a.set_xlabel("nghin tan"); a.grid(alpha=0.3, axis="x")

    fig.suptitle("EDA SAN LUONG CA PHE THEO TINH (2005-2024)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, "sanluong_overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def chart_heatmap(df, prov, outdir):
    order = df[prov].notna().sum().sort_values(ascending=False).index
    M = df.set_index("Nam")[order].T
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(M.notna().astype(int), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=90, fontsize=7)
    ax.set_title("Do phu du lieu san luong theo tinh x nam (xanh = co data)", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "sanluong_coverage.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"data/Processing/coffe/sanluong_cafe.csv")
    ap.add_argument("--outdir", default=r"reports/figures/cafe/sanluong")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.input)
    prov = [c for c in df.columns if c != "Nam"]
    p1 = chart_overview(df, prov, args.outdir)
    p2 = chart_heatmap(df, prov, args.outdir)
    print("Saved:", p1, p2)


if __name__ == "__main__":
    main()
