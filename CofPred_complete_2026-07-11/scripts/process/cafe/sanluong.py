# -*- coding: utf-8 -*-
"""
Processing SAN LUONG ca phe -> chi 5 tinh Tay Nguyen, chi xuat 2020-2025.
Viec lam:
  1) Loc 5 tinh Tay Nguyen: Dak Lak, Lam Dong, Gia Lai, Dak Nong, Kon Tum.
  2) Co diem bat thuong Kon Tum 2022 (san luong tut ~60%, nghi thay doi thong ke).
  3) Fill nam 2025 = forward-fill tu 2024 (san luong la bien nam, doi cham trong model).
  4) Them cot Tong_TayNguyen (tan) va cot flag.
  5) Chi xuat start_year..fill_year (mac dinh 2020-2025).
Chay: python3 25_process_sanluong.py --input sanluong_cafe.csv --out sanluong_taynguyen_clean.csv
"""
import argparse
import numpy as np, pandas as pd

TN5 = ["\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng", "Gia Lai", "\u0110\u1eafk N\u00f4ng", "Kon Tum"]


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Nam"})
    df["Nam"] = df["Nam"].astype(int)
    return df


def process(df, fill_year=2025, start_year=2020):
    out = df[["Nam"] + TN5].copy().sort_values("Nam").reset_index(drop=True)

    # 1) Co bat thuong: Kon Tum giam manh tu 2022 (>40% so nam truoc)
    out["flag"] = ""
    kt = "Kon Tum"
    drop = out[kt].pct_change()
    bad = drop < -0.4
    out.loc[bad, "flag"] = "KonTum_dut_gay"
    print("Co bat thuong Kon Tum tai nam:", list(out.loc[bad, "Nam"].astype(int)))

    # 2) Fill nam thieu o duoi (2025) = forward-fill tu nam cuoi co data
    last_year = int(out["Nam"].max())
    if fill_year > last_year:
        for y in range(last_year + 1, fill_year + 1):
            prev = out[out["Nam"] == y - 1].iloc[0]
            new_row = {"Nam": y, "flag": "ffill_2024"}
            for p in TN5:
                new_row[p] = prev[p]
            out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Da fill nam {last_year+1}..{fill_year} = forward-fill tu {last_year}")

    # noi suy lo trong giua (neu co)
    out[TN5] = out[TN5].interpolate(method="linear", limit_direction="both")

    # 3) cot tong
    out["Tong_TayNguyen"] = out[TN5].sum(axis=1)
    out["Nam"] = out["Nam"].astype(int)

    # 4) loc khoang nam
    out = out[(out["Nam"] >= start_year) & (out["Nam"] <= fill_year)].reset_index(drop=True)
    # sap xep cot
    out = out[["Nam"] + TN5 + ["Tong_TayNguyen", "flag"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"data/Processing/coffe/sanluong_cafe.csv")
    ap.add_argument("--out", default=r"data/Processing/coffe/sanluong_taynguyen_clean.csv")
    ap.add_argument("--start_year", type=int, default=2020)
    ap.add_argument("--fill_year", type=int, default=2025)
    args = ap.parse_args()

    df = load(args.input)
    out = process(df, fill_year=args.fill_year, start_year=args.start_year)
    out.to_csv(args.out, index=False)
    print("\nKet qua xuat ra:")
    print(out.to_string(index=False))
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
