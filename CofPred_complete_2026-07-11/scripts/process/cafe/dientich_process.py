# -*- coding: utf-8 -*-
"""
Xu ly DIEN TICH ca phe -> chi 5 tinh Tay Nguyen, chi xuat 2020-2025.
Viec lam:
  1) Sua loi cot Lam Dong trung khit Dak Lak (cac nam 1995-2014) -> NaN.
  2) Loc ra 5 tinh Tay Nguyen: Dak Lak, Lam Dong, Gia Lai, Dak Nong, Kon Tum.
  3) Fill nam 2025 = forward-fill tu 2024 (dien tich doi rat cham).
  4) Them cot Tong_TayNguyen = tong dien tich 5 tinh moi nam.
  5) Chi xuat khoang start_year..fill_year (mac dinh 2020-2025).
Chay:  python3 21_dientich_taynguyen.py --input dientich_cafe.csv --out dientich_taynguyen_clean.csv
"""
import argparse
import numpy as np
import pandas as pd

TN5 = ["\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng", "Gia Lai", "\u0110\u1eafk N\u00f4ng", "Kon Tum"]


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Nam"})
    df["Nam"] = df["Nam"].astype(int)
    return df


def process(df, fill_year=2025, start_year=2020):
    # 1) Sua loi Lam Dong trung khit Dak Lak (loi nhap lieu) -> NaN
    dup = df["L\u00e2m \u0110\u1ed3ng"] == df["\u0110\u1eafk L\u1eafk"]
    df.loc[dup, "L\u00e2m \u0110\u1ed3ng"] = np.nan
    print("Da loai", int(dup.sum()), "nam Lam Dong bi trung Dak Lak:",
          list(df.loc[dup, "Nam"].astype(int)))

    # 2) Loc 5 tinh Tay Nguyen
    out = df[["Nam"] + TN5].copy().sort_values("Nam").reset_index(drop=True)

    # 3) Fill nam 2025 (va cac nam thieu o duoi) = forward-fill tu nam cuoi co data
    last_year = int(out["Nam"].max())
    if fill_year > last_year:
        for y in range(last_year + 1, fill_year + 1):
            prev = out[out["Nam"] == y - 1].iloc[0]
            new_row = {"Nam": y}
            for p in TN5:
                new_row[p] = prev[p]  # forward-fill
            out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Da fill nam {last_year+1}..{fill_year} = forward-fill tu {last_year}")

    # Fill lo trong o giua (neu co) bang noi suy tuyen tinh theo nam
    out[TN5] = out[TN5].interpolate(method="linear", limit_direction="both")

    # 4) Them cot tong
    out["Tong_TayNguyen"] = out[TN5].sum(axis=1)
    out["Nam"] = out["Nam"].astype(int)

    # 5) Chi xuat start_year..fill_year
    out = out[(out["Nam"] >= start_year) & (out["Nam"] <= fill_year)].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Processing/coffe/dientich_cafe.csv")
    ap.add_argument("--out", default="data/Processing/coffe/dientich_taynguyen_clean.csv")
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
