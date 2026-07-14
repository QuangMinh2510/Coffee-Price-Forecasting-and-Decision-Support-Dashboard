#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tien ich luu du bao ra CSV, dung chung cho cac script CofPred."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "model"


def save_preds(
    model_name: str,
    horizon: int,
    dates: Iterable,
    y_true: Iterable,
    y_pred: Iterable,
    *,
    anchor: Optional[Iterable] = None,
    outdir: str | Path = "results/preds",
) -> Path:
    """Luu tung diem du bao vao CSV va tra ve duong dan file da tao."""
    dates_arr = pd.to_datetime(list(dates), errors="coerce")
    true_arr = np.asarray(list(y_true), dtype=float)
    pred_arr = np.asarray(list(y_pred), dtype=float)

    n = len(true_arr)
    if len(pred_arr) != n or len(dates_arr) != n:
        raise ValueError(
            "dates, y_true va y_pred phai co cung so phan tu: "
            f"dates={len(dates_arr)}, y_true={n}, y_pred={len(pred_arr)}"
        )

    payload = {
        "date": dates_arr,
        "y_true": true_arr,
        "y_pred": pred_arr,
        "error": true_arr - pred_arr,
        "abs_error": np.abs(true_arr - pred_arr),
    }

    if anchor is not None:
        anchor_arr = np.asarray(list(anchor), dtype=float)
        if len(anchor_arr) != n:
            raise ValueError(f"anchor phai co {n} phan tu, nhung nhan {len(anchor_arr)}")
        payload["anchor"] = anchor_arr

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"preds_{_safe_name(model_name)}_h{int(horizon)}.csv"
    pd.DataFrame(payload).to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path
