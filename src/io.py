from __future__ import annotations
import pandas as pd


def read_csv_bytes(file_bytes: bytes, encoding: str | None = None) -> pd.DataFrame:
    """
    Read CSV bytes safely. Try utf-8-sig first, then fallback to cp932 (common in JP CSV).
    """
    if encoding:
        return pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding=encoding)

    # try common encodings
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding=enc)
        except Exception:
            pass
    # last resort
    return pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding="utf-8", errors="replace")
