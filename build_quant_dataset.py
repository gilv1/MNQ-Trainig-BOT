#!/usr/bin/env python3
"""Construye un dataset cuantitativo multi-fuente para ES/NQ/MNQ/MES.

Genera automáticamente:
- CSVs en `quant_dataset/`
- SQLite DB en `quant_dataset/market_data.db`
- Dataset combinado para análisis cuantitativo

Fuentes:
- Yahoo Finance (yfinance)
- Stooq (verificación de precios)
- FRED (macro, requiere API key opcional)
"""

from __future__ import annotations

import argparse
import io
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

try:
    from fredapi import Fred
except ImportError:  # dependencia opcional
    Fred = None


# -----------------------------
# Configuración
# -----------------------------
@dataclass(frozen=True)
class ContractConfig:
    name: str
    yahoo_ticker: str
    stooq_symbol: str


CONTRACTS: tuple[ContractConfig, ...] = (
    ContractConfig("ES", "ES=F", "es.f"),
    ContractConfig("NQ", "NQ=F", "nq.f"),
    ContractConfig("MNQ", "MNQ=F", "mnq.f"),
    ContractConfig("MES", "MES=F", "mes.f"),
)

MACRO_SERIES = {
    "CPI": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "UNRATE": "UNRATE",
    "GDP": "GDP",
}

INTERVAL_RULES = {
    # Límites de Yahoo Finance para evitar errores de rango:
    # - 1m: máximo 8 días por petición
    # - 5m: últimos 60 días
    "1m": "7d",
    "5m": "60d",
    "1d": "10y",
}


# -----------------------------
# Utilidades
# -----------------------------
def ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    macro_dir = base_dir / "macro"
    base_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, macro_dir


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out = out.reset_index()

    # yfinance puede devolver Date o Datetime
    if "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "timestamp"})
    elif "Date" in out.columns:
        out = out.rename(columns={"Date": "timestamp"})

    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out = out.rename(columns=col_map)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def fetch_yf(ticker: str, interval: str, period: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_ohlcv(raw)


def fetch_yf_safe(ticker: str, interval: str, period: str, label: str) -> pd.DataFrame:
    try:
        df = fetch_yf(ticker, interval=interval, period=period)
    except Exception as exc:
        print(f"[WARN] {label}: error descargando {ticker} {interval}/{period}: {exc}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    if df.empty:
        print(f"[WARN] {label}: sin datos para {ticker} {interval}/{period}.")
    return df


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    # Stooq CSV endpoint: d1 = diario
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    if not resp.text.strip():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = pd.read_csv(io.StringIO(resp.text))
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = out.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_sqlite_tables(db_path: Path, frames: Dict[str, pd.DataFrame]) -> None:
    with sqlite3.connect(db_path) as conn:
        for table_name, df in frames.items():
            out = df.copy()
            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
                out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
            out.to_sql(table_name, conn, if_exists="replace", index=False)


def compute_intraday_signals(df_1m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()

    tmp = df_1m.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    tmp["ret_1m"] = tmp["close"].pct_change()
    tmp["minute_of_day"] = tmp["timestamp"].dt.strftime("%H:%M")
    tmp["hour"] = tmp["timestamp"].dt.hour
    tmp["is_breakout"] = (
        (tmp["high"] > tmp["high"].rolling(30, min_periods=30).max().shift(1))
        | (tmp["low"] < tmp["low"].rolling(30, min_periods=30).min().shift(1))
    ).astype("Int64")

    grouped = (
        tmp.groupby("minute_of_day", as_index=False)
        .agg(
            vol_1m=("ret_1m", "std"),
            edge_mean_ret=("ret_1m", "mean"),
            breakout_prob=("is_breakout", "mean"),
        )
        .sort_values("minute_of_day")
    )
    grouped.insert(0, "symbol", symbol)
    return grouped


def build_analysis_dataset(daily_frames: Dict[str, pd.DataFrame], vix: pd.DataFrame, dxy: pd.DataFrame) -> pd.DataFrame:
    def prep(df: pd.DataFrame, close_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["date", close_name])
        out = df[["timestamp", "close"]].copy()
        out["date"] = pd.to_datetime(out["timestamp"], utc=True).dt.date
        out = out.groupby("date", as_index=False)["close"].last().rename(columns={"close": close_name})
        return out

    es = prep(daily_frames.get("ES", pd.DataFrame()), "ES_close")
    nq = prep(daily_frames.get("NQ", pd.DataFrame()), "NQ_close")
    vix_ = prep(vix, "VIX_close")
    dxy_ = prep(dxy, "DXY_close")

    merged = es.merge(nq, on="date", how="outer").merge(vix_, on="date", how="outer").merge(dxy_, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    for col in ["ES_close", "NQ_close", "VIX_close", "DXY_close"]:
        merged[f"{col}_ret"] = merged[col].pct_change()

    # correlación rodante NQ vs VIX (aprox. 20 sesiones)
    merged["corr_nq_vix_20d"] = merged["NQ_close_ret"].rolling(20, min_periods=20).corr(merged["VIX_close_ret"])
    return merged


def fetch_fred_series(api_key: str, macro_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    if Fred is None:
        print("[WARN] fredapi no está instalado; se omiten series macro.")
        return out

    if not api_key:
        print("[WARN] FRED_API_KEY no configurada; se omiten series macro.")
        return out

    fred = Fred(api_key=api_key)

    for name, code in MACRO_SERIES.items():
        try:
            series = fred.get_series(code)
            df = pd.DataFrame({"timestamp": series.index, "value": series.values})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
            save_csv(df, macro_dir / f"{name}.csv")
            out[f"macro_{name.lower()}"] = df
            print(f"[OK] Macro guardado: {name}")
        except Exception as exc:
            print(f"[WARN] No se pudo descargar {name} ({code}): {exc}")

    return out


def build_dataset(output_dir: Path, fred_api_key: Optional[str]) -> None:
    data_dir, macro_dir = ensure_dirs(output_dir)

    sqlite_frames: Dict[str, pd.DataFrame] = {}
    daily_by_symbol: Dict[str, pd.DataFrame] = {}
    intraday_signals: list[pd.DataFrame] = []

    print("Descargando futuros (Yahoo Finance)...")
    for contract in CONTRACTS:
        for interval, period in INTERVAL_RULES.items():
            try:
                df = fetch_yf_safe(
                    contract.yahoo_ticker,
                    interval=interval,
                    period=period,
                    label=contract.name,
                )
                suffix = "daily" if interval == "1d" else interval
                file_name = f"{contract.name}_{suffix}.csv"
                save_csv(df, data_dir / file_name)
                table_name = f"{contract.name.lower()}_{suffix}".replace("-", "_")
                sqlite_frames[table_name] = df

                if interval == "1d":
                    daily_by_symbol[contract.name] = df

                if interval == "1m":
                    sig = compute_intraday_signals(df, contract.name)
                    if not sig.empty:
                        intraday_signals.append(sig)

                print(f"[OK] {contract.name} {interval} ({period}) -> {file_name}")
            except Exception as exc:
                print(f"[WARN] Error en {contract.name} {interval}: {exc}")

    # Compatibilidad con nombres esperados por el usuario
    if "NQ" in daily_by_symbol:
        save_csv(daily_by_symbol["NQ"], data_dir / "NQ_daily.csv")
    if "ES" in daily_by_symbol:
        save_csv(daily_by_symbol["ES"], data_dir / "ES_daily.csv")

    # Micro sintético a partir de contratos mini (si existe base)
    if "NQ" in daily_by_symbol:
        mnq_synth = daily_by_symbol["NQ"].copy()
        for col in ["open", "high", "low", "close"]:
            mnq_synth[col] = mnq_synth[col] * 0.1
        save_csv(mnq_synth, data_dir / "MNQ_synthetic.csv")
        sqlite_frames["mnq_synthetic"] = mnq_synth

    if "ES" in daily_by_symbol:
        mes_synth = daily_by_symbol["ES"].copy()
        for col in ["open", "high", "low", "close"]:
            mes_synth[col] = mes_synth[col] * 0.1
        save_csv(mes_synth, data_dir / "MES_synthetic.csv")
        sqlite_frames["mes_synthetic"] = mes_synth

    print("Descargando verificación desde Stooq...")
    for contract in CONTRACTS:
        try:
            stooq_df = fetch_stooq_daily(contract.stooq_symbol)
            path = data_dir / f"{contract.name}_stooq_daily.csv"
            save_csv(stooq_df, path)
            sqlite_frames[f"{contract.name.lower()}_stooq_daily"] = stooq_df
            print(f"[OK] Stooq {contract.name} -> {path.name}")
        except Exception as exc:
            print(f"[WARN] No se pudo descargar Stooq {contract.name}: {exc}")

    print("Descargando VIX y DXY...")
    vix = fetch_yf_safe("^VIX", interval="1d", period="10y", label="VIX")
    dxy = fetch_yf_safe("DX-Y.NYB", interval="1d", period="10y", label="DXY")
    save_csv(vix, data_dir / "VIX.csv")
    save_csv(dxy, data_dir / "DXY.csv")
    sqlite_frames["vix_daily"] = vix
    sqlite_frames["dxy_daily"] = dxy

    print("Descargando macro (FRED)...")
    fred_frames = fetch_fred_series(fred_api_key or "", macro_dir)
    sqlite_frames.update(fred_frames)

    if intraday_signals:
        intraday = pd.concat(intraday_signals, ignore_index=True)
        save_csv(intraday, data_dir / "intraday_signals.csv")
        sqlite_frames["intraday_signals"] = intraday

    analysis = build_analysis_dataset(daily_by_symbol, vix, dxy)
    save_csv(analysis, data_dir / "analysis_dataset.csv")
    sqlite_frames["analysis_dataset"] = analysis

    db_path = data_dir / "market_data.db"
    write_sqlite_tables(db_path, sqlite_frames)

    print("\nDataset cuantitativo listo.")
    print(f"- Carpeta: {data_dir.resolve()}")
    print(f"- Base de datos SQLite: {db_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye dataset cuantitativo multi-fuente")
    parser.add_argument(
        "--output-dir",
        default="quant_dataset",
        type=Path,
        help="Carpeta de salida para CSV y DB (default: quant_dataset)",
    )
    parser.add_argument(
        "--fred-api-key",
        default=os.getenv("FRED_API_KEY", ""),
        help="API key de FRED (default: variable FRED_API_KEY)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(output_dir=args.output_dir, fred_api_key=args.fred_api_key)


if __name__ == "__main__":
    main()
