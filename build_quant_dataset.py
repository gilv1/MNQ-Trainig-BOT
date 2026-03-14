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
import contextlib
import io
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf
from pandas.errors import EmptyDataError

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
    # Horizonte objetivo por intervalo.
    # Para intervalos con límites de Yahoo (1m, 5m), se aplica ajuste/chunking automáticamente.
    "1m": "30d",
    "5m": "1y",
    "1d": "10y",
}

YF_INTERVAL_LIMITS_DAYS = {
    # Máximo de días por request para evitar errores de yfinance.
    "1m": 8,
    "5m": 60,
}


# -----------------------------
# Utilidades
# -----------------------------
def ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    macro_dir = base_dir / "macro"
    base_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, macro_dir


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def read_env_file_value(path: Path, key: str) -> str:
    if not path.exists() or not path.is_file():
        return ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        found_key, _, found_value = line.partition("=")
        if found_key.strip() == key:
            return strip_wrapping_quotes(found_value)

    return ""


def resolve_fred_api_key(cli_value: str) -> str:
    """Resuelve la API key de FRED desde CLI, entorno o `.env` local."""
    if cli_value:
        return strip_wrapping_quotes(cli_value)

    env_value = strip_wrapping_quotes(os.getenv("FRED_API_KEY", ""))
    if env_value:
        return env_value

    return read_env_file_value(Path(".env"), "FRED_API_KEY")


def resolve_finnhub_api_key(cli_value: str) -> str:
    """Resuelve la API key de Finnhub desde CLI, entorno o `.env` local."""
    if cli_value:
        return strip_wrapping_quotes(cli_value)

    # Compatibilidad para ambas variantes: FINNHUB_API_KEY y FINHUB_API_KEY.
    for env_key in ("FINNHUB_API_KEY", "FINHUB_API_KEY"):
        env_value = strip_wrapping_quotes(os.getenv(env_key, ""))
        if env_value:
            return env_value

    for file_key in ("FINNHUB_API_KEY", "FINHUB_API_KEY"):
        file_value = read_env_file_value(Path(".env"), file_key)
        if file_value:
            return file_value

    return ""


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
    if interval in YF_INTERVAL_LIMITS_DAYS:
        return fetch_yf_with_limits(ticker=ticker, interval=interval, period=period)

    raw = run_yf_history_quiet(
        ticker=ticker,
        interval=interval,
        period=period,
    )
    return normalize_ohlcv(raw)


def run_yf_history_quiet(
    ticker: str,
    interval: str,
    period: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Ejecuta yfinance.history silenciando warnings ruidosos en consola."""
    kwargs = {
        "interval": interval,
        "auto_adjust": False,
    }
    if period is not None:
        kwargs["period"] = period
    if start is not None:
        kwargs["start"] = start.to_pydatetime()
    if end is not None:
        kwargs["end"] = end.to_pydatetime()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return yf.Ticker(ticker).history(**kwargs)


def _period_to_days(period: str) -> int:
    period = period.strip().lower()
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("y"):
        return int(period[:-1]) * 365
    raise ValueError(f"Periodo no soportado: {period}")


def fetch_yf_with_limits(ticker: str, interval: str, period: str) -> pd.DataFrame:
    requested_days = _period_to_days(period)
    limit_days = YF_INTERVAL_LIMITS_DAYS[interval]

    # Yahoo limita 1m (~8 días) y 5m (~60 días) sobre ventana reciente.
    if requested_days > limit_days:
        requested_days = limit_days

    end = pd.Timestamp.utcnow().floor("min")
    start = end - pd.Timedelta(days=requested_days)

    frames: list[pd.DataFrame] = []
    chunk_days = limit_days - 1 if interval == "1m" else limit_days
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)
        raw = run_yf_history_quiet(
            ticker=ticker,
            interval=interval,
            start=chunk_start,
            end=chunk_end,
        )
        norm = normalize_ohlcv(raw)
        if not norm.empty:
            frames.append(norm)
        # Evita loop infinito por ventanas minúsculas
        chunk_start = chunk_end + pd.Timedelta(minutes=1)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return out


def fetch_yf_safe(ticker: str, interval: str, period: str, label: str) -> pd.DataFrame:
    try:
        df = fetch_yf(ticker, interval=interval, period=period)
    except Exception as exc:
        print(f"[WARN] {label}: error descargando {ticker} {interval}/{period}: {exc}")
        # fallback adicional para tickers que fallan con history por lote
        try:
            if interval in YF_INTERVAL_LIMITS_DAYS:
                alt = fetch_yf_with_limits(ticker=ticker, interval=interval, period=period)
                return normalize_ohlcv(alt)

            alt = run_yf_history_quiet(ticker=ticker, interval=interval, period=period)
            df = normalize_ohlcv(alt)
        except Exception:
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

    body = resp.text.strip()
    if not body or "No data" in body:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    try:
        out = pd.read_csv(io.StringIO(body))
    except EmptyDataError:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
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


def fetch_finnhub_series(api_key: str, macro_dir: Path) -> Dict[str, pd.DataFrame]:
    """Descarga series macro desde Finnhub usando los códigos FRED."""
    out: Dict[str, pd.DataFrame] = {}

    if not api_key:
        print("[WARN] FINNHUB_API_KEY no configurada; se omiten series macro en Finnhub.")
        return out

    for name, code in MACRO_SERIES.items():
        url = "https://finnhub.io/api/v1/economic"
        try:
            resp = requests.get(url, params={"code": code, "token": api_key}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            data = payload.get("data", []) if isinstance(payload, dict) else []
            if not data:
                print(f"[WARN] Finnhub sin datos para {name} ({code}).")
                continue

            df = pd.DataFrame(data)
            if "date" not in df.columns or "value" not in df.columns:
                print(f"[WARN] Finnhub devolvió formato no esperado para {name} ({code}).")
                continue

            df = df.rename(columns={"date": "timestamp"})[["timestamp", "value"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["timestamp", "value"]).drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)

            save_csv(df, macro_dir / f"{name}.csv")
            out[f"macro_{name.lower()}"] = df
            print(f"[OK] Macro guardado con Finnhub: {name}")
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"[WARN] No se pudo descargar {name} ({code}) en Finnhub: {exc}")

    return out


def build_dataset(output_dir: Path, fred_api_key: Optional[str], finnhub_api_key: Optional[str]) -> None:
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

                if df.empty:
                    print(f"[WARN] {contract.name} {interval} ({period}) sin datos -> {file_name}")
                else:
                    print(f"[OK] {contract.name} {interval} ({period}) {len(df):,} filas -> {file_name}")
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
    if fred_frames:
        sqlite_frames.update(fred_frames)
    else:
        print("Descargando macro (Finnhub)...")
        finnhub_frames = fetch_finnhub_series(finnhub_api_key or "", macro_dir)
        sqlite_frames.update(finnhub_frames)

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
        default="",
        help="API key de FRED (prioridad: CLI > FRED_API_KEY > .env)",
    )
    parser.add_argument(
        "--finnhub-api-key",
        default="",
        help="API key de Finnhub (prioridad: CLI > FINNHUB_API_KEY > .env)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fred_api_key = resolve_fred_api_key(args.fred_api_key)
    finnhub_api_key = resolve_finnhub_api_key(args.finnhub_api_key)
    build_dataset(output_dir=args.output_dir, fred_api_key=fred_api_key, finnhub_api_key=finnhub_api_key)


if __name__ == "__main__":
    main()
