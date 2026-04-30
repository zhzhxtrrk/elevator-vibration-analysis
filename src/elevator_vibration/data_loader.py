"""
S3 data loader for the LBB SimTech elevator vibration dataset.

Bucket: lbb-simtech-raw (ap-southeast-1)
Path:   {serial}/{yyyy-mm-dd}/{trip_no}/acc/{start}-{end}-f{from}-t{to}.zip

Each zip contains a single .txt file with one CSV-like line of quoted records:
    "HH:MM:SS:microseconds,x,y,z","HH:MM:SS:microseconds,x,y,z",...
    Sampling: 100 Hz
    Axes: X/Y = lateral, Z = vertical (includes ~1g gravity)

Credentials are loaded from .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_DEFAULT_REGION) in the project root. Falls back to default AWS credential chain.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterator

import boto3
import numpy as np
from botocore.exceptions import ClientError


def _load_dotenv() -> None:
    """Load .env from project root if present. Preloads AWS creds into os.environ."""
    # Walk up from this file to find .env
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────

@dataclass
class TripRef:
    """Reference to a single elevator trip on S3."""

    serial: str
    trip_date: date
    trip_no: str
    start: str
    end: str
    from_floor: str
    to_floor: str
    _s3_key: str = field(default="", repr=False)

    @property
    def s3_key(self) -> str:
        if self._s3_key:
            return self._s3_key
        raise ValueError("s3_key not stored; use list_trip_files() to get full keys")

    @property
    def label(self) -> str:
        return (
            f"{self.serial} | {self.trip_date} | trip={self.trip_no} | "
            f"{self.from_floor}→{self.to_floor} | {self.start}–{self.end}"
        )


@dataclass
class VibrationRecord:
    """Parsed vibration record from S3 data."""

    trip: TripRef
    time_seconds: np.ndarray          # relative time in seconds from trip start
    ax: np.ndarray                    # lateral (door direction), m/s²
    ay: np.ndarray                    # lateral (sideways), m/s²
    az: np.ndarray                    # vertical (with gravity), m/s²
    sample_rate: float = 100.0

    @property
    def n_samples(self) -> int:
        return len(self.time_seconds)

    @property
    def duration(self) -> float:
        return self.time_seconds[-1] - self.time_seconds[0]

    @property
    def a_total(self) -> np.ndarray:
        """Composite magnitude (raw, gravity still in az)."""
        return np.sqrt(self.ax**2 + self.ay**2 + self.az**2)

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({
            "t": self.time_seconds,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "a_total": self.a_total,
        })


# ──────────────────────────────────────────────
#  Core loader
# ──────────────────────────────────────────────

class ElevatorDataLoader:
    """Load and parse elevator vibration data from S3."""

    BUCKET = "lbb-simtech-raw"
    REGION = "ap-southeast-1"

    # Regex to extract each quoted record: "HH:MM:SS:us,x,y,z"
    _RECORD_RE = re.compile(r'"([^"]+)"')

    def __init__(self, profile_name: str | None = None):
        session = boto3.Session(profile_name=profile_name, region_name=self.REGION)
        self.s3 = session.client("s3")

    # ── Listing ────────────────────────────────

    def list_serials(self) -> list[str]:
        """List all elevator serial numbers in the bucket."""
        return self._list_prefixes("")

    def list_dates(self, serial: str) -> list[str]:
        """List available dates for a serial, e.g. ['2026-04-01', ...]."""
        return self._list_prefixes(f"{serial}/")

    def list_trips(self, serial: str, trip_date: str | date) -> list[str]:
        """List trip numbers for a serial+date, e.g. ['0001', '0002', ...]."""
        if isinstance(trip_date, date):
            trip_date = trip_date.isoformat()
        return self._list_prefixes(f"{serial}/{trip_date}/")

    def list_trip_files(self, serial: str, trip_date: str | date, trip_no: str) -> list[TripRef]:
        """List all accelerometer zip files for a given trip."""
        if isinstance(trip_date, date):
            trip_date = trip_date.isoformat()
        prefix = f"{serial}/{trip_date}/{trip_no}/acc/"
        keys = self._list_keys(prefix)
        return [self._parse_trip_ref(key) for key in keys]

    def list_all_trip_refs(self, serial: str, trip_date: str | date) -> list[TripRef]:
        """List all trip references for a serial+date (across all trips)."""
        if isinstance(trip_date, date):
            trip_date = trip_date.isoformat()
        prefix = f"{serial}/{trip_date}/"
        keys = self._list_keys(prefix, suffix=".zip")
        refs = []
        for key in keys:
            try:
                refs.append(self._parse_trip_ref(key))
            except ValueError:
                continue
        return refs

    def list_recent_refs(
        self, serial: str, limit_dates: int = 7, limit_per_date: int = 10
    ) -> list[TripRef]:
        """Convenience: list recent trip refs for a serial."""
        dates = sorted(self.list_dates(serial), reverse=True)[:limit_dates]
        all_refs: list[TripRef] = []
        for d in dates:
            refs = self.list_all_trip_refs(serial, d)[:limit_per_date]
            all_refs.extend(refs)
        return all_refs

    # ── Loading ─────────────────────────────────

    def load_trip(self, ref: TripRef) -> VibrationRecord:
        """Download, unzip, parse a single trip's vibration data."""
        raw = self._download(ref.s3_key)
        arrays = self._parse_line(raw)
        time_sec = self._build_time_array(arrays)
        return VibrationRecord(
            trip=ref,
            time_seconds=time_sec,
            ax=arrays["ax"],
            ay=arrays["ay"],
            az=arrays["az"],
        )

    def load_trip_from_s3_key(self, s3_key: str) -> VibrationRecord:
        ref = self._parse_trip_ref(s3_key)
        return self.load_trip(ref)

    def load_trips(self, refs: list[TripRef]) -> list[VibrationRecord]:
        return [self.load_trip(r) for r in refs]

    # ── Internal helpers ────────────────────────

    def _list_prefixes(self, prefix: str) -> list[str]:
        """List common prefixes (directories) under a prefix."""
        try:
            resp = self.s3.list_objects_v2(
                Bucket=self.BUCKET, Prefix=prefix, Delimiter="/"
            )
        except ClientError as e:
            raise RuntimeError(f"S3 list failed: {e}") from e
        prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
        # Strip the `prefix` part and trailing slash
        offset = len(prefix)
        return [p[offset:].rstrip("/") for p in prefixes]

    def _list_keys(self, prefix: str, suffix: str | None = None) -> list[str]:
        """List object keys under a prefix."""
        keys: list[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix is None or key.endswith(suffix):
                    keys.append(key)
        return keys

    def _download(self, s3_key: str) -> str:
        """Download a file from S3 and return its decoded text."""
        try:
            resp = self.s3.get_object(Bucket=self.BUCKET, Key=s3_key)
        except ClientError as e:
            raise RuntimeError(f"S3 download failed for s3://{self.BUCKET}/{s3_key}: {e}") from e
        body = resp["Body"].read()
        # It's a zip — extract the first (and only) member
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            namelist = zf.namelist()
            if not namelist:
                raise ValueError(f"Empty zip: {s3_key}")
            return zf.read(namelist[0]).decode("utf-8")

    def _parse_line(self, text: str) -> dict[str, np.ndarray]:
        """Parse the single-line format: "HH:MM:SS:us,x,y,z","...",..."""
        matches = self._RECORD_RE.findall(text)
        if not matches:
            raise ValueError("No records found in input text")

        n = len(matches)
        timestamps = [None] * n
        ax = np.empty(n, dtype=np.float64)
        ay = np.empty(n, dtype=np.float64)
        az = np.empty(n, dtype=np.float64)

        for i, m in enumerate(matches):
            parts = m.split(",")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 comma-separated fields, got {len(parts)}: {m[:80]}")
            timestamps[i] = parts[0].strip()
            ax[i] = float(parts[1])
            ay[i] = float(parts[2])
            az[i] = float(parts[3])

        # Parse timestamps
        time_sec = self._parse_timestamps(timestamps)

        return {"time_sec": time_sec, "ax": ax, "ay": ay, "az": az}

    @staticmethod
    def _parse_timestamps(ts_list: list[str]) -> np.ndarray:
        """Parse HH:MM:SS:microseconds → seconds from midnight."""
        out = np.empty(len(ts_list), dtype=np.float64)
        for i, ts in enumerate(ts_list):
            parts = ts.split(":")
            if len(parts) != 4:
                raise ValueError(f"Expected HH:MM:SS:us, got: {ts}")
            h, m, s, us = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            out[i] = h * 3600.0 + m * 60.0 + s + us / 1_000_000.0
        return out

    @staticmethod
    def _build_time_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
        """Convert absolute timestamps to relative seconds from first sample."""
        return arrays["time_sec"] - arrays["time_sec"][0]

    @staticmethod
    def _parse_trip_ref(s3_key: str) -> TripRef:
        """Parse s3 key into TripRef.

        Actual format:
          S0783L01A/2026-04-29/1/acc/2026-04-29T06:35:32:768618-2026-04-29T06:36:10:224528-f1-t10.zip
        Also seen:
          510519A/2024-11-15/9243/acc/2024-11-15T00:10:43:091214.zip  (single timestamp)
        """
        key = s3_key.removesuffix(".zip")
        parts = key.split("/")
        if len(parts) < 5:
            raise ValueError(f"Cannot parse trip ref from key: {s3_key}")

        serial = parts[0]
        trip_date = date.fromisoformat(parts[1])
        trip_no = parts[2]
        # parts[3] = 'acc'
        filename = parts[4]

        # Try two-timestamp format: YYYY-MM-DDTHH:MM:SS:us-YYYY-MM-DDTHH:MM:SS:us-fF-tT
        m = re.match(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}:\d+)-"
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}:\d+)-"
            r"f([^-]+)-t(.+)",
            filename,
        )
        if m:
            return TripRef(
                serial=serial,
                trip_date=trip_date,
                trip_no=trip_no,
                start=m.group(1),
                end=m.group(2),
                from_floor=m.group(3),
                to_floor=m.group(4),
                _s3_key=f"{key}.zip" if not key.endswith(".zip") else key,
            )

        # Try single-timestamp format: YYYY-MM-DDTHH:MM:SS:us
        m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}:\d+)", filename)
        if m:
            return TripRef(
                serial=serial,
                trip_date=trip_date,
                trip_no=trip_no,
                start=m.group(1),
                end="?",
                from_floor="?",
                to_floor="?",
                _s3_key=f"{key}.zip" if not key.endswith(".zip") else key,
            )

        raise ValueError(f"Cannot parse filename: {filename}")


# ──────────────────────────────────────────────
#  Quick-start helpers
# ──────────────────────────────────────────────

def quick_browse(serial: str | None = None) -> list[TripRef]:
    """Browse recent trips. If serial is None, list all serials first."""
    loader = ElevatorDataLoader()
    if serial is None:
        serials = loader.list_serials()
        if not serials:
            raise RuntimeError("No serials found in bucket")
        print(f"Available serials: {serials}")
        serial = serials[0]
        print(f"Using first serial: {serial}")
    return loader.list_recent_refs(serial)
