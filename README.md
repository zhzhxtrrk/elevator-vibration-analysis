# Elevator Vibration Analysis / 电梯振动分析

Triaxial (X-Y-Z) elevator vibration data analysis toolkit. 100 Hz sampling rate.
Door-mounted accelerometer (single side).

三轴电梯振动数据分析工具包。100 Hz采样率。门装加速度计（单侧安装）。

## Sensor Configuration / 传感器配置

| Axis / 轴 | Direction / 方向 | Measures / 测量 |
|:----------|:-----------------|:----------------|
| X | Door open/close / 门开关方向 | Door rattle, door operation / 门板振动 |
| Y | Lateral along door / 门侧向 | Sideways sway / 侧向摆动 |
| Z | Vertical / 垂直 | Ride dynamics, gravity / 运行、重力 |

> ⚠️ All data represents **door panel vibration**, not car body vibration.
> 所有数据反映**门板振动**，不是轿厢本体振动。

## Quick Start / 快速开始

```bash
cd ~/Workspace/elevator-vibration-analysis
uv venv
uv pip install -e ".[dev]"
jupyter notebook notebooks/exploration.ipynb
```

## Project Structure / 项目结构

```
elevator-vibration-analysis/
├── docs/
│   └── high-level-report.md    # Analysis framework / 分析框架
├── src/elevator_vibration/     # Python toolkit
│   ├── data_loader.py          # S3 ingestion (lbb-simtech-raw)
│   ├── preprocessing.py        # DC, gravity, noise, door mask
│   ├── time_domain.py          # RMS, peak, kurtosis, jerk, run detection
│   ├── frequency.py            # FFT, PSD, STFT, envelope, coherence
│   ├── ride_quality.py         # ISO 2631-1, ISO 18738 + Z-only mode
│   ├── state_detection.py      # Phase segmentation + door rattle index
│   ├── diagnostics.py          # Door rattle, unbalance, misalignment, looseness, bearing
│   └── visualization.py        # Timeseries, spectrograms, dashboards
├── notebooks/
│   └── exploration.ipynb       # 26-cell comprehensive bilingual analysis
├── data/
│   └── samples/                # Test data (S0783L01A, 2 dates)
├── pyproject.toml
└── README.md
```

## Analysis Modules / 分析模块

| Module / 模块 | Description / 描述 | Status |
|:--------------|:-------------------|:------:|
| preprocessing | DC removal, gravity, noise floor, door mask | ✅ |
| time_domain | RMS, peak, kurtosis, crest factor, jerk | ✅ |
| frequency | FFT, PSD, STFT, envelope spectrum, cepstrum | ✅ |
| ride_quality | ISO 2631-1 (Z-only mode), ISO 18738 | ✅ |
| state_detection | Phase timeline, door operation, door rattle index | ✅ |
| diagnostics | Door rattle, unbalance, misalignment, looseness, bearing | ✅ |
| visualization | Timeseries, PSD, spectrograms, dashboards | ✅ |

## Key Features / 关键功能

- **Z-only ISO 2631** — Recommended for door-mounted sensors (X/Y contaminated by door panel)
- **Door Rattle Index** — X-axis cruise RMS / noise floor ratio
- **Door Operation Detection** — From X-axis spikes (no external door sensor needed)
- **Health Score** — 0-100 aggregate diagnostic score per trip
- **Occupancy Estimation** — Heuristic from standing-segment activity
- **Entrapment Risk** — Door cycle analysis + extended standing detection
- **Bilingual Reports** — Chinese + English auto-generated markdown reports

## References / 参考

- ISO 2631-1:1997 — Whole-body vibration
- ISO 18738:2003 — Lift ride quality
- GB/T 24474-2009 — Chinese elevator ride quality standard

See `docs/high-level-report.md` for the full analysis framework.
