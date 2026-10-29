"""Loading NHANES into an insulin-resistance analysis cohort.

NHANES is a complex, multistage, stratified probability sample, not a simple
random sample of the United States. Three consequences drive everything here,
and getting any of them wrong produces numbers that look fine and are wrong.

**The fasting subsample has its own weight.** Glucose, insulin and
triglycerides are measured only in participants examined in the morning
session after an overnight fast, which is roughly a third of the sample. That
subsample has its own weight, ``WTSAF2YR``. Using the general examination
weight ``WTMEC2YR`` instead is the standard mistake, and it biases every
population estimate because the fasting subsample is not a random subset.

**Pooled cycles need rescaled weights.** CDC's guidance for combining k
two-year cycles is to divide the two-year weight by k, otherwise the pooled
population total comes out k times too large.

**Standard errors need the design variables.** ``SDMVPSU`` and ``SDMVSTRA``
describe the clustering and stratification. Ignoring them and computing a
naive standard error understates it, often by a lot, because people in the
same sampled cluster resemble each other.

There is also an assay discontinuity, handled by :func:`load_cycles` refusing
to cross it. NHANES measured serum insulin on a Roche Elecsys 2010 through
2011-2012 and on a Tosoh AIA-900 from 2013-2014. CDC's crossover study on 237
specimens found the Roche mean ran 11.3% higher than Tosoh (16.01 against
14.39 uU/mL). Pooling across that break without applying CDC's conversion
silently shifts every insulin-derived quantity, HOMA-IR included, so the
loader treats it as an error rather than a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"


@dataclass(frozen=True)
class Cycle:
    """One two-year NHANES cycle.

    Attributes:
        name: the cycle as usually written, e.g. "2017-2018".
        start_year: first year, which is what the download URL uses.
        suffix: the letter CDC appends to every file in the cycle.
        insulin_assay: analyser used for serum insulin. Cycles with different
            assays are not directly comparable.
    """

    name: str
    start_year: int
    suffix: str
    insulin_assay: str


# Only cycles this project has verified against CDC documentation.
CYCLES: dict[str, Cycle] = {
    "2011-2012": Cycle("2011-2012", 2011, "G", "Roche Elecsys 2010"),
    "2013-2014": Cycle("2013-2014", 2013, "H", "Tosoh AIA-900"),
    "2015-2016": Cycle("2015-2016", 2015, "I", "Tosoh AIA-900"),
    "2017-2018": Cycle("2017-2018", 2017, "J", "Tosoh AIA-900"),
}

DEFAULT_CYCLES: tuple[str, ...] = ("2013-2014", "2015-2016", "2017-2018")

# NHANES file stem -> the columns this project needs from it.
COMPONENTS: dict[str, tuple[str, ...]] = {
    "DEMO": ("SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "RIDEXPRG",
             "SDMVPSU", "SDMVSTRA", "WTMEC2YR"),
    "GLU": ("SEQN", "LBXGLU", "WTSAF2YR"),
    "INS": ("SEQN", "LBXIN"),
    "BMX": ("SEQN", "BMXBMI", "BMXWAIST", "BMXHT", "BMXWT"),
    "TRIGLY": ("SEQN", "LBXTR"),
    "HDL": ("SEQN", "LBDHDD"),
    "GHB": ("SEQN", "LBXGH"),
    "BPX": ("SEQN", "BPXSY1", "BPXDI1"),
    "DIQ": ("SEQN", "DIQ010", "DIQ050"),
    "BIOPRO": ("SEQN", "LBXSATSI"),
}

# Readable names, so no downstream code has to remember that LBXIN is insulin.
RENAMES: dict[str, str] = {
    "RIAGENDR": "sex", "RIDAGEYR": "age", "RIDRETH3": "ethnicity",
    "RIDEXPRG": "pregnant", "SDMVPSU": "psu", "SDMVSTRA": "stratum",
    "LBXGLU": "glucose_mg_dl", "LBXIN": "insulin_uU_mL",
    "BMXBMI": "bmi", "BMXWAIST": "waist_cm", "BMXHT": "height_cm", "BMXWT": "weight_kg",
    "LBXTR": "triglycerides_mg_dl", "LBDHDD": "hdl_mg_dl", "LBXGH": "hba1c_pct",
    "BPXSY1": "systolic_bp", "BPXDI1": "diastolic_bp", "LBXSATSI": "alt_u_l",
}


class AssayMismatchError(ValueError):
    """Raised when pooling cycles whose insulin results are not comparable."""


def download_cycle(cycle: Cycle, directory: str | Path) -> dict[str, Path]:
    """Fetch every component file for one cycle, skipping what is already present."""
    from urllib.request import urlopen

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for component in COMPONENTS:
        stem = f"{component}_{cycle.suffix}"
        path = directory / f"{stem}.xpt"
        if not path.exists() or path.stat().st_size < 1024:
            url = f"{BASE_URL}/{cycle.start_year}/DataFiles/{stem}.xpt"
            with urlopen(url, timeout=300) as response:  # noqa: S310
                payload = response.read()
            # CDC answers a missing file with a 200 and an HTML error page, so
            # a status check is not enough; the content has to be inspected.
            if payload[:14].lstrip().startswith(b"<"):
                raise FileNotFoundError(f"{url} returned HTML, not an XPORT file")
            path.write_bytes(payload)
        paths[component] = path
    return paths


def read_cycle(cycle: Cycle, directory: str | Path) -> pd.DataFrame:
    """Merge one cycle's component files into a single wide frame."""
    directory = Path(directory)
    frame: pd.DataFrame | None = None

    for component, columns in COMPONENTS.items():
        path = directory / f"{component}_{cycle.suffix}.xpt"
        if not path.exists():
            continue
        part = pd.read_sas(path)
        keep = [c for c in columns if c in part.columns]
        part = part[keep]
        frame = part if frame is None else frame.merge(part, on="SEQN", how="left")

    if frame is None:
        raise FileNotFoundError(f"no component files for {cycle.name} under {directory}")

    frame = frame.rename(columns=RENAMES)
    frame["cycle"] = cycle.name
    frame["insulin_assay"] = cycle.insulin_assay
    return frame


def load_cycles(
    cycle_names: tuple[str, ...] = DEFAULT_CYCLES,
    directory: str | Path = "data/nhanes",
    allow_assay_mix: bool = False,
) -> pd.DataFrame:
    """Load and pool cycles, rescaling the fasting weight.

    Args:
        cycle_names: which cycles to pool.
        directory: where the .xpt files live.
        allow_assay_mix: permit pooling cycles measured on different insulin
            analysers. Off by default, and turning it on without applying
            CDC's conversion will bias every insulin-derived quantity.

    Returns:
        One row per participant, with ``weight`` already divided by the number
        of pooled cycles per CDC guidance.
    """
    unknown = set(cycle_names) - set(CYCLES)
    if unknown:
        raise KeyError(f"unknown cycle(s) {sorted(unknown)}; known: {sorted(CYCLES)}")

    cycles = [CYCLES[name] for name in cycle_names]
    assays = {c.insulin_assay for c in cycles}
    if len(assays) > 1 and not allow_assay_mix:
        raise AssayMismatchError(
            f"cycles {list(cycle_names)} span insulin assays {sorted(assays)}. "
            "CDC's crossover study found the Roche mean 11.3% above Tosoh "
            "(16.01 vs 14.39 uU/mL), so pooling without applying CDC's "
            "conversion biases HOMA-IR and every other insulin-derived value. "
            "Restrict to one assay, or pass allow_assay_mix=True deliberately."
        )

    frames = [read_cycle(cycle, directory) for cycle in cycles]
    pooled = pd.concat(frames, ignore_index=True)

    # CDC: when combining k two-year cycles, divide the two-year weight by k.
    pooled["weight"] = pooled["WTSAF2YR"] / len(cycles)
    return pooled
