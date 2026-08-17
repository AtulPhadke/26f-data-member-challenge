"""Loading and cleaning the survey.

Run standalone to see what each filter removed:

    python -m src.clean
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "survey.csv"

TARGET = "annual_salary_usd"

# Missing values in this file are the literal string "NA", not empty cells.
NA_VALUES = ["NA", ""]

# How many standard deviations from the mean a salary may sit before it is treated as
# junk, measured on log(salary / country median). 3 removes 98 rows of 4,916.
Z_THRESHOLD = 3.0

# A country needs this many respondents for its median to mean anything.
MIN_COUNTRY_SAMPLE = 5

# Countries need this many respondents to help derive the fallback floor.
BOUNDS_SAMPLE_FLOOR = 30

# Nobody holds a job before about 14. Used against the age bracket to catch impossible
# WorkExp. YearsCode gets no such floor, since coding starts as a hobby.
MIN_WORKING_AGE = 14
AGE_CEILING = {
    "Under 18 years old": 18,
    "18-24 years old": 24,
    "25-34 years old": 34,
    "35-44 years old": 44,
    "45-54 years old": 54,
    "55-64 years old": 64,
    "65 years or older": 100,
}

SECONDARY_SCHOOL = (
    "Secondary school (e.g. American high school, "
    "German Realschule or Gymnasium, etc.)"
)

# Every named qualification above secondary school already has a level, so an education
# write-in is probably off the ladder: a bootcamp, a trade certificate, self-teaching.
# Write-ins sit at 0.78x of own-country median and secondary school at 0.80x.
CATEGORY_MERGES: dict[str, dict[str, str]] = {
    "EdLevel": {"Other (please specify):": SECONDARY_SCHOOL},
}

# OrgSize is a headcount scale, so an answer that is neither a number nor a range has no
# place on it. "Just me" describes employment status, and 82 of its 110 rows are already
# marked freelance or self-employed under Employment.
NULLED_VALUES: dict[str, list[str]] = {
    "OrgSize": [
        "Just me - I am a freelancer, sole proprietor, etc.",
        "I don’t know",
    ],
}

# Blanks in these become their own level. They are meaningful: OrgSize, ICorPM and
# RemoteWork go blank together for the 527 respondents with no employer.
CATEGORICAL_COLS = [
    "Age",
    "EdLevel",
    "Employment",
    "DevType",
    "Industry",
    "OrgSize",
    "ICorPM",
    "RemoteWork",
    "Country",
]

MULTI_SELECT_COLS = ["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith"]

# The semicolon-delimited skills columns become one flag per common item, with this many
# kept from each. 20 languages covers 93% of all mentions; keeping more tests slightly
# worse, keeping fewer loses real signal.
#
# The vocabulary is counted over the whole file rather than per fold, which is a leak in
# principle. Measured, it is nil: the same 20 languages come out of every fold and of the
# full data, and only their tie-break order moves. Databases differ in 3 folds of 5.
SKILL_FLAGS = {"LanguageHaveWorkedWith": ("lang", 20), "DatabaseHaveWorkedWith": ("db", 12)}
NUMERIC_COLS = ["WorkExp", "YearsCode"]
MISSING_LABEL = "Unknown"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV, treating the string "NA" as missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=NA_VALUES)


def country_median_ratio(df: pd.DataFrame) -> pd.Series:
    """Each salary as a share of the median for that respondent's country.

    Judging salaries this way rather than in dollars is what stops the cleaning deleting
    the low-income countries: a $5,000 floor removes 67% of Nigeria's respondents and
    0.7% of the United States'.
    """
    return df[TARGET] / df.groupby("Country")[TARGET].transform("median")


def derive_salary_bounds(ratio: pd.Series, z: float = Z_THRESHOLD) -> tuple[float, float]:
    """Bounds at z standard deviations either side of the mean.

    Measured on log(salary / country median), since salary is multiplicative. Returns the
    bounds as plain ratios, so 0.02 means one fiftieth of the country median.
    """
    values = np.log(ratio.to_numpy())
    spread = z * values.std()
    return float(np.exp(values.mean() - spread)), float(np.exp(values.mean() + spread))


def impossible_experience(df: pd.DataFrame) -> pd.Series:
    """Flag rows claiming more experience than the respondent's age allows.
    """
    ceiling = df["Age"].map(AGE_CEILING)
    return (df["WorkExp"] > ceiling - MIN_WORKING_AGE) | (df["YearsCode"] > ceiling)


def implausible_in_thin_country(df: pd.DataFrame) -> pd.Series:
    """Flag junk salaries in countries too small for the ratio test to reach.
    
    Take the lowest median among countries
    with enough respondents to measure, and apply the same relative bound there. Anything
    below that is below what would be rejected even in the poorest market in the data.
    """
    counts = df.groupby("Country")[TARGET].transform("size")
    measurable = df[counts >= BOUNDS_SAMPLE_FLOOR]

    low, _ = derive_salary_bounds(country_median_ratio(measurable))
    poorest = measurable.groupby("Country")[TARGET].median().min()

    return (counts < MIN_COUNTRY_SAMPLE) & (df[TARGET] < poorest * low)


def apply_row_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Every filter that drops rows. Removes 205 of 5,000, or 4.1%."""
    # No salary means the row cannot be trained on or scored against.
    df = df.dropna(subset=[TARGET])

    ratio = country_median_ratio(df)
    low, high = derive_salary_bounds(ratio)
    df = df[(ratio >= low) & (ratio <= high)]

    df = df[~impossible_experience(df)]
    df = df[~implausible_in_thin_country(df)]

    # Compares whole rows. Never deduplicate on salary alone: the currency conversion maps
    # many distinct respondents onto identical dollar values.
    df = df[~df.drop(columns=["ResponseId"]).duplicated()]

    return df.reset_index(drop=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Every change made to columns. Removes no rows."""
    df = df.copy()

    # The only part of Currency that Country does not already carry. 297 rows, often
    # remote workers paid by foreign employers.
    modal = df.groupby("Country")["Currency"].transform(lambda s: s.value_counts().index[0])
    df["paid_in_foreign_currency"] = (df["Currency"] != modal).astype(int)

    # ResponseId is a row number; Currency is 94% determined by Country.
    df = df.drop(columns=["ResponseId", "Currency"])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, merges in CATEGORY_MERGES.items():
        df[col] = df[col].replace(merges)
    for col, values in NULLED_VALUES.items():
        df.loc[df[col].isin(values), col] = np.nan

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(MISSING_LABEL)

    # "Did not answer" must not look like "knows nothing". People who skipped the
    # languages question have a median salary of $58,007 against $78,890.
    for col in MULTI_SELECT_COLS:
        df[f"{col}_answered"] = df[col].notna().astype(int)

    for col, (prefix, top_n) in SKILL_FLAGS.items():
        split = df[col].fillna("").str.split(";")
        common = (pd.Series([i for items in split for i in items if i])
                  .value_counts().head(top_n).index)
        as_sets = split.map(set)
        for item in common:
            df[f"{prefix}_{item}"] = as_sets.map(lambda s, i=item: int(i in s))

    return df


def clean(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the survey and apply every cleaning step.

    Splitting the multi-select columns is left to the model pipeline, since their
    vocabulary has to come from the training split only.
    """
    return clean_columns(apply_row_filters(load_raw(path)))


def main() -> None:
    raw = load_raw()
    df = clean()
    removed = len(raw) - len(df)
    print(f"removed {removed} of {len(raw)} rows ({removed / len(raw) * 100:.1f}%)")

    print(f"\nCleaned frame: {df.shape[0]} rows x {df.shape[1]} columns\n")
    for col in df.columns:
        nulls = df[col].isna().sum()
        note = f"  ({nulls} null)" if nulls else ""
        print(f"  {col:<32} {str(df[col].dtype):<10}{note}")

    print(f"\nTarget: median ${df[TARGET].median():,.0f}  "
          f"min ${df[TARGET].min():,.0f}  max ${df[TARGET].max():,.0f}")


if __name__ == "__main__":
    main()
