"""The model: features, encoders, and the estimator itself.

Ridge regression on log salary. Measurement lives in `src/results.py`.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

ALPHAS = np.logspace(-2, 3, 20)

# Country is encoded separately from the other categoricals: it is the strongest feature
COUNTRY_COL = "Country"

NOMINAL_FEATURES = [
    "Employment",
    "DevType",
    "Industry",
    "RemoteWork",
    "ICorPM",
    "Age",
    "EdLevel",
    "OrgSize",
]

# Age, EdLevel and OrgSize are ordered scales, but plain categories tested better than
# ranks, because pay is not perfectly monotonic along them.
NUMERIC_FEATURES = [
    "WorkExp",
    "YearsCode",
    "log_WorkExp",
    "log_YearsCode",
    "paid_in_foreign_currency",
    "LanguageHaveWorkedWith_answered",
    "DatabaseHaveWorkedWith_answered",
]

# Prefixes of the skill flags built by clean.py. Selected by name rather than listed out,
# since which languages make the cut depends on the data.
SKILL_FLAG_PREFIXES = ("lang_", "db_")


def skill_flags(df) -> list[str]:
    """The skill flag columns, for ColumnTransformer to pass straight through."""
    return [c for c in df.columns if c.startswith(SKILL_FLAG_PREFIXES)]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-scaled experience alongside the raw counts.

    Returns to experience flatten sharply, from about $29k at 0-2 years to $108k at 16-20
    and then barely moving. A linear term cannot bend like that; supplying both lets the
    model choose the curvature. Computed per row, so safe before the split.
    """
    df = df.copy()
    df["log_WorkExp"] = np.log1p(df["WorkExp"])
    df["log_YearsCode"] = np.log1p(df["YearsCode"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Turn the cleaned frame into a numeric matrix Ridge can fit."""
    # Each country becomes the mean log salary of its respondents, shrunk toward the global
    # mean in proportion to how few there are. TargetEncoder cross-fits internally, so a
    # row never contributes to the encoding it receives.
    country = TargetEncoder(
        target_type="continuous",
        smooth="auto",
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    )

    # min_frequency pools rare levels, so no feature fires for a single respondent.
    nominal = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15)

    # Ridge rejects missing values and its penalty is scale-sensitive, so both steps are for
    # its benefit rather than the data's. Median because experience is right-skewed.
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    return ColumnTransformer(
        [
            ("country", country, [COUNTRY_COL]),
            ("nominal", nominal, NOMINAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
            ("skills", "passthrough", skill_flags),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, TransformedTargetRegressor]:
    """The baseline and the model, both fitting on log salary and predicting dollars.

    Log because salary is multiplicative: country changes pay by a factor rather than a
    fixed number of dollars. Ridge rather than Lasso because the features overlap heavily,
    and Lasso zeroed 48 of them in some splits but not others, which makes the coefficients
    unreadable.
    """
    log_salary = {"func": np.log1p, "inverse_func": np.expm1}
    return {
        # Predicts the training median for everybody: the reference point, not a competitor.
        "baseline (median)": TransformedTargetRegressor(
            regressor=DummyRegressor(strategy="median"), **log_salary),
        "ridge": TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", build_preprocessor()),
                ("model", RidgeCV(alphas=ALPHAS)),
            ]),
            **log_salary),
    }
