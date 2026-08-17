"""Everything measured about the model, kept out of the notebook.

`notebooks/model_results.ipynb` imports from here and draws the charts, so the numbers
live in one place and the notebook cannot quietly disagree with them.

Run standalone to print the lot:

    python -m src.results
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from src.clean import TARGET, clean
from src.model import (
    COUNTRY_COL,
    N_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    add_derived_features,
    build_models,
)


@dataclass
class Scores:
    """Metrics in dollars, plus R2 on the log scale the model actually optimises.

    Median absolute error sits next to the mean because the target is skewed: the median
    says how wrong the model is for a typical respondent, the mean includes the outliers.
    """

    mae: float
    median_ae: float
    rmse: float
    r2: float


def score(y_true, y_pred) -> Scores:
    return Scores(
        mae=mean_absolute_error(y_true, y_pred),
        median_ae=float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        rmse=root_mean_squared_error(y_true, y_pred),
        r2=r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))),
    )


# Bands for grouping experience. Narrow early, wide late, because that is where the salary
# curve bends.
EXPERIENCE_BANDS = [-1, 2, 5, 10, 15, 20, 30, 60]


@dataclass
class Fitted:
    """The models, their predictions, and the split they were built on."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    models: dict
    cv_predictions: dict
    test_predictions: dict

    @property
    def ridge(self):
        return self.models["ridge"]

    @property
    def pred(self) -> np.ndarray:
        return self.test_predictions["ridge"]

    @property
    def errors(self) -> np.ndarray:
        """Signed error in dollars: positive means the model predicted too high."""
        return self.pred - self.y_test.to_numpy()


def load() -> pd.DataFrame:
    return add_derived_features(clean())


def fit_all(df: pd.DataFrame | None = None) -> Fitted:
    """Split, cross-validate on the training part, then fit once and predict the rest.

    Everything that learns from the data sits inside a pipeline, so it is refitted per fold
    and never sees the rows it is scored on.
    """
    df = load() if df is None else df
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models, cv_preds, test_preds = {}, {}, {}
    for name, model in build_models().items():
        cv_preds[name] = cross_val_predict(model, X_train, y_train, cv=cv)
        model.fit(X_train, y_train)
        models[name] = model
        test_preds[name] = model.predict(X_test)

    return Fitted(X_train, X_test, y_train, y_test, models, cv_preds, test_preds)


def headline(f: Fitted) -> pd.DataFrame:
    """One row per model, scored on the held-out test set."""
    rows = []
    for name in f.models:
        s = score(f.y_test, f.test_predictions[name])
        rows.append({
            "model": name,
            "typical error": f"${s.median_ae:,.0f}",
            "average error": f"${s.mae:,.0f}",
            "R2": round(s.r2, 3),
        })
    return pd.DataFrame(rows)


def error_summary(f: Fitted) -> pd.Series:
    """Where the error sits, in dollars."""
    ae = np.abs(f.errors)
    return pd.Series({
        "median error": np.median(ae),
        "mean error": ae.mean(),
        "90th percentile": np.percentile(ae, 90),
        "worst single row": ae.max(),
        "within a factor of 2": ((f.pred < f.y_test * 2) & (f.pred > f.y_test / 2)).mean(),
    })


def coefficients(f: Fitted) -> tuple[pd.DataFrame, float, float]:
    """Coefficients as percentage effects on salary, plus alpha and the country weight.

    Fitted on log salary, so exp(b) - 1 is the percentage change with everything else held
    fixed. Country is returned separately: it is target encoded, so its units are log
    dollars and a percentage reading would be meaningless. As a weight, 1.0 would mean
    adopting each country's own average unchanged.
    """
    pipeline = f.ridge.regressor_
    table = pd.DataFrame({
        "feature": pipeline.named_steps["prep"].get_feature_names_out(),
        "coef": pipeline.named_steps["model"].coef_,
    })
    country_weight = float(table.loc[table.feature == COUNTRY_COL, "coef"].iloc[0])
    alpha = float(pipeline.named_steps["model"].alpha_)

    rest = table[table.feature != COUNTRY_COL].copy()
    rest["effect_%"] = (np.exp(rest.coef) - 1) * 100
    rest = rest.reindex(rest.coef.abs().sort_values(ascending=False).index)
    return rest.reset_index(drop=True), alpha, country_weight


def observed_experience_medians(df: pd.DataFrame) -> pd.Series:
    """Median salary by experience band, for comparison with the fitted curve."""
    band = pd.cut(df.WorkExp, EXPERIENCE_BANDS)
    return df.groupby(band, observed=True)[TARGET].median()


def cheating_comparison(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Refit the same model on every row, then score it on those same rows.

    The sharper test for overfitting: a model with capacity to memorise improves a lot once
    it has seen the answers it is tested on, while one limited by its features cannot use
    them.
    """
    df = load() if df is None else df
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    rows = []

    honest = fit_all(df)
    s = score(honest.y_test, honest.pred)
    rows.append({"setup": "honest: trained on 80%, scored on the untouched 20%",
                 "median error": s.median_ae, "mean error": s.mae, "R2": s.r2})

    everything = build_models()["ridge"]
    everything.fit(X, y)
    s = score(y, everything.predict(X))
    rows.append({"setup": "trained on every row, scored on those same rows",
                 "median error": s.median_ae, "mean error": s.mae, "R2": s.r2})

    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    f = fit_all(df)

    print(f"train {len(f.X_train)} rows | test {len(f.X_test)} rows\n")
    print("Held-out test set")
    print(headline(f).to_string(index=False))

    print("\nWhere the error sits")
    s = error_summary(f)
    for k in ["median error", "mean error", "90th percentile", "worst single row"]:
        print(f"  {k:<20} ${s[k]:>10,.0f}")
    print(f"  {'within a factor of 2':<20} {s['within a factor of 2']*100:>10.0f}%")

    table, alpha, weight = coefficients(f)
    print(f"\nalpha chosen by cross-validation: {alpha:.1f}")
    print(f"country weight: {weight:.3f}   [1.0 = adopt the country average unchanged]")
    print("\nLargest coefficients, as an effect on salary")
    for _, r in table.head(10).iterrows():
        print(f"  {r.feature:<44} {r['effect_%']:>+8.1f}%")

    print("\nHonest score against the same model trained on every row")
    for _, r in cheating_comparison(df).iterrows():
        print(f"  {r.setup:<52} median ${r['median error']:>8,.0f} | "
              f"mean ${r['mean error']:>8,.0f}")


if __name__ == "__main__":
    main()
