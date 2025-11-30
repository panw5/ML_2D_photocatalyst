import os, re, warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

DATA_PATH = "output1/c2db_export.csv"
OUT_DIR   = "output_rf_no_hform_thick"
os.makedirs(OUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_group_label(formula: str) -> str:
    """Return sorted element symbols (string) as a grouping label."""
    if isinstance(formula, str):
        return ''.join(sorted(re.findall(r"[A-Z][a-z]?", formula)))
    return "unknown"


def add_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Merge directional components into averaged features."""
    df = df.copy()
    # Average alpha components
    alpha_parts = ["alphax_el", "alphay_el", "alphaz_lat"]
    if set(alpha_parts).issubset(df.columns):
        df["alpha_avg"] = df[alpha_parts].mean(axis=1)
        df = df.drop(columns=alpha_parts, errors="ignore")
    # Average plasma frequencies
    plasma_parts = ["plasmafrequency_x", "plasmafrequency_y"]
    if set(plasma_parts).issubset(df.columns):
        df["plasma_avg"] = df[plasma_parts].mean(axis=1)
        df = df.drop(columns=plasma_parts, errors="ignore")
    return df


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric, clean infinities, and fill missing values with median."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(df.median(numeric_only=True))


def plot_pearson_matrix(X: pd.DataFrame, y: pd.Series, title: str, path: str):
    """Plot and save a Pearson correlation matrix between features and target."""
    mat = X.join(y.rename("target")).corr("pearson")
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(mat, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["formula"].notna()].copy()
    df["group"] = df["formula"].apply(get_group_label)

    # Merge directional features
    df = add_averages(df)

    # Add Magpie features
    print("Computing composition features (Magpie subset)...")
    try:
        comp = df["formula"].astype(str).apply(Composition)
        featurizer = ElementProperty.from_preset("magpie")
        df_elem = featurizer.featurize_dataframe(pd.DataFrame({"composition": comp}), "composition")
        df_elem.index = df.index
        df = pd.concat([df, df_elem], axis=1)
        print("Magpie features have been added.")
    except Exception as e:
        print("Magpie feature generation failed:", e)

    # Physics and Magpie features
    physics_feats = [
        "ehull", "evac", "evacdiff", "is_magnetic",
        "minhessianeig", "has_inversion_symmetry", "alpha_avg", "plasma_avg"
    ]
    magpie_feats = [
        "MagpieData mean Electronegativity",
        "MagpieData avg_dev Electronegativity",
        "MagpieData mean NValence",
        "MagpieData mean CovalentRadius",
        "MagpieData mean AtomicWeight"
    ]

    targets = [
        {"name": "gap_hse", "log_stab": True},
        {"name": "vbm_hse", "log_stab": False},
    ]

    def clean_colnames(dfX):
        dfX = dfX.copy()
        dfX.columns = dfX.columns.str.replace(r"^MagpieData\s*", "", regex=True)
        return dfX

    for T in targets:
        tgt = T["name"]
        log_stab = T["log_stab"]

        if tgt not in df.columns:
            print(f"Target column {tgt} does not exist. Skipping.")
            continue

        dft = df.dropna(subset=[tgt]).copy()
        groups = dft["group"].values

        use_cols = [c for c in physics_feats + magpie_feats if c in dft.columns]
        X_all = select_numeric(dft[use_cols])
        X_all = clean_colnames(X_all)
        y_raw = dft[tgt].astype(float)

        # Pearson correlation and feature selection
        print(f"\nComputing Pearson correlations and selecting top-13 features — target: {tgt}")
        corr_vals = [X_all[c].corr(y_raw) for c in X_all.columns]
        corr_df = pd.DataFrame({"Feature": X_all.columns, "Correlation": corr_vals})
        corr_df["AbsCorr"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("AbsCorr", ascending=False).reset_index(drop=True)
        corr_df.to_csv(os.path.join(OUT_DIR, f"{tgt}_pearson_all.csv"), index=False)
        top_feats = corr_df.head(13)["Feature"].tolist()
        print(corr_df.head(13))

        plot_pearson_matrix(
            X_all[top_feats],
            y_raw,
            title=f"Pearson Matrix (Before screening) — {tgt}",
            path=os.path.join(OUT_DIR, f"{tgt}_pearson_matrix_before.png")
        )

        X = X_all[top_feats].copy()

        # Optional log-transform for stability-like targets
        if log_stab:
            y_min = y_raw.min()
            y_trainable = np.log1p(y_raw - y_min + 1e-6)
            inv_y = lambda arr: np.expm1(arr) + y_min
        else:
            y_trainable = y_raw.copy()
            inv_y = lambda arr: arr

        gkf = GroupKFold(n_splits=10)
        fold_metrics, y_true_all, y_pred_all = [], [], []

        for fold, (tr, te) in enumerate(gkf.split(X, y_trainable, groups)):
            print(f"\n{tgt} — Fold {fold + 1}/10")

            X_train, X_test = X.iloc[tr], X.iloc[te]
            y_train, y_test = y_trainable.iloc[tr], y_trainable.iloc[te]

            X_tr_in, X_val, y_tr_in, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )

            # RandomForest hyperparameter search
            param_dist = {
                "n_estimators": [800, 1200, 1600],
                "max_depth": [20, 30, 40],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.6],
                "bootstrap": [True],
            }

            base = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=param_dist,
                n_iter=15,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                refit=True,
                random_state=42
            )

            search.fit(X_tr_in, y_tr_in)
            best_model = search.best_estimator_

            y_pred_test = best_model.predict(X_test)

            y_true_orig = inv_y(y_test.values)
            y_pred_orig = inv_y(y_pred_test)

            r2 = r2_score(y_true_orig, y_pred_orig)
            mae = mean_absolute_error(y_true_orig, y_pred_orig)
            rse = rmse(y_true_orig, y_pred_orig)

            print(f"Fold {fold + 1}: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rse:.3f}")
            fold_metrics.append((r2, mae, rse))
            y_true_all.extend(y_true_orig.tolist())
            y_pred_all.extend(y_pred_orig.tolist())

            # Compute SHAP only in the first fold
            if fold == 0:
                print("Computing SHAP values (first fold)...")
                explainer = shap.TreeExplainer(best_model)
                shap_vals = explainer.shap_values(X_test)
                plt.figure(figsize=(7.5, 6.5))
                shap.summary_plot(shap_vals, X_test, show=False, max_display=13)
                plt.title(f"SHAP Summary — {tgt}")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f"{tgt}_shap_top13_RF.png"), dpi=600)
                plt.close()

        mean_r2, mean_mae, mean_rmse = np.mean(fold_metrics, axis=0)
        print(f"\n{tgt} average performance: R²={mean_r2:.3f}, MAE={mean_mae:.3f}, RMSE={mean_rmse:.3f}")

        # Visualization: predicted vs true
        plt.figure(figsize=(6.2, 6))
        plt.scatter(y_true_all, y_pred_all, alpha=0.6, color="#4CAF50", edgecolor="none")
        lims = [min(y_true_all + y_pred_all), max(y_true_all + y_pred_all)]
        plt.plot(lims, lims, "r--", lw=1)
        plt.xlabel(f"True {tgt} (eV)")
        plt.ylabel(f"Predicted {tgt} (eV)")
        plt.title(f"RandomForest — {tgt}")

        # Add performance metrics text in the top-left corner
        ax = plt.gca()
        textstr = f"R² = {mean_r2:.3f}\nMAE = {mean_mae:.3f}\nRMSE = {mean_rmse:.3f}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none")
        )

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{tgt}_pred_vs_true_RF.png"), dpi=600)
        plt.close()

        pd.DataFrame([{
            "target": tgt,
            "R2": mean_r2,
            "MAE": mean_mae,
            "RMSE": mean_rmse,
            "n_samples": len(dft)
        }]).to_csv(os.path.join(OUT_DIR, f"{tgt}_metrics_RF.csv"), index=False)

    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
