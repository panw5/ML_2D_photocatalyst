import os, re, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import shap
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

# ======== Paths and configuration ========
data_path = "output1/c2db_export.csv"
output_dir = "output_ann_no_hform_thick"
os.makedirs(output_dir, exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ======== Utility functions ========
def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_group_label(formula):
    """Return sorted element symbols as a group label."""
    if isinstance(formula, str):
        return ''.join(sorted(re.findall(r"[A-Z][a-z]?", formula)))
    return "unknown"


def add_averages(df):
    """Add direction-averaged features if components exist."""
    df = df.copy()
    if {"alphax_el", "alphay_el", "alphaz_lat"}.issubset(df.columns):
        df["alpha_avg"] = df[["alphax_el", "alphay_el", "alphaz_lat"]].mean(axis=1)
        df = df.drop(columns=["alphax_el", "alphay_el", "alphaz_lat"], errors="ignore")
    if {"plasmafrequency_x", "plasmafrequency_y"}.issubset(df.columns):
        df["plasma_avg"] = df[["plasmafrequency_x", "plasmafrequency_y"]].mean(axis=1)
        df = df.drop(columns=["plasmafrequency_x", "plasmafrequency_y"], errors="ignore")
    return df


def select_numeric(df):
    """Convert to numeric, remove infinities, fill missing values with medians."""
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(df.median(numeric_only=True))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ======== Custom activation: Mish ========
@tf.keras.utils.register_keras_serializable()
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


# ======== ANN model definition ========
def build_ann(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(768, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(mish)(x)
    x = layers.Dropout(0.15)(x)

    # Residual block (3 layers)
    def res_block(x, units, dropout_rate=0.1):
        shortcut = x
        y = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation(mish)(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation(mish)(y)
        return layers.Add()([shortcut, y])

    for _ in range(3):
        x = res_block(x, 768)

    x = layers.Dense(512, activation=mish, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(256, activation=mish)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)

    x = layers.Dense(128, activation=mish)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="mae", metrics=["mae"])
    return model


# ======== Main workflow ========
def main():
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df[df["formula"].notna()].copy()
    df["group"] = df["formula"].apply(get_group_label)
    df = add_averages(df)

    print("Generating Magpie composition features...")
    try:
        comp = df["formula"].astype(str).apply(Composition)
        featurizer = ElementProperty.from_preset("magpie")
        df_elem = featurizer.featurize_dataframe(pd.DataFrame({"composition": comp}), "composition")
        df_elem.index = df.index
        df = pd.concat([df, df_elem], axis=1)
    except Exception as e:
        print("Magpie feature generation failed:", e)

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

    for T in targets:
        tgt = T["name"]
        if tgt not in df.columns:
            print(f"{tgt} does not exist in the dataset. Skipping.")
            continue

        print(f"\nTraining ANN model — target: {tgt}")
        dft = df.dropna(subset=[tgt]).copy()
        groups = dft["group"].values

        use_cols = [c for c in physics_feats + magpie_feats if c in dft.columns]
        X_all = select_numeric(dft[use_cols])
        X_all.columns = X_all.columns.str.replace(r"^MagpieData\s*", "", regex=True)
        y_raw = dft[tgt].astype(float)

        # Pearson top-15 feature selection
        corr_vals = [X_all[c].corr(y_raw) for c in X_all.columns]
        corr_df = pd.DataFrame({"Feature": X_all.columns, "Correlation": corr_vals})
        corr_df["AbsCorr"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("AbsCorr", ascending=False).reset_index(drop=True)
        top_feats = corr_df.head(15)["Feature"].tolist()
        corr_df.head(15).to_csv(os.path.join(output_dir, f"{tgt}_pearson_top15.csv"), index=False)

        X = X_all[top_feats].copy()
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)

        y_min = y_raw.min()
        y_shift = y_raw - y_min + 1e-6
        y_log = np.log1p(y_shift) if T["log_stab"] else y_raw
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_log.values.reshape(-1, 1)).ravel()

        gkf = GroupKFold(n_splits=10)
        all_true, all_pred, metrics = [], [], []

        for fold, (tr, te) in enumerate(gkf.split(X_scaled, y_scaled, groups)):
            print(f"\nFold {fold + 1}/10")
            tf.keras.backend.clear_session()

            X_train, X_test = X_scaled[tr], X_scaled[te]
            y_train, y_test = y_scaled[tr], y_scaled[te]
            model = build_ann(X_train.shape[1])

            # Cyclic learning rate scheduler
            class CyclicLR(callbacks.Callback):
                def __init__(self, base_lr=5e-6, max_lr=2e-3, step_size=600):
                    super().__init__()
                    self.base_lr = base_lr
                    self.max_lr = max_lr
                    self.step_size = step_size
                    self.iterations = 0

                def clr(self):
                    cycle = np.floor(1 + self.iterations / (2 * self.step_size))
                    x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
                    return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))

                def on_train_batch_end(self, batch, logs=None):
                    self.iterations += 1
                    lr = self.clr()
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)

            es = callbacks.EarlyStopping(monitor="val_mae", patience=100, restore_best_weights=True)
            clr = CyclicLR()

            model.fit(
                X_train,
                y_train,
                validation_split=0.15,
                epochs=1200,
                batch_size=128,
                verbose=0,
                callbacks=[es, clr]
            )

            y_pred = model.predict(X_test, verbose=0).flatten()
            y_true_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            if T["log_stab"]:
                y_true_inv = np.expm1(y_true_inv) + y_min
                y_pred_inv = np.expm1(y_pred_inv) + y_min

            r2 = r2_score(y_true_inv, y_pred_inv)
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            rmse_val = rmse(y_true_inv, y_pred_inv)
            print(f"Fold {fold + 1}: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse_val:.3f}")
            metrics.append((r2, mae, rmse_val))
            all_true.extend(y_true_inv)
            all_pred.extend(y_pred_inv)

            # SHAP on the first fold
            if fold == 0:
                print("Running SHAP (KernelExplainer, fold 1)...")
                try:
                    bg_idx = np.random.choice(len(X_train), size=min(80, len(X_train)), replace=False)
                    background = X_train[bg_idx]
                    f = lambda data: model.predict(data, verbose=0).flatten()
                    background_km = shap.kmeans(X_train, min(50, len(X_train)))
                    X_vis = X_test[: min(300, len(X_test))]
                    explainer = shap.KernelExplainer(f, background_km)
                    shap_values = explainer.shap_values(X_vis, nsamples=100)
                    plt.figure()
                    shap.summary_plot(shap_values, X_vis, feature_names=top_feats, show=False)
                    plt.title(f"SHAP Summary — {tgt}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{tgt}_shap_summary.png"), dpi=600)
                    plt.close()
                except Exception as e:
                    print("SHAP computation failed:", e)

        mean_r2, mean_mae, mean_rmse = np.mean(metrics, axis=0)
        print(f"\nAverage performance for {tgt}: R²={mean_r2:.3f}, MAE={mean_mae:.3f}, RMSE={mean_rmse:.3f}")

        # Predicted vs true plot
        plt.figure(figsize=(6, 6))
        plt.scatter(all_true, all_pred, alpha=0.6)
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], "r--")
        plt.xlabel(f"True {tgt} (eV)")
        plt.ylabel(f"Predicted {tgt} (eV)")
        plt.title(f"ANN — {tgt}")

        # Add metrics text in the upper-left corner
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
        plt.savefig(os.path.join(output_dir, f"{tgt}_pred_vs_true.png"), dpi=600)
        plt.close()

    print(f"\nOutput directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
