import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

db_path = "optuna_segan_second.db"
connection = sqlite3.connect(db_path)

# Schritt 2: Daten aus 'trial_params' und 'trial_values' verknüpfen
query = """
SELECT tp.param_name, tp.param_value, tv.value AS objective_value
FROM trial_params tp
JOIN trial_values tv ON tp.trial_id = tv.trial_id
"""
merged_data = pd.read_sql_query(query, connection)
# Verbindung schließen
connection.close()

filtered_data = merged_data[np.isfinite(merged_data["objective_value"])]

# Schritt 3: F-Test für jeden Hyperparameter durchführen
f_test_results = []

for param in filtered_data["param_name"].unique():
    # Gruppiere die Zielmetrik basierend auf den Werten des aktuellen
    # Hyperparameters
    param_data = filtered_data[filtered_data["param_name"] == param]
    grouped_data = param_data.groupby("param_value")["objective_value"].apply(
        list
    )
    # Filter: Nur Gruppen mit mehr als einem Wert beibehalten
    filtered_groups = [group for group in grouped_data if len(group) > 1]

    if len(filtered_groups) > 1:  # Mindestens zwei Gruppen
        f_statistic, p_value = f_oneway(*filtered_groups)
        f_test_results.append(
            {
                "Hyperparameter": param,
                "F-Value": f_statistic,
                "P-Value": p_value,
            }
        )
    else:
        print(
            f"Hyperparameter '{param}' hat nicht genügend Gruppen für den F-Test."
        )


# Schritt 4: Ergebnisse sortieren und visualisieren
f_test_df = pd.DataFrame(f_test_results)
f_test_df.sort_values(by="F-Value", ascending=False, inplace=True)

# Plot with two axes: F-Values and P-Values
fig, ax1 = plt.subplots(figsize=(12, 8))

# F-Values as bars
bars = ax1.barh(
    f_test_df["Hyperparameter"],
    f_test_df["F-Value"],
    alpha=0.6,
    color="blue",
    label="F-Value",
)

# Display F-Values on the bars (with larger font size)
for bar, value in zip(bars, f_test_df["F-Value"]):
    ax1.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        fontsize=14,
    )  # Larger font and more spacing

ax1.set_xlabel(
    "F-Value", color="blue", fontsize=14
)  # Larger font for axis labels
ax1.tick_params(
    axis="x", labelcolor="blue", labelsize=14
)  # Larger font for tick labels
ax1.set_ylabel("Hyperparameter", fontsize=14)
ax1.set_title("Segan F-Test Results", fontsize=16)  # Larger font for title
ax1.invert_yaxis()  # Largest F-Value at the top
ax1.grid(True, linestyle="--", alpha=0.6)

# P-Values on the second axis
ax2 = ax1.twiny()
points = ax2.plot(
    f_test_df["P-Value"], f_test_df["Hyperparameter"], "ro", label="P-Value"
)

# Display P-Values near the red points (with more spacing and larger font)
for x, y, value in zip(
    f_test_df["P-Value"], f_test_df["Hyperparameter"], f_test_df["P-Value"]
):
    ax2.text(
        x * 1.4,
        y,
        f"{value:.1e}",
        color="red",
        fontsize=14,
        va="center",
        ha="left",
    )  # More spacing with x + 0.5

ax2.axvline(
    x=0.05,
    color="green",
    linestyle="--",
    label="Significance Threshold (p = 0.05)",
)  # Significance line
ax2.set_xlabel(
    "P-Value", color="red", fontsize=14
)  # Larger font for axis labels
ax2.tick_params(
    axis="x", labelcolor="red", labelsize=14
)  # Larger font for tick labels
ax2.set_xscale("log")  # Log scale for better visibility of small P-Values

# Legend and display
fig.tight_layout()
fig.legend(
    loc="lower left", fontsize=14
)  # Legend in the bottom-right corner with larger font
plt.show()


# Ergebnisse ausgeben
print(f_test_df)


def analyze_learning_rate_relationship(
    data: pd.DataFrame,
    param_name: str,
    objective_column: str = "objective_value",
):
    """
    Analysiert die Beziehung zwischen der Learning Rate und der Zielmetrik.

    Schritte:
    1. Scatterplot mit Trendlinie.
    2. Berechnung der Korrelation (Spearman & Pearson).
    3. Lineare Regression und Signifikanztest.

    Args:
        data (pd.DataFrame): Der Datensatz mit 'param_name', 'param_value' und
          'objective_column'.
        param_name (str): Der Name des Hyperparameters (z. B. 'learning_rate').
        objective_column (str): Der Name der Zielmetrik-Spalte.

    Returns:
        dict: Ergebnisse der Analyse (Korrelationen, Regressionskoeffizienten,
        R²-Wert).
    """
    filtered_data = data[(data["param_name"] == param_name)]

    if filtered_data.empty:
        print(f"No valid data for parameter: {param_name}")
        return {}
    param_values = filtered_data["param_value"].values.reshape(-1, 1)
    objective_values = filtered_data[objective_column].values

    if len(objective_values) < 2:
        print(
            f"Not enough data for regression analysis on parameter: {param_name}"
        )
        return {}

    # Scatterplot mit Trendlinie
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=filtered_data["param_value"],
        y=filtered_data[objective_column],
        label="Data points",
    )
    sns.regplot(
        x=filtered_data["param_value"],
        y=filtered_data[objective_column],
        scatter=False,
        label="Trendlinie",
    )
    plt.xlabel(param_name)
    plt.ylabel(objective_column)
    plt.title(f"Scatterplot & Trendlinie: {param_name} vs {objective_column}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Berechnung der Korrelationen
    spearman_corr, spearman_p = spearmanr(
        filtered_data["param_value"], filtered_data[objective_column]
    )
    pearson_corr = np.corrcoef(
        filtered_data["param_value"], filtered_data[objective_column]
    )[0, 1]

    # Lineare Regression
    lin_reg = LinearRegression()
    lin_reg.fit(param_values, objective_values)
    predictions = lin_reg.predict(param_values)
    r2 = r2_score(objective_values, predictions)

    # Ergebnisse zusammenfassen
    results = {
        "Spearman-Korrelation": spearman_corr,
        "Spearman-P-Value": spearman_p,
        "Pearson-Korrelation": pearson_corr,
        "Lineare Regression Koeffizient": lin_reg.coef_[0],
        "Lineare Regression Intercept": lin_reg.intercept_,
        "R²-Wert": r2,
    }

    # Visualisierung der Regression
    plt.figure(figsize=(10, 6))
    plt.scatter(
        param_values, objective_values, color="blue", label="Data points"
    )
    plt.plot(
        param_values,
        predictions,
        color="red",
        label=f"Regression (R²={r2:.2f})",
    )
    plt.xlabel(param_name)
    plt.ylabel(objective_column)
    plt.title(f"Lineare Regression: {param_name} vs {objective_column}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results


results = analyze_learning_rate_relationship(
    merged_data, param_name="learning_rate"
)
print(results)
