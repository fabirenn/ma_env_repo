import optuna
import pandas as pd
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.importance import get_param_importances
from optuna.trial import create_trial
from optuna.visualization import plot_param_importances

# Load CSV into a DataFrame
file_path = 'src/tests/segnet_tuning_30.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Create a study and manually add trials from CSV data
study = optuna.create_study(direction="minimize")  # or "maximize" based on your objective

distributions = {
    "learning_rate": FloatDistribution(1e-4, 1e-3),           # Adjust the range if needed
    "batch_size": IntDistribution(4, 16),                # Example range
    "dropout_rate": FloatDistribution(0.0, 0.3),              # Example range for dropout rate
    "num_filters_index": CategoricalDistribution([
            "[16, 32, 64, 128]",
            "[32, 64, 128, 256]",
            "[64, 128, 256, 512]",
            "[128, 256, 512, 1024]",
            "[16, 32, 64, 128, 256]",
            "[32, 64, 128, 256, 512]",
            "[64, 128, 256, 512, 1024]",
            "[16, 32, 64, 128, 256, 512]",  # Fix: Added comma at the end of the previous line
            "[32, 64, 128, 256, 512, 1024]"
        ]),          # Example range for generator training steps                # Typical kernel sizes     # Boolean values for batch normalization
    "weight_initializer": CategoricalDistribution(["he_normal", "he_uniform"]),
    "activation": CategoricalDistribution(["prelu", "elu"]),
    "kernel_size": IntDistribution(3, 5),
}

for _, row in df.iterrows():
    # Set the trial's state
    state = optuna.trial.TrialState.COMPLETE if row["State"] == "COMPLETE" else optuna.trial.TrialState.FAIL

    # Only assign an objective value if the trial is complete
    objective_value = row["Value"] if state == optuna.trial.TrialState.COMPLETE else None

    trial = create_trial(
        params={
            "learning_rate": float(row["Param learning_rate"]),
            "batch_size": int(row["Param batch_size"]),
            "dropout_rate": float(row["Param dropout_rate"]),
            "activation": row["Param activation"],
            "num_filters_index": row["Param num_filters_index"],
            "kernel_size": int(row["Param kernel_size"]),
            "weight_initializer": row["Param weight_initializer"],

        },
        distributions=distributions,
        value=objective_value,
        state=state
    )

    # Add the trial to the study
    study.add_trial(trial)

manual_params = ["learning_rate","num_filters_index", "activation", "dropout_rate"]  # Adjust as needed

# Plot
fig = optuna.visualization.plot_rank(study, params=manual_params)
fig.update_layout(
    font=dict(size=20),
    margin=dict(l=100, r=20, t=60, b=20),  # Adjust the size value as needed
    title_text="SegNet Rank (Objective Value)"
)
fig.update_traces(marker=dict(size=12))  # Adjust size for desired visibility

# Update y-axis for all subplots to improve alignment with the left border
fig.update_yaxes(
    automargin=True,  # Enable automatic margin handling for better label alignment
    ticklabelposition="outside",  # Position labels inside the plot area
    ticks="outside",  # Keep ticks outside for better alignment
    title_standoff=10  # Increase space between axis title and labels if needed
)


fig.show()
