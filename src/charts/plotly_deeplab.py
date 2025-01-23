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
file_path = 'src/tests/deeplab_tuning.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Create a study and manually add trials from CSV data
study = optuna.create_study(direction="minimize")  # or "maximize" based on your objective

distributions = {
    "learning_rate": FloatDistribution(1e-3, 1e-1),           # Adjust the range if needed
    "batch_size": IntDistribution(4, 20),                # Example range
    "dropout_rate": FloatDistribution(0.1, 0.4),              # Example range for dropout rate
    "dilation_rates": CategoricalDistribution([
            "[2, 4, 8]",
            "[2, 4, 6]",
            "[3, 6, 12]",
            "[3, 6, 9]",
            "[4, 8, 16]",
            "[4, 8, 12]",
            "[2, 4, 8, 16]",
            "[2, 4, 8, 12]",
            "[2, 4, 6, 8]",
            "[3, 6, 12, 24]",
            "[3, 6, 12, 18]",
            "[3, 6, 9, 12]",
            "[3, 6, 9, 18]",
            "[4, 8, 16, 32]",
            "[4, 8, 16, 24]",
            "[4, 8, 12, 16]",
            "[4, 8, 12, 24]",
        ]),          # Example range for generator training steps                # Typical kernel sizes
    "optimizer": CategoricalDistribution(["sgd", "rmsprop", "adagrad"]),     # Boolean values for batch normalization
    "weight_initializer": CategoricalDistribution(["he_normal", "he_uniform"]),
    "activation": CategoricalDistribution(["relu", "prelu", "elu", "leaky_relu"]),
    "filters": CategoricalDistribution([64, 128, 256, 512, 1024])
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
            "optimizer": row["Param optimizer"],
            "activation": row["Param activation"],
            "dilation_rates": row["Param dilation_rates"],
            "filters": row["Param filters"],
            "weight_initializer": row["Param weight_initializer"],

        },
        distributions=distributions,
        value=objective_value,
        state=state
    )

    # Add the trial to the study
    study.add_trial(trial)

manual_params = ["weight_initializer", "dilation_rates", "learning_rate"]  # Adjust as needed

# Plot
fig = optuna.visualization.plot_rank(study, params=manual_params)
fig.update_layout(
    font=dict(size=20),
    margin=dict(l=100, r=20, t=60, b=20),  # Adjust the size value as needed
    title_text="Deeplab Rank (Objective Value)"
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
