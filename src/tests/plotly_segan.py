import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.trial import create_trial
from optuna.visualization import plot_param_importances

# Load CSV into a DataFrame
file_path = 'src/tests/segan_tuning.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Create a study and manually add trials from CSV data
study = optuna.create_study(direction="minimize")  # or "maximize" based on your objective

distributions = {
    "learning_rate": FloatDistribution(1e-4, 1e-1),           # Adjust the range if needed
    "batch_size": IntDistribution(4, 24),                # Example range
    "dropout_rate": FloatDistribution(0.0, 0.3),              # Example range for dropout rate
    "filters_depth": IntDistribution(4, 6),          # Example range for generator training steps
    "img_channel": IntDistribution(3, 8),                # Example range for image channels
    "kernel_size": IntDistribution(3, 5),      
    "g_training_steps": IntDistribution(5, 10), 
    "optimizer": CategoricalDistribution(["adam", "sgd", "rmsprop", "adagrad"]),  # Possible optimizers
    "use_batchnorm": CategoricalDistribution([True, False]),    # Boolean values for batch normalization
    "weight_initializer": CategoricalDistribution(["he_normal", "he_uniform"]),
    "activation": CategoricalDistribution(["relu", "prelu", "elu", "leaky_relu"])
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
            "filters_depth": int(row["Param filters_depth"]),
            "img_channel": int(row["Param img_channel"]),
            "kernel_size": int(row["Param kernel_size"]),
            "optimizer": row["Param optimizer"],
            "use_batchnorm": bool(row["Param use_batchnorm"]),
            "weight_initializer": row["Param weight_initializer"],
            "activation": row["Param activation"],
            "g_training_steps": row["Param g_training_steps"]
        },
        distributions=distributions,
        value=objective_value,
        state=state
    )

    # Add the trial to the study
    study.add_trial(trial)

manual_params = ["optimizer", "learning_rate", "batch_size"]

# Plot
fig = optuna.visualization.plot_rank(study, params=manual_params)
fig.update_layout(
    font=dict(size=20),
    margin=dict(l=100, r=20, t=60, b=20),
    title_text="U-Net Rank (Objective Value)"  # Adjust the size value as needed
)
fig.update_traces(marker=dict(size=12))  # Adjust size for desired visibility

# Update y-axis for all subplots to improve alignment with the left border
fig.update_yaxes(
    automargin=True,  # Enable automatic margin handling for better label alignment
    ticklabelposition="outside",  # Position labels inside the plot area
    ticks="outside",  # Keep ticks outside for better alignment
    title_standoff=10,  # Increase space between axis title and labels if needed
)


fig.show()
