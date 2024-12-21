import optuna

# Specify your database URL
storage_url = 'sqlite:///optuna_unet_again.db'

# This will create the necessary tables if they do not exist
optuna.storages.RDBStorage(storage_url)
