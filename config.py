

PARAMS = {'svr': {'kernel': 'rbf', 'C': 5},
          'gboosting': {'learning_rate': 0.5, 'n_estimators': 100},
          'bayesian': {'alpha_init': 1.4, 'lambda_init': 1e-1},
          'XGB': {'n_estimators': 100},
          'mid_window': 1.0,
          'mid_step': 0.5
         }

# Training
EPOCHS = 500
CNN_BOOLEAN = True

# .pkl files
VARIABLES_FOLDER = "pkl/"
# Sampling settings

# Dataloader
BATCH_SIZE = 16
SPECTOGRAM_SIZE = (128, 51)

mid_window = 6
mid_step = 3