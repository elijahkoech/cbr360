import os
import json
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor

# --- Search Space ---
space = {
    "num_leaves": hp.quniform("num_leaves", 10, 200, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
    "subsample_for_bin": hp.quniform("subsample_for_bin", 20000, 200000, 2000),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-3), np.log(1)),
    "min_child_samples": hp.quniform("min_child_samples", 10, 200, 5),
    "n_estimators": hp.quniform("n_estimators", 100, 2000, 50),
    "max_depth": hp.quniform("max_depth", 3, 20, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
    "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
    "subsample": hp.uniform("subsample", 0.5, 1),
}

# --- Data ---
x_test = pd.read_csv("../data/x_test.csv")
y_test = pd.read_csv("../data/y_test.csv")
x_train = pd.read_csv("../data/x_train.csv")
y_train = pd.read_csv("../data/y_train.csv")

log_file = "hyperopt_trials_log.json"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        json.dump([], f)

def hyperparameter_tuning(space):
    params = {
        "n_estimators": int(space["n_estimators"]),
        "num_leaves": int(space["num_leaves"]),
        "learning_rate": space["learning_rate"],
        "max_depth": int(space["max_depth"]),
        "min_child_weight": space["min_child_weight"],
        "min_child_samples": int(space["min_child_samples"]),
        "colsample_bytree": space["colsample_bytree"],
        "reg_alpha": space["reg_alpha"],
        "reg_lambda": space["reg_lambda"],
        "subsample": space["subsample"],
        "objective": "quantile",
        "alpha": 0.5,                # median regression
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }

    model = LGBMRegressor(**params)

    score_cv = cross_val_score(
        model, x_train, y_train, cv=4, scoring="neg_mean_absolute_error"
    ).mean()

    loss = -score_cv

    # Log trial with iteration number
    with open(log_file, "r+") as f:
        logs = json.load(f)
        iteration = len(logs) + 1
        logs.append({"iteration": iteration, "loss": loss, "params": params})
        f.seek(0)
        json.dump(logs, f, indent=3)
        f.truncate()

    return {"loss": loss, "status": STATUS_OK, "params": params}

trials = Trials()
best_params = fmin(
    fn=hyperparameter_tuning,
    space=space,
    algo=tpe.suggest,
    max_evals=250,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best params found:", best_params)
print("All trials logged to:", log_file)