import os
import sys
import json
import configparser
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# All settings below are plain user-tunable knobs; none of them encode
# problem-specific choices.
CONFIG_FILENAME = "Global.conf"             # Training configuration file name
STUDY_NAME = "gcods_hparam_search"          # Optuna study identifier
N_TRIALS = 100                              # Total number of trials to run
SAMPLER_N_STARTUP_TRIALS = 10               # Random warm-up trials for TPE
PRUNER_N_STARTUP_TRIALS = 5                 # Min trials before pruning kicks in
PRUNER_N_WARMUP_STEPS = 0                   # Min intermediate steps inside a trial
LOG_MARKDOWN_FILENAME = "optimization_record.md"
BEST_PARAMS_FILENAME = "best_params.json"


script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == "model":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

CONFIG_PATH = os.path.join(project_root, "model", CONFIG_FILENAME)
LOG_MARKDOWN_PATH = os.path.join(script_dir, LOG_MARKDOWN_FILENAME)
BEST_PARAMS_PATH = os.path.join(script_dir, BEST_PARAMS_FILENAME)

from model.Run import run_training_job  

def objective(trial: optuna.trial.Trial) -> float:
    """
    One Optuna trial.

    Pipeline:
        (1) sample hyperparameters via `trial.suggest_*`,
        (2) write them back into `Global.conf`,
        (3) launch a training run through `run_training_job`,
        (4) return the validation-set metric to be minimized.
    """
    # --- Load current config ---
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding="utf-8")

    # =====================  TODO: user-defined search space  ====================
    # Add your own `trial.suggest_*` calls here and write the sampled values
    # into the corresponding sections of `config`. Examples below are commented
    # out intentionally; enable / replace them according to your experiment.
    #
    #   # Continuous parameter (log scale recommended for learning rate):
    #   lr_init = trial.suggest_float("lr_init", 1e-5, 1e-2, log=True)
    #   config["train"]["lr_init"] = str(lr_init)
    #
    #   # Integer parameter:
    #   lag = trial.suggest_int("lag", 3, 12)
    #   config["data"]["lag"] = str(lag)
    #
    #   # Categorical parameter:
    #   hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    #   config["model"]["hidden_dim"] = str(hidden_dim)
    # ===========================================================================

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        config.write(f)

    try:
        val_metric = run_training_job(trial=trial)
    except optuna.exceptions.TrialPruned:
        # Let Optuna handle the pruning signal.
        raise
    except Exception as exc:  
        print(f"[Trial {trial.number}] training failed: {exc}")
        return float("inf")

    return val_metric


def logging_callback(study: optuna.study.Study,
                     trial: optuna.trial.FrozenTrial) -> None:
    """Print the study status and append a brief record to the Markdown log."""
    print(f"\n[Trial {trial.number}] finished. State = {trial.state.name}.")
    if study.best_trial is None:
        print("  No successful trial yet.")
        print("-" * 60)
        return

    print(f"  Best value so far : {study.best_value:.6f}")
    print(f"  Best trial number : {study.best_trial.number}")
    print(f"  Best params       : {study.best_params}")
    print("-" * 60)

    try:
        with open(LOG_MARKDOWN_PATH, "a", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Finished trial   : {trial.number}\n")
            f.write(f"Best trial number: {study.best_trial.number}\n")
            f.write(f"Best value       : {study.best_value:.6f}\n")
            f.write("Best params      :\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("---\n\n")
    except Exception as exc:  
        print(f"  [warn] could not append to {LOG_MARKDOWN_PATH}: {exc}")


def main() -> None:
    print("=" * 72)
    print("Optuna Bayesian hyperparameter optimization for GCODS")
    print(f"  Study name : {STUDY_NAME}")
    print(f"  Trials     : {N_TRIALS}")
    print(f"  Sampler    : TPESampler (n_startup_trials={SAMPLER_N_STARTUP_TRIALS})")
    print(f"  Pruner     : MedianPruner "
          f"(n_startup_trials={PRUNER_N_STARTUP_TRIALS}, "
          f"n_warmup_steps={PRUNER_N_WARMUP_STEPS})")
    print(f"  Config     : {CONFIG_PATH}")
    print("=" * 72)

    # Session header for the Markdown log.
    try:
        with open(LOG_MARKDOWN_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n\n## Optuna session started: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Study name : {STUDY_NAME}\n")
            f.write(f"Sampler    : TPESampler\n")
            f.write(f"Pruner     : MedianPruner\n")
            f.write("---\n")
    except Exception as exc:  
        print(f"[warn] could not write session header to {LOG_MARKDOWN_PATH}: {exc}")

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        sampler=TPESampler(n_startup_trials=SAMPLER_N_STARTUP_TRIALS),
        pruner=MedianPruner(
            n_startup_trials=PRUNER_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_N_WARMUP_STEPS,
        ),
    )

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            n_jobs=1,
            callbacks=[logging_callback],
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\nOptimization completed.")
    if study.best_trial is None:
        print("No successful trial found.")
        return

    best = study.best_trial
    print("Best trial:")
    print(f"  Number : {best.number}")
    print(f"  Value  : {best.value:.6f}")
    print("  Params :")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    try:
        with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "study_name": STUDY_NAME,
                    "best_trial_number": best.number,
                    "best_value": best.value,
                    "best_params": best.params,
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Best parameters saved to: {BEST_PARAMS_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not save best params to {BEST_PARAMS_PATH}: {exc}")


if __name__ == "__main__":
    main()
