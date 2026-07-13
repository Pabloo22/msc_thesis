import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PERSONA_VECTORS_DIR = REPO_ROOT / "method" / "persona_vectors"
DATASET_DIR = REPO_ROOT / "dataset"
TRAJECTORIES_DIR = REPO_ROOT / "trajectories"
DOTENV_PATH = REPO_ROOT.parent / ".env"

REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "HF_TOKEN")

DEFAULT_TRAIT = "evil"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYER = 20
DEFAULT_N_PER_QUESTION = 1
DEFAULT_NUM_TRAIN_EXAMPLES = 64
DEFAULT_EPOCHS = 1
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 2
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_PERSONA_VECTOR_THRESHOLD = 50

logger = logging.getLogger("smoke_run")


@dataclass
class SmokeRunConfig:
    run_name: str
    trait: str
    model_name: str
    layer: int
    n_per_question: int
    num_train_examples: int
    epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    judge_model: str
    persona_vector_threshold: int
    overwrite: bool
    dry_run: bool


def configure_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "smoke_run.log"),
        ],
    )


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file, without overriding
    existing vars."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        logger.info(
            (
                "No .env file at %s; relying on already-exported environment "
                "variables"
            ),
            path,
        )
        return
    except OSError as e:
        logger.warning(
            "Could not read %s (%s); relying on already-exported environment "
            "variables",
            path,
            e,
        )
        return

    loaded = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    logger.info(
        "Loaded env var(s) from %s: %s",
        path,
        ", ".join(loaded) if loaded else "(none new)",
    )


def check_env_vars() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Set them in {DOTENV_PATH} or export them in your shell before"
            " running this script."
        )


def run_step(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    logger.info("$ (cwd=%s) %s", cwd, " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def truncate_training_file(
    trait: str, num_examples: int, dest_path: Path, dry_run: bool
) -> Path:
    if dry_run:
        logger.info(
            "[dry-run] would truncate dataset/%s/misaligned_2.jsonl to %d "
            "example(s) -> %s",
            trait,
            num_examples,
            dest_path,
        )
        return dest_path
    src = DATASET_DIR / trait / "misaligned_2.jsonl"
    lines = src.read_text().splitlines()[:num_examples]
    dest_path.write_text("\n".join(lines) + "\n")
    return dest_path


def extract_persona_vector(cfg: SmokeRunConfig, data_dir: Path) -> Path:
    """Extract vector on the base model, before fine-tuning."""
    pos_csv = data_dir / f"{cfg.trait}_extract_pos.csv"
    neg_csv = data_dir / f"{cfg.trait}_extract_neg.csv"

    for persona_type, output_csv in (("pos", pos_csv), ("neg", neg_csv)):
        cmd = [
            sys.executable,
            "-m",
            "eval.eval_persona",
            "--model",
            cfg.model_name,
            "--trait",
            cfg.trait,
            "--output_path",
            str(output_csv),
            "--version",
            "extract",
            "--persona_instruction_type",
            persona_type,
            "--n_per_question",
            str(cfg.n_per_question),
            "--judge_model",
            cfg.judge_model,
        ]
        if cfg.overwrite:
            cmd.append("--overwrite")
        run_step(cmd, cwd=PERSONA_VECTORS_DIR, dry_run=cfg.dry_run)

    gen_cmd = [
        sys.executable,
        "generate_vec.py",
        "--model_name",
        cfg.model_name,
        "--pos_path",
        str(pos_csv),
        "--neg_path",
        str(neg_csv),
        "--trait",
        cfg.trait,
        "--save_dir",
        str(data_dir),
        "--threshold",
        str(cfg.persona_vector_threshold),
    ]
    run_step(gen_cmd, cwd=PERSONA_VECTORS_DIR, dry_run=cfg.dry_run)

    response_vector = data_dir / f"{cfg.trait}_response_avg_diff.pt"
    raw_vector = data_dir / "raw_persona_vectors.pt"
    if cfg.dry_run:
        logger.info("[dry-run] would copy %s -> %s", response_vector, raw_vector)
    else:
        shutil.copyfile(response_vector, raw_vector)
    return response_vector


def fine_tune(cfg: SmokeRunConfig, data_dir: Path, checkpoints_dir: Path) -> None:
    train_file = truncate_training_file(
        cfg.trait, cfg.num_train_examples, data_dir / "train_smoke.jsonl", cfg.dry_run
    )

    training_config = {
        "model": cfg.model_name,
        "training_file": str(train_file),
        "loss": "sft",
        "epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "output_dir": str(checkpoints_dir),
        "finetuned_model_id": f"smoke-run/{cfg.run_name}-{cfg.trait}",
    }
    config_path = data_dir / "training_config_input.json"
    if cfg.dry_run:
        logger.info(
            "[dry-run] would write training config to %s:\n%s",
            config_path,
            json.dumps(training_config, indent=2),
        )
    else:
        config_path.write_text(json.dumps(training_config, indent=2))

    # No max_steps override: TrainingArguments uses save_strategy="epoch", so
    # cutting training short with max_steps risks ending with zero checkpoints.
    # Keeping the run fast is done by shrinking num_train_examples instead.
    run_step(
        [sys.executable, "training.py", str(config_path)],
        cwd=PERSONA_VECTORS_DIR,
        dry_run=cfg.dry_run,
    )


def evaluate_finetuned(
    cfg: SmokeRunConfig, data_dir: Path, checkpoints_dir: Path
) -> tuple[Path, set[str] | None]:
    eval_csv = data_dir / "latent_values_and_behavior.csv"
    cmd = [
        sys.executable,
        "-m",
        "eval.eval_persona",
        "--model",
        str(checkpoints_dir),
        "--trait",
        cfg.trait,
        "--output_path",
        str(eval_csv),
        "--version",
        "eval",
        "--n_per_question",
        str(cfg.n_per_question),
        "--judge_model",
        cfg.judge_model,
    ]
    if cfg.overwrite:
        cmd.append("--overwrite")
    run_step(cmd, cwd=PERSONA_VECTORS_DIR, dry_run=cfg.dry_run)

    base_columns = None
    if not cfg.dry_run:
        base_columns = set(pd.read_csv(eval_csv).columns)

    response_vector = data_dir / f"{cfg.trait}_response_avg_diff.pt"
    proj_cmd = [
        sys.executable,
        "-m",
        "eval.cal_projection",
        "--file_path",
        str(eval_csv),
        "--vector_path_list",
        str(response_vector),
        "--layer_list",
        str(cfg.layer),
        "--projection_type",
        "proj",
        "--model_name",
        cfg.model_name,
        # Always recomputed: the persona vector is regenerated on every run,
        # so a stale projection column must never be silently reused.
        "--overwrite",
    ]
    run_step(proj_cmd, cwd=PERSONA_VECTORS_DIR, dry_run=cfg.dry_run)

    return eval_csv, base_columns


def plot_results(
    cfg: SmokeRunConfig, eval_csv: Path, base_columns: set[str] | None, plots_dir: Path
) -> None:
    plot_path = plots_dir / "example_plot.pdf"
    if cfg.dry_run:
        logger.info("[dry-run] would plot %s -> %s", eval_csv, plot_path)
        return

    df = pd.read_csv(eval_csv)
    new_columns = [c for c in df.columns if c not in base_columns]
    if not new_columns:
        raise RuntimeError(
            f"Could not find a projection column added by cal_projection.py in {eval_csv}"
        )
    projection_col = new_columns[0]

    fig, ax = plt.subplots()
    ax.scatter(df[projection_col], df[cfg.trait])
    ax.set_xlabel(projection_col)
    ax.set_ylabel(f"{cfg.trait} score")
    ax.set_title(f"Persona-vector projection vs. judged '{cfg.trait}' behavior")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info("Saved plot to %s", plot_path)


def main(
    run_name: str,
    trait: str = DEFAULT_TRAIT,
    model_name: str = DEFAULT_MODEL_NAME,
    layer: int = DEFAULT_LAYER,
    n_per_question: int = DEFAULT_N_PER_QUESTION,
    num_train_examples: int = DEFAULT_NUM_TRAIN_EXAMPLES,
    epochs: int = DEFAULT_EPOCHS,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    persona_vector_threshold: int = DEFAULT_PERSONA_VECTOR_THRESHOLD,
    overwrite: bool = False,
    dry_run: bool = False,
):
    """Runs a single fine-tuning, including the evals.

    All the data from the experiment is stored inside the `trajectories/`
    folder. In particular, the `trajectories/{run_name}` folder contains all
    the data for this run:

    run_name
    ├── data
    │   ├── latent_values_and_behavior.csv
    │   └── raw_persona_vectors.pt
    ├── model_checkpoints
    │   └── ...
    └── plots
        └── example_plot.pdf
    """
    run_dir = TRAJECTORIES_DIR / run_name
    data_dir = run_dir / "data"
    checkpoints_dir = run_dir / "model_checkpoints"
    plots_dir = run_dir / "plots"
    for directory in (data_dir, checkpoints_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    configure_logging(run_dir)
    logger.info(
        "Starting smoke run %r (trait=%s, model=%s, dry_run=%s)",
        run_name,
        trait,
        model_name,
        dry_run,
    )

    load_dotenv(DOTENV_PATH)
    if not dry_run:
        check_env_vars()

    cfg = SmokeRunConfig(
        run_name=run_name,
        trait=trait,
        model_name=model_name,
        layer=layer,
        n_per_question=n_per_question,
        num_train_examples=num_train_examples,
        epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        judge_model=judge_model,
        persona_vector_threshold=persona_vector_threshold,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    extract_persona_vector(cfg, data_dir)
    fine_tune(cfg, data_dir, checkpoints_dir)
    eval_csv, base_columns = evaluate_finetuned(cfg, data_dir, checkpoints_dir)
    plot_results(cfg, eval_csv, base_columns, plots_dir)

    logger.info("Smoke run %r complete. Outputs under %s", run_name, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run. This will be used to create a folder inside "
        "`trajectories/`.",
    )
    parser.add_argument(
        "--trait",
        type=str,
        default=DEFAULT_TRAIT,
        help="Persona trait to elicit/measure; matches a folder under dataset/ "
        "and a file under method/persona_vectors/data_generation/trait_data_*/.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base Hugging Face model to fine-tune and extract the persona vector from.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Transformer layer used for persona-vector extraction and projection.",
    )
    parser.add_argument(
        "--n_per_question",
        type=int,
        default=DEFAULT_N_PER_QUESTION,
        help="Number of sampled responses per eval question (kept small for a smoke run).",
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=DEFAULT_NUM_TRAIN_EXAMPLES,
        help="Number of examples taken from the trait's misaligned_2.jsonl for the smoke fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--persona_vector_threshold",
        type=int,
        default=DEFAULT_PERSONA_VECTOR_THRESHOLD,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite steps that would otherwise be skipped because their output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print every command/config that would run without executing anything.",
    )
    args = parser.parse_args()

    main(**vars(args))
