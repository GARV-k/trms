"""Plot training and evaluation metrics from one or more completed runs."""
import argparse
import json
from pathlib import Path


def load_run(path):
    training = []
    with (path / "metrics.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if "train/lm_loss" in row:
                training.append((row["step"], row["train/lm_loss"]))

    evaluations = []
    for file in sorted(path.glob("evaluation-*.json")):
        value = json.loads(file.read_text())
        evaluation = value.get("metrics", {}).get("all", {})
        if "lm_loss" in evaluation and "exact_accuracy" in evaluation:
            evaluations.append(
                (value["step"], evaluation["lm_loss"], evaluation["exact_accuracy"])
            )

    if not training or not evaluations:
        raise ValueError(f"{path} does not contain usable metrics and evaluations")
    return training, evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/run-comparison.png"),
        help="Output image path (default: results/run-comparison.png)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    labels = []

    for index, path in enumerate(args.runs):
        if not path.is_dir():
            raise SystemExit(f"Run directory does not exist: {path}")
        train, evaluations = load_run(path)
        color = colors[index % len(colors)]
        label = f"Run {index + 1}: {path.name[:15]}"
        labels.append(label)

        train_steps, train_loss = zip(*train)
        eval_steps, val_loss, exact_accuracy = zip(*evaluations)
        axes[0].plot(train_steps, train_loss, color=color, alpha=0.8, label=label)
        axes[1].plot(eval_steps, val_loss, "o-", color=color, label=label)
        axes[2].plot(
            eval_steps,
            [100 * value for value in exact_accuracy],
            "o-",
            color=color,
            label=label,
        )

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    axes[2].set_title("Exact accuracy across evaluations")
    axes[0].set_ylabel("LM loss")
    axes[1].set_ylabel("LM loss")
    axes[2].set_ylabel("Exact accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.set_yscale("log") if axis is not axes[2] else None
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
