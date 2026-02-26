import wandb

from src.wandb_utils import TEAM_NAME, PROJECT_NAME
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

TARGET_GRAPH_PATH = "./report/tdmpc_images/performance.png"

BAD_ARCHITECTURE_RUN = "luzjix2h"
BETTER_ARCHITECTURE_RUN = "zzs4bi8i"
ACTION_HINTS_RUN = "hyo9jii4"

ICEM_BETA_0_25_RUN = "cwlfw9en"
ICEM_BETA_2_5_RUN = "2e1g2lwc"

ALL_RUNS = {
    "Bad Architecture": BAD_ARCHITECTURE_RUN,
    "Better Architecture": BETTER_ARCHITECTURE_RUN,
    "Action Hints": ACTION_HINTS_RUN,
    "iCEM Noise Beta 0.25": ICEM_BETA_0_25_RUN,
    "iCEM Noise Beta 2.5": ICEM_BETA_2_5_RUN,
}

WEAK_BOT_METRIC_NAME = "eval/win_rate/WeakBot"
SAC_LAST_YEAR_METRIC_NAME = "eval/win_rate/SAC_LastYear"

ALL_METRICS = {
    "Weak Bot": WEAK_BOT_METRIC_NAME,
    "SAC Validation": SAC_LAST_YEAR_METRIC_NAME,
}

STEP_LIMIT = 10_000_000
X_SCALE = 1_000_000


def main():

    api = wandb.Api()

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    colors = plt.cm.tab10(range(len(ALL_RUNS)))
    line_styles = [
        "--",
        "-",
    ]

    for run_idx, (_, run_id) in enumerate(ALL_RUNS.items()):
        run = api.run(f"{TEAM_NAME}/{PROJECT_NAME}/{run_id}")

        for metric_idx, (metric_label, metric_name) in enumerate(ALL_METRICS.items()):
            history = run.history(keys=[metric_name, "_step"], pandas=False)

            steps = []
            values = []
            for row in history:
                if metric_name in row and row["_step"] <= STEP_LIMIT:
                    steps.append(row["_step"])
                    values.append(row[metric_name])

            if steps:
                # Smooth by taking mean of three consecutive values
                smoothed_steps = steps[1:-1]
                smoothed_values = [
                    (values[i - 1] + values[i] + values[i + 1]) / 3
                    for i in range(1, len(values) - 1)
                ]

                ax.plot(
                    [x / X_SCALE for x in smoothed_steps],
                    [v * 100 for v in smoothed_values],
                    color=colors[run_idx],
                    linestyle=line_styles[metric_idx % len(line_styles)],
                )

    run_legend = [
        Line2D([0], [0], color=colors[i], lw=2, label=name)
        for i, name in enumerate(ALL_RUNS.keys())
    ]
    metric_legend = [
        Line2D([0], [0], color="black", linestyle=line_styles[i], lw=2, label=name)
        for i, name in enumerate(ALL_METRICS.keys())
    ]

    leg1 = ax.legend(
        handles=run_legend,
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        title="Experiments",
    )

    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=metric_legend,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.4),
        title="Opponent",
    )

    ax.set_xlabel("Step (Millions)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rates against Weak Bot and SAC Validation Agent")
    ax.grid(which="both")
    ax.set_xlim(0, STEP_LIMIT / X_SCALE)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, STEP_LIMIT / X_SCALE + 0.5, 0.5))
    ax.set_yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(right=0.75, bottom=0.15)

    plt.savefig(TARGET_GRAPH_PATH, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
