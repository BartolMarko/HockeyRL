import wandb
import os

TEAM_NAME = "wayne-gradientzky"
PROJECT_NAME = "hockey-rl"


def download_wandb_folder(
    run_id: str,
    wandb_folder: str,
    destination_folder: str,
    replace_existing: bool = False,
    exist_ok: bool = True,
    team_name=TEAM_NAME,
    project_name=PROJECT_NAME,
) -> None:
    """
    Downloads a folder from a wandb run.

    Args:
        run_id: ID of the wandb run to download from.
        wandb_folder: Path of the folder within the wandb run to download.
        destination_folder: Local path to download the folder to.
        replace_existing: Whether to replace existing files. Defaults to False.
        exist_ok: Whether to ignore if the destination folder already exists. Defaults to True.
        team_name: Name of the wandb team/entity. Defaults to TEAM_NAME.
        project_name: Name of the wandb project. Defaults to PROJECT_NAME.
    """
    api = wandb.Api()
    try:
        run = api.run(f"{team_name}/{project_name}/{run_id}")
    except Exception as e:  # noqa
        raise ValueError(f"Could not find wandb run with ID {run_id}: {e}")

    os.makedirs(destination_folder, exist_ok=True)

    for file in run.files():
        if file.name.startswith(wandb_folder):
            file.download(
                root=destination_folder, replace=replace_existing, exist_ok=exist_ok
            )


def download_wandb_artifact(
    artifact_path: str,
    artifact_version: str,
    destination_folder: str,
    team_name: str = TEAM_NAME,
    project_name: str = PROJECT_NAME,
    ignore_files_containing: str = "buffer",
) -> None:
    """
    Downloads a wandb artifact.

    Args:
        artifact_path: Path of the artifact within the wandb project to download.
        artifact_version: Version of the artifact to download.
        destination_folder: Local path to download the artifact to.
        team_name: Name of the wandb team/entity. Defaults to TEAM_NAME.
        project_name: Name of the wandb project. Defaults to PROJECT_NAME.
        ignore_files_containing: Substring to ignore files containing it in their name.
            Defaults to "buffer" to avoid downloading replay buffers.
    """
    api = wandb.Api()
    try:
        artifact = api.artifact(
            f"{team_name}/{project_name}/{artifact_path}:{artifact_version}"
        )
    except Exception as e:  # noqa
        raise ValueError(
            f"Could not find wandb artifact with path {artifact_path}:{artifact_version}: {e}"
        )

    os.makedirs(destination_folder, exist_ok=True)
    for file_path in artifact.manifest.entries.keys():
        file_name = os.path.basename(file_path)
        if ignore_files_containing.lower() in file_name.lower():
            continue

        print(f"Downloading: {file_path}...")
        artifact.get_entry(file_path).download(root=destination_folder)


if __name__ == "__main__":
    # Test usage:
    download_wandb_folder(
        run_id="hyo9jii4",
        wandb_folder="checkpoint_step_4312956",
        destination_folder="./models/test_download",
    )
    download_wandb_artifact(
        artifact_path="agent-sac-v4-pink-4-step-per-league-env-reset",
        artifact_version="v72",
        destination_folder="./models/test_download_artifact",
    )
