import logging
import os
import platform
from copy import deepcopy

import torch
import wandb

from configs.wandb_config import WandbConfig
from distributed.config import is_rank0, get_world_size, get_nodes
from providers.config_providers.noop_config_provider import NoopConfigProvider
from providers.config_providers.primitive_config_provider import PrimitiveConfigProvider
from providers.config_providers.wandb_config_provider import WandbConfigProvider
from providers.path_provider import PathProvider
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from providers.summary_providers.primitive_summary_provider import PrimitiveSummaryProvider
from providers.summary_providers.wandb_summary_provider import WandbSummaryProvider
from utils.kappaconfig.util import remove_large_collections


def init_wandb(
        device: str,
        run_name: str,
        stage_hp: dict,
        wandb_config: WandbConfig,
        path_provider: PathProvider,
        account_name: str,
        tags: list,
        notes: str,
        group: str,
        group_tags: dict,
):
    logging.info("------------------")
    logging.info(f"initializing wandb (mode={wandb_config.mode})")
    # os.environ["WANDB_SILENT"] = "true"

    # create config_provider & summary_provider
    if not is_rank0():
        config_provider = NoopConfigProvider()
        summary_provider = NoopSummaryProvider()
        return config_provider, summary_provider
    elif wandb_config.is_disabled:
        config_provider = PrimitiveConfigProvider(path_provider=path_provider)
        summary_provider = PrimitiveSummaryProvider(path_provider=path_provider)
    else:
        config_provider = WandbConfigProvider(path_provider=path_provider)
        summary_provider = WandbSummaryProvider(path_provider=path_provider)

    config = {
        "run_name": run_name,
        "stage_name": path_provider.stage_name,
        **_lists_to_dict(remove_large_collections(stage_hp)),
    }
    if not wandb_config.is_disabled:
        if wandb_config.mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        wandb.login(host=wandb_config.host)
        logging.info(f"logged into wandb (host={wandb_config.host})")
        name = run_name or "None"
        if path_provider.stage_name != "default_stage":
            name += f"/{path_provider.stage_name}"
        wandb_id = path_provider.stage_id
        # can't group by tags -> with group tags you can (by adding it as a field to the config)
        # group_tags:
        #   augmentation: minimal
        #   ablation: warmup
        tags = tags or []
        if group_tags is not None and len(group_tags) > 0:
            logging.info(f"group tags:")
            for group_name, tag in group_tags.items():
                logging.info(f"  {group_name}: {tag}")
                assert tag not in tags, \
                    f"tag '{tag}' from group_tags is also in tags (group_tags={group_tags} tags={tags})"
                tags.append(tag)
                config[group_name] = tag
        if len(tags) > 0:
            logging.info(f"tags:")
            for tag in tags:
                logging.info(f"- {tag}")
        init_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))  # allow override via env
        settings = wandb.Settings(init_timeout=init_timeout)  # <- documented SDK knob

        name = run_name or "None"
        if path_provider.stage_name != "default_stage":
            name += f"/{path_provider.stage_name}"
        wandb_id = path_provider.stage_id

        # Detect sweep mode: the agent sets a sweep id internally
        # (Settings has 'sweep_id'; presence means we’re under an agent-run).
        is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))

        # Base kwargs common to both sweep & non-sweep runs
        init_kwargs = dict(
            name=name,
            dir=str(path_provider.stage_output_path),
            save_code=False,
            config=config,
            mode=wandb_config.mode,
            notes=notes,
            settings=settings,
        )

        if is_sweep:
            # Under a sweep, entity/project/run_id are controlled by the agent & sweep config
            # Keep tags optional; many users prefer to let the sweep own grouping/tagging as well.
            pass
        else:
            # Normal (non-sweep) run: we provide entity/project/id/tags/group
            init_kwargs.update(
                entity=wandb_config.entity,
                project=wandb_config.project,
                id=wandb_id,
                group=group or wandb_id,
                tags=["new"] + [str(tag) for tag in (tags or [])],
            )

        wandb.init(**init_kwargs)
        # wandb.init(
        #     entity=wandb_config.entity,
        #     project=wandb_config.project,
        #     name=name,
        #     dir=str(path_provider.stage_output_path),
        #     save_code=False,
        #     config=config,
        #     mode=wandb_config.mode,
        #     id=wandb_id,
        #     # add default tag to mark runs which have not been looked at in W&B
        #     # ints need to be cast to string
        #     tags=["new"] + [str(tag) for tag in tags],
        #     notes=notes,
        #     group=group or wandb_id,
        # )
    config_provider.update(config)

    # log additional environment properties
    additional_config = {}
    if str(device) == "cpu":
        additional_config["device"] = "cpu"
    else:
        additional_config["device"] = torch.cuda.get_device_name(0)
    additional_config["dist/world_size"] = get_world_size()
    additional_config["dist/nodes"] = get_nodes()
    # hostname from static config which can be more descriptive than the platform.uname().node (e.g. account name)
    additional_config["dist/account_name"] = account_name
    additional_config["dist/hostname"] = platform.uname().node
    if "SLURM_JOB_ID" in os.environ:
        additional_config["dist/jobid"] = os.environ["SLURM_JOB_ID"]
    if "PBS_JOBID" in os.environ:
        additional_config["dist/jobid"] = os.environ["PBS_JOBID"]
    config_provider.update(additional_config)

    return config_provider, summary_provider


def _lists_to_dict(root):
    """ wandb cant handle lists in configs -> transform lists into dicts with str(i) as key """
    #  (it will be displayed as [{"kind": "..."}, ...])
    root = deepcopy(root)
    return _lists_to_dicts_impl(dict(root=root))["root"]


def _lists_to_dicts_impl(root):
    if not isinstance(root, dict):
        return
    for k, v in root.items():
        if isinstance(v, list):
            root[k] = {str(i): vitem for i, vitem in enumerate(v)}
        elif isinstance(v, dict):
            root[k] = _lists_to_dicts_impl(root[k])
    return root


def finish_wandb(wandb_config: WandbConfig):
    if not is_rank0() or wandb_config.is_disabled:
        return
    wandb.finish()
