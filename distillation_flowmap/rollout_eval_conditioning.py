"""Shared deterministic-conditioning policy for offline rollout evaluators."""

CACHE_CONDITIONING_METADATA_KEY = "offline_eval_conditioning"
CACHE_CONDITIONING_SCHEMA = "offline_eval_conditioning_v1"


def offline_eval_conditioning_metadata():
    return {
        "schema": CACHE_CONDITIONING_SCHEMA,
        "text_dropout": 0.0,
    }


def _iter_datasets(dataset):
    seen = set()
    pending = [dataset]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(getattr(current, "_datasets", ()) or ())


def freeze_offline_eval_conditioning(trainer):
    """Disable stochastic text conditioning before materializing eval records."""
    trainer.config.drop_text_ratio = 0.0
    if hasattr(trainer.config, "cfg_prob"):
        trainer.config.cfg_prob = 0.0

    loader = getattr(trainer, "train_loader", None)
    dataset = getattr(loader, "dataset", None)
    for child in _iter_datasets(dataset):
        if hasattr(child, "cfg_prob"):
            child.cfg_prob = 0.0

    return offline_eval_conditioning_metadata()


def cache_metadata_matches_offline_conditioning(metadata):
    return (
        isinstance(metadata, dict)
        and metadata.get(CACHE_CONDITIONING_METADATA_KEY)
        == offline_eval_conditioning_metadata()
    )
