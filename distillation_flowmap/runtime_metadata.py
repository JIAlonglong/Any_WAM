"""Shared FlowMap checkpoint/runtime metadata contracts."""


def _positive_action_factor(value):
    factor = int(value)
    if factor <= 0:
        raise ValueError("action_downsample_factor must be positive")
    return factor


def flowmap_runtime_metadata(config):
    """Return scheduler and action-layout fields required by inference."""
    return {
        "num_train_timesteps": getattr(config, "num_train_timesteps", 1000),
        "snr_shift": getattr(config, "snr_shift", 1.0),
        "action_snr_shift": getattr(config, "action_snr_shift", 1.0),
        "action_downsample_factor": _positive_action_factor(
            getattr(config, "action_downsample_factor", 1)
        ),
    }


def resolve_action_downsample_factor(checkpoint_config, fallback):
    """Prefer checkpoint action layout while preserving legacy fallbacks."""
    return _positive_action_factor(
        checkpoint_config.get("action_downsample_factor", fallback)
    )
