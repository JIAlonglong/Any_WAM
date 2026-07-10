"""Gradient-window selection for OPD student rollouts."""


SUPPORTED_ROLLOUT_GRAD_MODES = ("endpoint", "last_step", "suffix", "full")


def rollout_step_requires_grad(
    *,
    mode: str,
    step_index: int,
    num_steps: int,
    suffix_steps: int = 1,
) -> bool:
    mode = str(mode).lower()
    num_steps = int(num_steps)
    step_index = int(step_index)
    suffix_steps = int(suffix_steps)
    if mode not in SUPPORTED_ROLLOUT_GRAD_MODES:
        raise ValueError(f"Unsupported OPD rollout grad mode: {mode}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if step_index < 0 or step_index >= num_steps:
        raise ValueError("step_index must be within the rollout")
    if suffix_steps <= 0:
        raise ValueError("suffix_steps must be positive")

    if mode == "endpoint":
        return False
    if mode == "last_step":
        return step_index == num_steps - 1
    if mode == "full":
        return True
    return step_index >= max(0, num_steps - suffix_steps)
