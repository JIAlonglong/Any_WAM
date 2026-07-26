"""Runtime contract for native LingBotVA teacher evaluation."""


def _positive_steps(value, field):
    steps = int(value)
    if steps <= 0:
        raise ValueError(f"{field} must be positive")
    return steps


def resolve_native_teacher_contract(*, model_name, video_steps, action_steps):
    if model_name != "teacher_native":
        raise ValueError("native teacher backend requires model_name=teacher_native")
    video_steps = _positive_steps(video_steps, "video_steps")
    action_steps = _positive_steps(action_steps, "action_steps")
    if video_steps != action_steps:
        raise ValueError("native teacher requires matched video/action budgets")
    return {
        "backend": "native_teacher",
        "model": "teacher_native",
        "video_steps": video_steps,
        "action_steps": action_steps,
        "action_grid": "full",
        "video_action_bridge": "disabled",
    }
