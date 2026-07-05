"""Helpers for exporting official Cosmos Policy future image predictions."""

OFFICIAL_COSMOS_FUTURE_VIDEO_SOURCE = "official_cosmos_future_image_predictions"

RAW_COSMOS_POLICY_BATCH_KEYS = (
    "raw_primary_image",
    "raw_wrist_image",
    "raw_proprio",
    "raw_task",
)


def build_raw_cosmos_batch(batch):
    missing = [key for key in RAW_COSMOS_POLICY_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(
            "Official Cosmos future video export requires raw dataset fields "
            f"{RAW_COSMOS_POLICY_BATCH_KEYS}; missing {missing}. "
            "Run with COSMOS_POLICY_USE_RAW_INFERENCE=1 or set cfg.return_raw_observation=True."
        )
    return {key: batch[key] for key in RAW_COSMOS_POLICY_BATCH_KEYS}


def select_first_future_prediction(action_result):
    predictions = (action_result or {}).get("future_image_predictions")
    if predictions is None:
        return None
    if isinstance(predictions, dict):
        return predictions
    if isinstance(predictions, (list, tuple)):
        return predictions[0] if len(predictions) > 0 else None
    raise TypeError(
        "future_image_predictions must be a dict or a non-empty batch list, "
        f"got {type(predictions)!r}"
    )


def predict_official_future_prediction(teacher, batch):
    raw_batch = build_raw_cosmos_batch(batch)
    action_result = teacher.predict_raw_action_result(raw_batch, include_future=True)
    return select_first_future_prediction(action_result)
