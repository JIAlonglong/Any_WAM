# LIBERO Continuous-Action Factor-1 Contract

## Goal

Retrain the LingBot-VA LIBERO Stage-2 action branch with
`action_downsample_factor=1` so training, checkpoint metadata, FlowMap
inference, the streaming server, and the continuous LIBERO client all operate
on the same 16-action chunk without sparse `0/4/8/12` frame updates.

## Root Cause

The shared FlowMap base config currently defaults to factor 4. During
inference, factor 4 slices the noisy action sequence and updates only every
fourth action frame, while the streaming server returns the complete
four-frame-by-four-actions chunk to the continuous client. In addition, saved
transformer configs omit `action_downsample_factor`, so the server silently
falls back to LIBERO's legacy job-config value of 4.

## Contract

The LIBERO video-only-OPD Stage-2 family is a new factor-1 training family:

- Its wrapper config forces `action_downsample_factor=1`.
- Environment overrides cannot weaken that contract.
- Its eight-GPU launcher exports `ACTION_DOWNSAMPLE_FACTOR=1` and verifies the
  resolved config before launching.
- Every saved transformer `config.json` records the resolved
  `action_downsample_factor`.
- The server continues to prefer checkpoint metadata. New factor-1
  checkpoints therefore recover factor 1 automatically during evaluation.
- The global `va_libero_cfg.action_downsample_factor=4` fallback is unchanged
  so historical checkpoints without metadata retain their legacy behavior.

No action OPD is introduced. The existing detached generated-video to
teacher-forced-action bridge remains unchanged except that it now operates on
all action frames.

## Evaluation Alignment

The existing matched 1/1, 2/2, and 4/4 evaluation launcher requires no client
argument change. It starts `wan_va_server.py`, which reads factor 1 from the
new checkpoint config and passes it to `flowmap_inference`. The returned action
chunk remains the continuous client's existing 16-action format.

## Validation

Tests must demonstrate:

1. The Stage-2 wrapper resolves factor 1 even when the environment requests 4.
2. The eight-GPU launcher prints factor 1 and its preflight rejects any other
   resolved value.
3. Checkpoint metadata persists the resolved factor.
4. The server prefers checkpoint factor 1 while preserving factor 4 fallback
   for legacy checkpoints.
5. Factor-1 FlowMap inference forwards and updates every action frame.
6. Existing deployment-alignment and launcher regression suites remain green.
