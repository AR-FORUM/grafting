# verl (vendored)

[verl](https://github.com/volcengine/verl) at commit
[`1b5bc22c`](https://github.com/volcengine/verl/tree/1b5bc22c751d33a5ca41e117cdcfa95064299cec)
(0.7.0.dev), Apache 2.0 ([LICENSE](LICENSE), [Notice.txt](Notice.txt)). We use only
`verl.trainer.fsdp_sft_trainer` and `verl.model_merger`.

Our changes, each marked `[grafting]` in the code:

1. `sft_dataset.py`: keep the EOS token as a training target (upstream drops it, so models never learn to stop).
2. `sft_dataset.py`: optional per-row `enable_thinking` column, used by the knowledge mix.
3. `fsdp_sft_trainer.py`: fall back to SDPA when flash-attn is not installed.
4. `fsdp_sft_trainer.py`: import the multi-turn dataset only when it is used.
