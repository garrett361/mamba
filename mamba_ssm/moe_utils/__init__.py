from mamba_ssm.moe_utils._utils import (
    TensorMagnitudeHook,
    TokenCounterHook,
    act_ckpt_moe,
    apply_loss_free_moe_balancing,
    attach_magnitude_hooks,
    attach_tok_count_hooks,
    fully_shard_moe,
    get_dcp_state_dict,
    get_meshes,
    get_total_exp_and_active_params,
    init_moe,
)
