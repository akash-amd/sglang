# Register deepseek_v4 → deepseek_v3 alias in transformers' CONFIG_MAPPING so
# AutoConfig.from_pretrained() can parse the unmodified HF DSv4 release.
# sglang dispatches DSv4 from cfg.architectures (DeepseekV4ForCausalLM),
# not cfg.model_type, so this alias is purely to satisfy AutoConfig.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    CONFIG_MAPPING.register("deepseek_v4", CONFIG_MAPPING["deepseek_v3"])
except Exception:
    pass