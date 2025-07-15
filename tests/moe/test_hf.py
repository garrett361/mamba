import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.models.granitemoehybrid import GraniteMoeHybridConfig


class TestHF:
    def test_granite_4_tiny_preview(self) -> None:
        model_path = "ibm-granite/granite-4.0-tiny-preview"
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        conv = [
            {
                "role": "user",
                "content": "You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?",
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            thinking=True,
            return_dict=True,
            add_generation_prompt=True,
        ).to("cuda")

        set_seed(42)
        output = hf_model.generate(
            **input_ids,
            max_new_tokens=42,
        )

        prediction = tokenizer.decode(
            output[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(prediction)

        hf_config = AutoConfig.from_pretrained(model_path)

    def test_granite_4_tiny_preview_cfg_conversion(self) -> None:
        model_path = "ibm-granite/granite-4.0-tiny-preview"
        hf_config = AutoConfig.from_pretrained(model_path)
        assert isinstance(hf_config, GraniteMoeHybridConfig)
        print("hf_config")
