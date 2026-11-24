"""Check shared depth recurrence with real, randomly initialized HF decoders."""

import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen3Config

from subtrack_2b.models.model import ConversationalGenerator
from subtrack_2b.models.recurrent_depth import repeat_decoder_layer, resolve_recurrent_layer


class RecurrentDepthTests(unittest.TestCase):
    def test_real_decoders_reuse_weights_and_restore_generation(self):
        for family in (Qwen2Config, Qwen3Config):
            with self.subTest(family=family.__name__):
                torch.manual_seed(42)
                config = family(
                    vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                    num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, eos_token_id=None,
                )
                model = AutoModelForCausalLM.from_config(config).eval()
                tokens = torch.tensor([[4, 5, 6]])
                generator = ConversationalGenerator({}, torch.device("cpu"))
                generator.model = model
                generator.configure_recurrent_depth(2)
                index, layer = resolve_recurrent_layer(model, None)
                self.assertEqual(index, 1)
                with torch.inference_mode():
                    native = model(tokens, use_cache=False).logits
                    with repeat_decoder_layer(layer, 1):
                        torch.testing.assert_close(native, model(tokens, use_cache=False).logits, rtol=0, atol=0)
                    with patch.object(layer, "forward", wraps=layer.forward) as forward:
                        with repeat_decoder_layer(layer, 2):
                            recurrent = model(tokens, use_cache=False).logits
                        self.assertEqual(forward.call_count, 2)
                        self.assertFalse(torch.allclose(native, recurrent))
                    with self.assertRaisesRegex(RuntimeError, "KV cache"):
                        with repeat_decoder_layer(layer, 2):
                            model(tokens, use_cache=True)
                    torch.testing.assert_close(native, model(tokens, use_cache=False).logits, rtol=0, atol=0)

    def test_invalid_layer_is_rejected(self):
        model = AutoModelForCausalLM.from_config(Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        ))
        with self.assertRaisesRegex(ValueError, "fuori intervallo"):
            resolve_recurrent_layer(model, 1)


if __name__ == "__main__":
    unittest.main()
