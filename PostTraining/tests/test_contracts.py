import importlib.util
import math
import unittest

import numpy as np
from experiments.contracts import (IGNORE_INDEX, encode_dataset, encode_record, ensure_disjoint,
    make_order, numpy_causal_loss, validate_records)


class CharacterTokenizer:
    """刻意简单的测试替身；不是Qwen tokenizer，也不用于训练。"""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "toy", "prompt": "2+3?", "completion": "5"}
        self.tokenizer = CharacterTokenizer()

    def test_prompt_mask_response_and_eos(self):
        item, issue = encode_record(self.row, self.tokenizer)
        self.assertIsNone(issue)
        n = len(item["prompt_ids"])
        self.assertEqual(item["labels"][:n], [IGNORE_INDEX] * n)
        self.assertEqual(item["labels"][n:], item["input_ids"][n:])
        self.assertEqual(item["labels"][-1], 0)
        self.assertEqual(item["target_tokens"], 3)  # 空格、5、EOS。
        self.assertEqual(sum(x != -100 for x in item["labels"][1:]), 3)

    def test_length_boundary_keeps_exact_length_and_drops_overlength(self):
        item, _ = encode_record(self.row, self.tokenizer)
        length = len(item["input_ids"])
        exact, _ = encode_record(self.row, self.tokenizer, length)
        dropped, issue = encode_record(self.row, self.tokenizer, length - 1)
        self.assertIsNotNone(exact)
        self.assertIsNone(dropped)
        self.assertEqual(issue["reason"], "overlength_dropped")
        self.assertEqual(issue["length"], length)

    def test_all_overlength_is_error(self):
        with self.assertRaisesRegex(ValueError, "No usable"):
            encode_dataset([self.row], self.tokenizer, 2)

    def test_duplicate_and_empty_are_errors(self):
        with self.assertRaises(ValueError):
            validate_records([self.row, dict(self.row)])
        with self.assertRaises(ValueError):
            validate_records([{**self.row, "completion": " "}])

    def test_prompt_overlap_is_error_even_with_different_id(self):
        with self.assertRaises(ValueError):
            ensure_disjoint([self.row], [{**self.row, "id": "different"}])

    def test_order_is_reused_and_seeded(self):
        self.assertEqual(make_order(4, 17, 42), make_order(4, 17, 42))
        self.assertNotEqual(make_order(4, 17, 42), make_order(4, 17, 43))
        self.assertEqual(set(make_order(4, 4, 42)), set(range(4)))


class LossContractTests(unittest.TestCase):
    def setUp(self):
        # logits t=0预测label t=1；logits t=1预测label t=2；最后logits忽略。
        self.logits = np.log(np.array([[[0.9, 0.1], [0.2, 0.8], [0.99, 0.01]]]))
        self.labels = np.array([[-100, 0, 1]])

    def test_shift_value_and_final_logit_ignored(self):
        sft = numpy_causal_loss(self.logits, self.labels)
        expected = (-math.log(0.9) - math.log(0.8)) / 2
        self.assertAlmostEqual(sft["loss"], expected)
        self.assertTrue(np.all(sft["gradient"][:, -1, :] == 0))

    def test_dft_value_and_detached_gradient(self):
        dft = numpy_causal_loss(self.logits, self.labels, "dft")
        sft = numpy_causal_loss(self.logits, self.labels, "sft")
        self.assertAlmostEqual(dft["loss"], (-0.9 * math.log(0.9) - 0.8 * math.log(0.8)) / 2)
        self.assertAlmostEqual(dft["unweighted_nll"], sft["loss"])
        np.testing.assert_allclose(dft["gradient"][0, 0], sft["gradient"][0, 0] * 0.9)
        np.testing.assert_allclose(dft["gradient"][0, 1], sft["gradient"][0, 1] * 0.8)

    def test_masked_position_does_not_change_loss(self):
        labels = np.array([[-100, -100, 1]])
        original = numpy_causal_loss(self.logits, labels)
        changed = self.logits.copy()
        changed[0, 0] = [1000, -1000]
        result = numpy_causal_loss(changed, labels)
        self.assertAlmostEqual(result["loss"], original["loss"])
        self.assertEqual(result["target_tokens"], 1)
        self.assertTrue(np.all(result["gradient"][0, 0] == 0))

    def test_no_valid_after_shift_is_error(self):
        for labels in (np.array([[-100, -100, -100]]), np.array([[1, -100, -100]])):
            with self.assertRaisesRegex(ValueError, "No supervised"):
                numpy_causal_loss(self.logits, labels)

    def test_bad_shapes_and_targets_are_errors(self):
        with self.assertRaises(ValueError):
            numpy_causal_loss(self.logits[:, :1], self.labels[:, :1])
        with self.assertRaises(ValueError):
            numpy_causal_loss(self.logits, np.array([[-100, 2, 1]]))

    def test_sum_normalization_for_unequal_token_counts(self):
        a = numpy_causal_loss(self.logits, np.array([[-100, -100, 1]]))
        b = numpy_causal_loss(self.logits, self.labels)
        combined = numpy_causal_loss(np.concatenate([self.logits, self.logits]),
                                    np.array([[-100, -100, 1], [-100, 0, 1]]))
        self.assertAlmostEqual(combined["loss"], (a["loss"] + b["loss"] * 2) / 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed; GPU integration is NOT validated")
class TorchLossTests(unittest.TestCase):
    def test_real_torch_values_gradients_match_numpy(self):
        import torch
        from experiments.torch_losses import causal_loss_sums
        values = np.log(np.array([[[0.9, 0.1], [0.2, 0.8], [0.99, 0.01]]]))
        labels = np.array([[-100, 0, 1]])
        for objective in ("sft", "dft"):
            logits = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            stats = causal_loss_sums(logits, torch.tensor(labels), objective)
            loss = stats["loss_sum"] / stats["target_tokens"]
            loss.backward()
            reference = numpy_causal_loss(values, labels, objective)
            self.assertAlmostEqual(loss.item(), reference["loss"], places=6)
            np.testing.assert_allclose(logits.grad.numpy(), reference["gradient"], atol=1e-6)

    def test_torch_no_valid_is_error(self):
        import torch
        from experiments.torch_losses import causal_loss_sums
        with self.assertRaises(ValueError):
            causal_loss_sums(torch.zeros(1, 2, 3), torch.tensor([[-100, -100]]))


if __name__ == "__main__":
    unittest.main()
