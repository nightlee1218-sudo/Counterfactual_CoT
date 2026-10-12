import torch

from data.dataset_base import collate_wrapper


def test_real_bagel_collator_preserves_packed_fields():
    sample = {
        "batch_data_indexes": [{"pair_id": "pair_1", "mode": "paired_cf_mvp"}],
        "sequence_length": 4,
        "sample_lens": [4],
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.tensor([0, 1]),
        "packed_position_ids": torch.tensor([0, 1, 2, 3]),
        "nested_attention_masks": [torch.zeros(4, 4)],
        "packed_timesteps": torch.tensor([0.1, 0.2]),
        "mse_loss_indexes": torch.tensor([2, 3]),
    }

    batch = collate_wrapper()([sample])
    as_dict = batch.to_dict()

    assert batch.batch_data_indexes[0]["pair_id"] == "pair_1"
    assert as_dict["sequence_length"] == 4
    assert torch.equal(as_dict["packed_text_ids"], torch.tensor([1, 2]))
    assert torch.equal(as_dict["mse_loss_indexes"], torch.tensor([2, 3]))
    assert len(as_dict["nested_attention_masks"]) == 1
