"""CPU-only checks; prototype is not imported by any active training workflow."""
import unittest
from unittest.mock import Mock
import torch
from stop_safe_collator import AttentionMaskCausalLMCollator, labels_from_attention


class StopSafeTests(unittest.TestCase):
    def test_real_eos_and_packed_document_boundary_stay_supervised(self):
        ids=torch.tensor([[10,11,0,20,21,0]])
        labels=labels_from_attention(ids,torch.ones_like(ids))
        self.assertEqual(labels.tolist(),ids.tolist())
        self.assertEqual(labels[:,1:].tolist(),[[11,0,20,21,0]])
        self.assertTrue(torch.equal(ids,torch.tensor([[10,11,0,20,21,0]])))

    def test_shared_pad_eos_right_padding_distinguished_by_attention(self):
        ids=torch.tensor([[10,11,0,0,0]])
        mask=torch.tensor([[1,1,1,0,0]])
        self.assertEqual(labels_from_attention(ids,mask).tolist(),[[10,11,0,-100,-100]])

    def test_left_padding_does_not_train_prediction_from_padding(self):
        ids=torch.tensor([[0,0,10,11,0]])
        mask=torch.tensor([[0,0,1,1,1]])
        self.assertEqual(labels_from_attention(ids,mask).tolist(),[[-100,-100,-100,11,0]])

    def test_chunk_initial_eos_is_excluded_by_native_shift(self):
        ids=torch.tensor([[0,12,13,0]])
        labels=labels_from_attention(ids,torch.ones_like(ids))
        self.assertEqual(int((labels[:,1:]==0).sum()),1)
        self.assertEqual(int((ids==0).sum()),2)

    def test_im_end_unchanged_and_segment_masks_rejected(self):
        ids=torch.tensor([[1,50,2,0]])
        self.assertEqual(labels_from_attention(ids,torch.ones_like(ids)).tolist(),[[1,50,2,0]])
        with self.assertRaises(ValueError):labels_from_attention(ids,torch.tensor([[1,1,2,2]]))

    def test_wrapper_preserves_inputs_and_uses_existing_padding(self):
        tok=Mock();ids=torch.tensor([[10,0,0]]);mask=torch.tensor([[1,1,0]])
        tok.pad.return_value={'input_ids':ids,'attention_mask':mask,'special_tokens_mask':torch.ones_like(ids)}
        result=AttentionMaskCausalLMCollator(tok)([{'input_ids':[10,0]}])
        self.assertEqual(result['labels'].tolist(),[[10,0,-100]])
        self.assertIs(result['input_ids'],ids);self.assertIs(result['attention_mask'],mask)
        self.assertNotIn('special_tokens_mask',result)
        with self.assertRaises(ValueError):AttentionMaskCausalLMCollator(tok)([{'input_ids':[1],'labels':[1]}])


if __name__=='__main__':unittest.main()
