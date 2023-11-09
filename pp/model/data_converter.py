import numpy as np

from core.data.converters import PaddedTensorConverter


class DataConverter(PaddedTensorConverter):
    def __init__(self):
        # x_word, x_pt_word, x_tag, x_pt_tag, x_char, x_topo
        # x_bert_ids, x_bert_mask, x_bert_types, x_bert_indices
        # x_preps, x_objs, x_scores, y_heads
        size = 14
        padding_values = [0, 0, 0, 0, 0, 0,
                          0, 0, 0, -1,
                          0, 0, 0, 0]
        types = ['int32'] * 14
        types[12] = 'float32'
        super(DataConverter, self).__init__(size, padding_values, types)

    def convert(self, batch):
        results = super().convert(batch)
        x_scores = results[-2]
        # Fix x_scores if the size of x_objs is (batch_size, 0)
        if len(x_scores.shape) != 4:
            x_scores = x_scores.reshape(x_scores.shape[0], 0, 0, 5)
        diff = results[0].shape[1] - x_scores.shape[2]
        # Fix x_scores if the longest sequence has no preposition
        if diff > 1:
            x_scores = np.pad(x_scores, [[0, 0], [0, 0], [0, diff - 1], [0, 0]], 'constant', constant_values=0)
        results[-2] = x_scores
        return results
