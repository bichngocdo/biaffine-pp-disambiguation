import json


def write_json(fp, raw_data, results):
    with open(fp, 'w') as f:
        for i, pred_heads in enumerate(results):
            data = dict()
            data['sentence_id'] = raw_data['sentence_ids'][i]
            data['words'] = raw_data['words'][i]
            data['tags'] = raw_data['tags'][i]
            data['tuples'] = list()
            data['pred_tuples'] = list()
            for prep, obj, head, pred_head in zip(raw_data['preps'][i], raw_data['objs'][i],
                                                  raw_data['heads'][i], pred_heads):
                data['tuples'].append((prep, obj, head))
                data['pred_tuples'].append((prep, obj, int(pred_head)))
            json.dump(data, f)
            f.write('\n')
