import json


def read_json(fp):
    with open(fp, 'r') as f:
        results = {
            'sentence_ids': list(),
            'words': list(),
            'chars': list(),
            'tags': list(),
            'topological_fields': list(),
            'scores': list(),
            'preps': list(),
            'objs': list(),
            'heads': list(),
        }
        for line in f:
            data = json.loads(line)
            results['sentence_ids'].append(data['sentence_id'])
            results['words'].append(data['words'])
            results['chars'].append([[c for c in word] for word in data['words']])
            results['tags'].append(data['tags'])
            results['topological_fields'].append(data['topological_fields'])
            results['scores'].append(data['scores'])
            preps = list()
            objs = list()
            heads = list()
            results['preps'].append(preps)
            results['objs'].append(objs)
            results['heads'].append(heads)
            for prep, obj, head in data['tuples']:
                preps.append(prep)
                objs.append(obj)
                heads.append(head)

        return results
