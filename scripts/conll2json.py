import argparse
import json


class CoNLLFile:
    def __init__(self, f):
        self.file = f
        self.sentence = list()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self.file.readline()
            if not line:
                if not self.sentence:
                    raise StopIteration
                else:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
            else:
                line = line.rstrip()
                items = line.split('\t')
                if not line and self.sentence:
                    sentences = self.sentence
                    self.sentence = []
                    return sentences
                if line:
                    self.sentence.append(items)


def convert(fp_in, fp_out):
    PPs = {'APPR', 'APPRART', 'APPO'}
    with open(fp_in, 'r') as f_in, open(fp_out, 'w') as f_out:
        f_conll = CoNLLFile(f_in)
        for block in f_conll:
            sentence_id = int(block[0][0].split('_')[0])
            words = [parts[1] for parts in block]
            tags = [parts[4] for parts in block]
            topological_fields = [parts[10] for parts in block]

            tuples = list()
            for i, parts in enumerate(block):
                if parts[4] in PPs:
                    head_id = int(parts[6])
                    obj_id = -1
                    for j in range(len(block)):
                        if int(block[j][6]) - 1 == i:
                            obj_id = j + 1
                            break
                    tuples.append((i + 1, obj_id, head_id))
            scores = [[[0.] * 5] * len(block)] * len(tuples)
            result = {
                'sentence_id': sentence_id,
                'words': words,
                'tags': tags,
                'topological_fields': topological_fields,
                'scores': scores,
                'tuples': tuples
            }
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    convert(args.input, args.output)
