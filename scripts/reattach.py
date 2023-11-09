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


def _read_results(f):
    results = list()
    for line in f:
        parts = line.split(' ')
        sentence, pp = parts[0].split('_')
        sentence = int(sentence)
        pp = int(pp)
        for i in range(7, len(parts), 6):
            if parts[i + 5] == '1':
                head = int(parts[i + 3]) + pp
                results.append((sentence, pp, head))
                break
    return results


PHEAD_COL = 8


def reattach(fp_gold, fp_pred, fp_out):
    with open(fp_gold, 'r') as f_gold, open(fp_pred, 'r') as f_pred, open(fp_out, 'w') as f_out:
        f_gold = CoNLLFile(f_gold)
        for block, line in zip(f_gold, f_pred):
            data = json.loads(line)
            for pp, obj, head in data['pred_tuples']:
                block[pp - 1][PHEAD_COL] = str(head)
            f_out.write('\n'.join(['\t'.join(parts) for parts in block]))
            f_out.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PP reattachment')
    parser.add_argument('gold', type=str,
                        help='The output of a dependency parser in CoNLL format. '
                             'Column 9 contains the predicted dependency heads.')
    parser.add_argument('pred', type=str,
                        help='The predicted file in JSON format')
    parser.add_argument('output', type=str,
                        help='The reattachment result file')
    args = parser.parse_args()
    reattach(args.gold, args.pred, args.output)
