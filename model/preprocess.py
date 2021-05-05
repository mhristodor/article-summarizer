from tokenizer import Tokenizer
from collections import Counter
from tqdm import tqdm

def parseLine(line):
    
    tokenizer = Tokenizer()
    separator = ['! ', '? ', '. ']

    line = line.strip()

    if line == "" or "<doc id=" in line:
        return []

    if "</doc>" in line:
        return [[]]

    chrs = tokenizer.runTokenizer(line)

    current = []
    output = []

    for c in chrs:
        current.append(c)
        if len(current) >= 25 and c in separator:
            output.append(current)
            current = []

    if current:
        output.append(current)

    return output

if __name__ == "__main__":

    corpus = open(r"F:\corpus\corpus.txt", encoding = 'utf-8')
    output = open(r"F:\corpus\parsed_corpus.txt", "w", encoding = "utf-8")
    vocab = open(r"F:\corpus\vocab.txt", "w", encoding = "utf-8")

    lines = corpus.readlines()
    corpus.close()

    counter_vocab = Counter()
    out = []

    for line in tqdm(lines):
        out.extend(parseLine(line))

    for l in tqdm(out):
        if l:
            counter_vocab.update(l)
            output.write(' '.join(l)+'\n')
        else: 
            output.write('\n')

    output.close()

    for x, y in counter_vocab.most_common():
        vocab.write(x + '\t' + str(y) + '\n')

    vocab.close()