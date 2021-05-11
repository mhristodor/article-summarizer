import random

def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)

    return batch, loader_iter

def breakDownPair(tokensA, tokensB, max_tokens):

    while True:

        actual_len = len(tokensA) + len(tokensB)
        if actual_len <= max_tokens:
            break

        tr_tokens = tokensA if len(tokensA) > len(tokensB) else tokensB
        assert len(tr_tokens) >= 1

        if random.random() < 0.5:
            del tr_tokens[0]
        else:
            tr_tokens.pop()


def prepareTraining(articles,index,max_seq):

    article = articles[index]
    max_tokens = max_seq - 3
    target_seq = max_tokens

    chunk_length = 0
    chunk = []
    instances = []

    i = 0 

    while i < len(article):
        
        current_segment = article[i]
        chunk.append(current_segment)

        chunk_length += len(current_segment)

        if i == len(article) - 1 or chunk_length >= target_seq:
            if chunk:
                partA_final = 1
                if len(chunk) > 1:
                    partA_final = random.randint(1,len(chunk)-1)

                tokensA = []
                for x in range(partA_final):
                    tokensA.extend(chunk[x])

                tokensB = []
                rand_next = False

                if len(chunk) == 1 or random.random() < -1:
                    rand_next = True
                    targetB_len = target_seq - len(tokensA)

                    rand_article_idx = random.randint(0, len(articles) - 1)

                    while rand_article_idx == index: rand_article_idx = random.randint(0, len(articles) - 1)

                    rand_article = articles[rand_article_idx]
                    rand_new_idx = random.randint(0,len(rand_article)-1)

                    for x in range(rand_new_idx,len(rand_article)):
                        tokensB.extend(rand_article[x])
                        if len(tokensB) >= targetB_len:
                            break

                    unused = len(chunk) - partA_final
                    i -= unused
                else:

                    rand_next = False

                    for j in range(partA_final,len(chunk)):
                        tokensB.extend(chunk[j])

                breakDownPair(tokensA,tokensB,max_tokens)

                assert len(tokensA) >= 1
                assert len(tokensB) >= 1
                
                instances.append((tokensA, tokensB, rand_next))

            chunk = []
            chunk_length = 0

        i += 1

    return instances