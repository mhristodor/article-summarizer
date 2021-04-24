import sys 
from gensim.corpora import WikiCorpus
from tqdm import tqdm_notebook as tqdm
import logging
import io

if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)
    
    wiki = WikiCorpus("F:\\corpus\\rowiki-latest-pages-articles.xml.bz2")
    print("Corpus Loaded | Starting article processing")

    with open("F:\\corpus\\latest_ro_2.txt","w",encoding = "utf-8") as f:

        i = 0
        for texts in tqdm(wiki.get_texts()):
            i = i + 1
            if i % 10000 == 0:
                print("Processed ",i, " articles")

            article = ' '.join(texts)
            
            one_line_article = article.replace("\n", " ")
            if len(one_line_article) < 100: continue  

            f.write(bytes(one_line_article, 'utf-8').decode('utf-8') + '\n')
        print("Processed ",i, " articles")