import numpy as np
import pprint 
import re
import newspaper
from newspaper import Article
import scipy


link = "https://www.digi24.ro/stiri/actualitate/social/legea-asociatiilor-de-proprietari-a-fost-modificata-in-senat-pentru-copiii-mai-mici-de-trei-ani-nu-se-plateste-intretinere-1493181"

site = newspaper.build("https://evz.ro",memoize_articles=False)  

article = Article(link)
article.download()
article.parse()

content = [re.sub(r"[^a-z\u00C2\u0102\u00CE\u015E\u0162\u00E2\u0103\u00EE\u015F\u0163A-Z\d\s:]","",item.strip("\n").strip()) for item in re.split('\!|\.|\?',article.text.strip())]

site.article_urls()

for article in site.articles:
	try:
		article.download()
		article.parse()
		print(article.title)
	except:
		pass



print(content)