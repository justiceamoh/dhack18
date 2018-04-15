import gc
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"


fname = 'yelp.json'
df    = pd.read_json(fname)
data  = df.text.values
label = df.review_id.values
df    = None
gc.collect()

# data  = data[:500]
# label = label[:500]

tokenizer = RegexpTokenizer(r'\w+')
stopset   = set(stopwords.words('english'))

docs      = []

for i in range(len(data)):
    tag    = label[i]
    words  = data[i].lower()
    tokens = tokenizer.tokenize(words)
    tokens = [w for w in tokens if not w in stopset]
    docs.append(TaggedDocument(tokens,[str(tag)]))


ndim  = 100
model = Doc2Vec(dm=0,vector_size=ndim, window=10, min_count=3,
                alpha=0.025, min_alpha=0.025, workers=cores)
model.build_vocab(docs)


# Training of model
for epoch in range(10):
    print 'iteration '+str(epoch+1)
    model.train(docs, total_examples=len(docs), epochs=1)
    model.alpha -= 0.001
    model.min_alpha = model.alpha


# Saving the created model
model.save('doc2vec100.model')
print 'model saved'

# model = Doc2Vec.load('doc2vec.model')
# print 'model loaded'

# docvec = model.docvecs.infer_vector('war.txt')