import gc
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd


fname = 'yelp.json'
df    = pd.read_json(fname)
data  = df.text.values
label = df.review_id.values
df    = None
gc.collect()


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def nlp_clean(data):
   new_data = []
   for i in range(len(data)):
      new_str = data[i].lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data


class TaggedDocs(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield TaggedDocument(doc, [self.labels_list[idx]])

dat = nlp_clean(data)
itr = TaggedDocs(data,label)

ndim  = 100
model = Doc2Vec(vector_size=ndim, window=10, min_count=1, 
                alpha=0.025, min_alpha=0.025, workers=2,dm=1)
model.build_vocab(it)

# Training of model
model.train(it, total_examples=len(dat), epochs=2)


# Saving the created model
# model.save('doc2vec.model')
# print 'model saved'

model = Doc2Vec.load('doc2vec.model')
print 'model loaded'

# docvec = model.docvecs.infer_vector('war.txt')