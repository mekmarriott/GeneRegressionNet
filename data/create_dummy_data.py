import numpy as np
import data_utils

NUM_EXAMPLES = 250
NUM_FEATURES = 2000
EMBED_SZ = 10
WRITE_DATA_DIR = "dummy_patient"
EMBEDDING_DATA_PATH = "embeddings"
CANCERS = ['gbm']

for cancer in CANCERS:
  print("Creating dummy patient data for cancer %s" % cancer)
  sparse, label_years, label_survival, embeddings = data_utils.dummy_dataset_sparse_embed(NUM_EXAMPLES, NUM_FEATURES, EMBED_SZ)
  # sparse = data_utils.sparsify(one_hot)
  np.save('%s/%s/sparse.npy' % (WRITE_DATA_DIR, cancer), sparse)
  # np.save('%s/%s/one_hot.npy' % (WRITE_DATA_DIR, cancer), one_hot)
  np.save('%s/%s/labels.npy' % (WRITE_DATA_DIR, cancer), label_years)
  np.save('%s/%s/survival.npy' % (WRITE_DATA_DIR, cancer), label_survival)
  np.save('%s/%s/dummy_embedding.npy' % (WRITE_DATA_DIR, cancer), embeddings)


  

