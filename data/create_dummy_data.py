import numpy as np
import data_utils

NUM_EXAMPLES = 250
NUM_FEATURES = 2000
CANCERS = ['gbm']

for cancer in CANCERS:
	X, Y, D, gene_idx_to_name = data_utils.dummy_dataset(NUM_EXAMPLES, NUM_FEATURES)
	np.save(X, 'dummy_patient/%s/onehot.npy')
	np.save(Y, 'dummy_patient/%s/labels.npy')
	np.save(D, 'dummy_patient/%s/survival.npy')
	np.save(gene_idx_to_name, 'dummy_patient/%s/gene_idx_to_name.npy')

	gene_name_to_embed = np.load('embeddings/%s/gene_name_to_embed.npy')
	data_utils.create_embeddings(X, gene_idx_to_name, gene_name_to_embed)
	np.save(X_embed, 'dummy_patient/%s/embed.npy')
