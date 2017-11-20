import numpy as np
import data_utils

WRITE_DATA_DIR = "patient"
CANCERS = ['gbm']

if __name__ == "__main__":
  # Read patient data and create one hot embeddings, patient->mutation names dictionary, and patient survivals
  for cancer in CANCERS:
    one_hot, label_years, label_survival, mutations = data_utils.process_patient_data(cancer)

    # np.save('%s/%s/mutations.npy' % (WRITE_DATA_DIR, cancer), mutations)
    np.save('%s/%s/one_hot.npy' % (WRITE_DATA_DIR, cancer), one_hot)
    np.save('%s/%s/labels.npy' % (WRITE_DATA_DIR, cancer), label_years)
    np.save('%s/%s/survival.npy' % (WRITE_DATA_DIR, cancer), label_survival)

    # Take dictionary of patient -> gene names and convert to patient -> standardized gene id
    with open('embeddings/genes_id', 'r') as f:
      lines = f.readlines()
      mutations_to_id = {}
      for line in lines:
        gene, idx = line.split()
        mutations_to_id[gene] = int(idx)

    mutation_ids = []
    for row in mutations:
      _row = []
      for gene in row:
        if gene in mutations_to_id:
          _row.append(mutations_to_id[gene])
        else:
          print("Gene %s not found in standardized gene ids" % gene)
      mutation_ids.append(_row)
    np.save('%s/%s/mutation_ids.npy' % (WRITE_DATA_DIR, cancer), mutation_ids)

    # Take patient -> standardized gene id and create + save embeddings for all of our gene embeddings
