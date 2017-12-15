import numpy as np
import data_utils

WRITE_DATA_DIR = "patient"
CANCERS = ['gbm', 'luad', 'lusc']
EMBEDDING_DATA_PATH = "embeddings"
EMBEDDINGS = ['embedding_disease', 'embedding_gene_coexpression', 'embedding_gene_gene_interaction']

if __name__ == "__main__":
  # Read patient data and create one hot embeddings, patient->mutation names dictionary, and patient survivals
  for cancer in CANCERS:
    print("Processing patient data for cancer %s" % cancer)
    genes, one_hot, label_years, label_survival = data_utils.process_patient_data(cancer)
    sparse = data_utils.sparsify(one_hot)
    np.save('%s/%s/genes.npy' % (WRITE_DATA_DIR, cancer), genes)
    np.save('%s/%s/sparse.npy' % (WRITE_DATA_DIR, cancer), sparse)
    np.save('%s/%s/one_hot.npy' % (WRITE_DATA_DIR, cancer), one_hot)
    np.save('%s/%s/labels.npy' % (WRITE_DATA_DIR, cancer), label_years)
    np.save('%s/%s/survival.npy' % (WRITE_DATA_DIR, cancer), label_survival)

    # Creating a dictionary to convert standardized gene id for mutations of interest to its indices in genes
    print("Creating standardized gene id -> our mapping")

    num_non_translatable = 0    
    # First create mapping from the gene name to the standard id
    with open('embeddings/genes_id', 'r') as f:
      lines = f.readlines()
      name_to_standard = {}
      for i in range(len(lines)):
        gene, idx = lines[i].split()
        name_to_standard[gene] = int(idx)

    # Create mapping from standard id to our custom id (in genes)
    standard_to_custom = {}
    for i in range(len(genes)):
      gene_name = genes[i]
      if gene_name not in name_to_standard:
        # print("Gene %s not found in standardized gene ids" % gene_name)
        num_non_translatable += 1
        continue
      standard_to_custom[name_to_standard[gene_name]] = i
    print("Total number of genes that had no mapping from patient mutation -> standardized gene id is %d" % num_non_translatable)

    # Take patient -> standardized gene id and create + save embeddings for all of our gene embeddings
    for embedding in EMBEDDINGS:
      print("Creating embedding for %s" % embedding)
      embed_mat, oov = data_utils.create_reduced_embeddings(embedding, standard_to_custom, genes)
      print("Number of OOV elements for %s is %d" % (embedding, len(oov)))    
      np.save('%s/%s/%s.npy' % (WRITE_DATA_DIR, cancer, embedding), embed_mat)
