import numpy as np
from collections import defaultdict

ROOT_DATA_DIR = "patient/CancerNet/Datasets"

def dummy_dataset(num_examples, num_features, seed=0):
    np.random.seed(seed)
    X = (2*np.random.rand(num_examples,num_features)).astype('int32')
    Y = (100*np.random.rand(num_examples,1)).astype('int32')
    D = (2*np.random.rand(num_examples,1)).astype('int32')
    return X, Y, D

def dummy_dataset_embed(num_examples, num_features, embed_sz, seed=0):
    np.random.seed(seed)
    X = np.zeros([num_examples, num_features])
    Y = (100*np.random.rand(num_examples,1)).astype('int32')
    D = (2*np.random.rand(num_examples,1)).astype('int32')
    # Create embeddings for each gene/feature that is a random uniform vector of size embed_sz
    embeddings = np.random.uniform(size=[num_features, embed_sz])
    return X, Y, D, embeddings

def dummy_dataset_sparse_embed(num_examples, num_features, embed_sz, seed=0):
  np.random.seed(seed)
  X = []
  for i in range(num_examples):
    row = []
    for _ in range(np.random.randint(1,30)):
      row.append(np.array([i, np.random.randint(num_features)]))
    X.append(np.array(row))
  Y = (100*np.random.rand(num_examples,1)).astype('int32')
  D = (2*np.random.rand(num_examples,1)).astype('int32')
  # Create embeddings for each gene/feature that is a random uniform vector of size embed_sz
  embeddings = np.random.uniform(size=[num_features, embed_sz])
  return np.array(X), Y, D, embeddings

def process_patient_data(cancer):
  """
  Reads the interaction network for a given patient and creates a dictionary of gene-gene interactions that indexes by a gene name in to a list of the names of all the genes it interacts with.
  Input: 1) Name of the cancer to process a dataset for. The expected filepath to look in is ROOT_DATA_DIR/cancer, and the expected files within that directory are a {cancer}_matrix.txt file, {cancer}_survival.txt and a {cancer}_React.txt file. The first file is expected to have a header of 'patient id' with the patient ids in the first column and the gene names and 1/0 representation of a mutation in the following columns. The second file is expected to have a 'gene' header with all the gene names in the first column and first row, and be filled with 1/0 indicating reaction the two genes have been observed in together."""
  
  with open("%s/%s/%s_matrix.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    genes = f.readline().split() # Read first line (header)
    matrix = f.readlines()
  with open("%s/%s/%s_React.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    network = f.readlines()
  with open("%s/%s/%s_survival.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    survival = f.readlines()[1:]

  # First read in patient data and 
  interactions = defaultdict(list)

  for val in network:
    gene1, gene2 = val.split()
    if gene2 not in interactions[gene1]:
      interactions[gene1].append(gene2)
      interactions[gene2].append(gene1)

  patient_mutations, patient_mutation_names = {}, {}
  for line in matrix:
    arr = line.split()
    pid, mutations = arr[0].split('-'), arr[1:]
    pid, mutations = '-'.join(pid[:3]), [int(x) for x in mutations]
    mutation_names = [genes[i] for i in range(len(mutations)) if mutations[i] > 0]
    patient_mutations[pid] = mutations
    patient_mutation_names[pid] = mutation_names

  patient_survival = {}
  for line in survival:
    pid, years, alive = line.split()
    patient_survival[pid] = [float(years), int(alive)]

  one_hot, label_years, label_survival, mutations = [], [], [], []
  for pid in patient_survival:
    if pid not in patient_mutations:
        print "Patient %s not found in mutation information" % pid
        continue
    one_hot.append(patient_mutations[pid])
    label_years.append(patient_survival[pid][0])
    label_survival.append(patient_survival[pid][1])
    mutations.append(patient_mutation_names[pid])
  
  return one_hot, label_years, label_survival, mutations
