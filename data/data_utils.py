import numpy as np
from collections import defaultdict

ROOT_DATA_DIR = "patient/CancerNet/Datasets"
EMBEDDING_DATA_PATH = "embeddings"

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

# Take dictionary of patient -> custom id (one hot) and create a sparse embedding from it
def sparsify(M):
  result = []
  for i in range(len(M)):
    for element in M[i]:
      if element > 0:
        result.append([i,element])

def create_reduced_embeddings(embedding, dimension_mapping, genes):
  # Create lookup from custom ids to embeddings
  embed_mat, lookup, oov = [], {}, set()
  with open('%s/%s' % (EMBEDDING_DATA_PATH, embedding), 'r') as f:
    embed_sz = -1
    lines = f.readlines()[1:]
    for line in lines:
      arr = [float(x) for x in line.split()]
      std_id, vec = int(arr[0]), arr[1:]
      if embed_sz < 0:
        embed_sz = len(vec)
      else:
        assert(embed_sz) == len(vec), "Embedding vector with standard gene id %d has length of %d, expected %d" % (std_id, len(vec), embed_sz) 
      if std_id not in dimension_mapping:
        continue
      lookup[dimension_mapping[std_id]] = vec

  # Populate embeddings in an embedding matrix
  for i in range(len(genes)):
    if i not in lookup:
      # print("No embedding was found for gene %s on embedding %s - using zero vector as out of vocabulary" % (genes[i], embedding))
      oov.add(genes[i])
      embed_mat.append([0]*embed_sz)
      continue
    embed_mat.append(lookup[i])
  return embed_mat, oov

def process_patient_data(cancer):
  """
  Reads the interaction network for a given patient and creates a dictionary of gene-gene interactions that indexes by a gene name in to a list of the names of all the genes it interacts with.
  Input: 1) Name of the cancer to process a dataset for. The expected filepath to look in is ROOT_DATA_DIR/cancer, and the expected files within that directory are a {cancer}_matrix.txt file, {cancer}_survival.txt and a {cancer}_React.txt file. The first file is expected to have a header of 'patient id' with the patient ids in the first column and the gene names and 1/0 representation of a mutation in the following columns. The second file is expected to have a 'gene' header with all the gene names in the first column and first row, and be filled with 1/0 indicating reaction the two genes have been observed in together."""
  
  with open("%s/%s/%s_matrix.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    genes = f.readline().split() # Read first line (header)
    matrix = f.readlines()
  with open("%s/%s/%s_survival.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    survival = f.readlines()[1:]

  patient_mutations = {}
  for line in matrix:
    arr = line.split()
    pid, mutations = arr[0].split('-'), arr[1:]
    pid, mutations = '-'.join(pid[:3]), [int(x) for x in mutations]
    patient_mutations[pid] = mutations

  patient_survival = {}
  for line in survival:
    pid, years, alive = line.split()
    patient_survival[pid] = [float(years), int(alive)]

  one_hot, label_years, label_survival = [], [], []
  for pid in patient_survival:
    if pid not in patient_mutations:
        print "Patient %s not found in mutation information" % pid
        continue
    one_hot.append(np.array(patient_mutations[pid]))
    label_years.append(patient_survival[pid][0])
    label_survival.append(patient_survival[pid][1])
 
  # Convert everything to numpy arrays of the correct dimensions
  one_hot = np.array(one_hot)
  label_years = np.expand_dims(np.array(label_years), axis=-1)
  label_survival = np.expand_dims(np.array(label_survival), axis=-1)
  
  return genes, one_hot, label_years, label_survival
