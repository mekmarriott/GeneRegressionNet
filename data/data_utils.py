import numpy as np

ROOT_DATA_DIR = "patient/raw_data"
WRITE_DATA_DIR = "patient"

def dummy_dataset(num_examples, num_features, seed=0):
    np.random.seed(seed)
    X = (2*np.random.rand(num_examples,num_features)).astype('int32')
    Y = (100*np.random.rand(num_examples,1)).astype('int32')
    D = (2*np.random.rand(num_examples,1)).astype('int32')
    return X, Y, D

def read_gene_interaction_data(cancer):
  """
  Reads the interaction network for a given patient and creates a dictionary of gene-gene interactions that indexes by a gene name in to a list of the names of all the genes it interacts with.
  Input: 1) Name of the cancer to process a dataset for. The expected filepath to look in is ROOT_DATA_DIR/cancer, and the expected files within that directory are a {cancer}_matrix.txt file, {cancer}_survival.txt and a {cancer}_React.txt file. The first file is expected to have a header of 'patient id' with the patient ids in the first column and the gene names and 1/0 representation of a mutation in the following columns. The second file is expected to have a 'gene' header with all the gene names in the first column and first row, and be filled with 1/0 indicating reaction the two genes have been observed in together."""
  with open("%s/%s/%s_matrix.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    gene_indices = f.readline() # Read first line (header)
    matrix = f.readlines()
  with open("%s/%s/%s_React.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    network = f.readlines()
  with open("%s/%s/%s_survival.txt" % (ROOT_DATA_DIR, cancer, cancer)) as f:
    survival = f.readlines()

  # First read in patient data and 
  interactions = defaultdict(list)

  for val in network:
    gene1, gene2 = val.split()
    if gene2 not in interactions[gene1]:
      interactions[gene1].append(gene2)
      interactions[gene2].append(gene1)
    # Add to gene_indices if not already there
    if gene1 not in genes:
      genes.append(gene1)
    if gene2 not in genes:
      genes.append(gene2)

  patient_mutations = {}
  for line in matrix:
    arr = line.split()
    pid, mutations = arr[0].split(), arr[1:]
    pid, mutations = '-'.join(patient_info[:2]), [int(x) for x in mutations]
    patient_mutations[pid] = mutations

  patient_survival = {}
  for line in survival:
    pid, years, alive = line.split()
    patient_survival[pid] = [years, alive]

  one_hot, label_years, label_survival = [], []
  for pid in patient_survival:
    one_hot.append(patient_mutations[pid])
    label_years.append(patient_survival[pid][0])
    label_survival.append(patient_survival[pid][1])

  np.save('%s/%s/one_hot.npy' % (WRITE_DATA_DIR, cancer), one_hot)
  np.save('%s/%s/labels.npy' % (WRITE_DATA_DIR, cancer), label_years)
  np.save('%s/%s/survival.npy' % (WRITE_DATA_DIR, cancer), label_survival)
