import numpy as np
from collections import defaultdict

def sparse_reorder_split(m, new_order, split_idx):
  assert m.shape[1] == 2, "Shape of sparse input must be (?, 2)"
  idx_to_values = defaultdict(list)
  train, test = [], []
  for pair in m:
    idx_to_values[pair[0]].append(pair[1])
  for i in range(len(new_order)):
    values = idx_to_values[new_order[i]]
    assert len(values) > 0, "There is no data for patient at index %d" % new_order[i]
    if i < split_idx:
      train.extend([[i, val] for val in values])
    else:
      test.extend([[i-split_idx, val] for val in values])
  return np.array(train), np.array(test)

def flatten_nested_list(d):
  result = []
  for i,row in enumerate(d):
    _row = [[i, x] for x in row]
    result.extend(_row)
  return np.array(result)

def train_test_split(d, split=0.7, sparse_keys=[]):
    assert len(d) > 0, "Must have at least one key/entry for the dataset"
    for key in d:
    	assert type(d[key]).__module__ == np.__name__, "Must pass in a numpy array type!"
    N = len(list(d.values())[0])
    indices = np.random.permutation(N)
    split_idx = int(N*split)
    assert split_idx > 0 and split_idx < N, "Split index cannot leave empty training or test set - split index is %d, split is %f, and the size to split is %d" % (split_idx, split, N)
    result = {}
    for key in d:
      x = d[key]
      if key in sparse_keys:
        result[key + "_train"], result[key + "_test"] = sparse_reorder_split(x, indices, split_idx)
        continue
      assert len(x) == N, "Dataset under key %s has shape %s, expected first dimension of %d" % (key, str(x.shape), N)
      result[key + "_train"] = x[indices[:split_idx]]
      result[key + "_test"] = x[indices[split_idx:]]
    return result

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma  
