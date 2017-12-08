import numpy as np
from collections import defaultdict

def sparse_reorder_split(m, new_order, split_idx):
  assert m.shape[1] == 2, "Shape of sparse input must be (?, 2)"
  idx_to_values = defaultdict(list)
  train_indices, train_values, test_indices, test_values = [], [], [], []
  for i in range(m.shape[0]):
    pid = m[i][0]
    new_order_idx, = np.where(new_order==pid)
    assert len(new_order_idx) == 1, "Patient index %d is out of bounds - must be in range(%d)" % (pid, len(new_order))
    idx_to_values[new_order_idx[0]].append(m[i][1])
  for i in range(len(new_order)):
    values = idx_to_values[new_order[i]]
    if len(values) == 0:
      print("There is no data for patient at index %d" % new_order[i])
      continue
    if i < split_idx:
      train_indices.extend([np.array([i])]*len(values))
      train_values.extend(values)
    else:
      test_indices.extend([np.array([i-split_idx])]*len(values))
      test_values.extend(values)
  return np.array(train_indices), np.array(train_values), np.array(test_indices), np.array(test_values)

def flatten_nested_list(d):
  print(d.shape)
  result = []
  for i,row in enumerate(d):
    _row = [[i, x] for x in row]
    result.extend(_row)
  print(len(result))
  return np.array(result)

def train_test_split(d, split=0.7, sparse_keys=[]):
    assert len(d) > 0, "Must have at least one key/entry for the dataset"
    N = -1
    for key in d:
        assert type(d[key]).__module__ == np.__name__, "Must pass in a numpy array type!"
        if key not in sparse_keys:
            if N > 0:
                assert N == d[key].shape[0], "Dataset under key %s has length %d, expected length %d" % (key, d[key].shape[0], N)
            else:
                N = d[key].shape[0]
    indices = np.random.permutation(N)
    split_idx = int(N*split)
    assert split_idx > 0 and split_idx < N, "Split index cannot leave empty training or test set - split index is %d, split is %f, and the size to split is %d" % (split_idx, split, N)
    result = {}
    for key in d:
      x = d[key]
      if key in sparse_keys:
        result[key + "-indices_train"], result[key + "-values_train"], result[key + "-indices_test"], result[key + "-values_test"] = sparse_reorder_split(x, indices, split_idx)
        result[key + "-shape_train"] = np.array(result[key + "-values_train"].shape)
        result[key + "-shape_test"] = np.array(result[key + "-values_test"].shape)
        continue
      assert len(x) == N, "Dataset under key %s has shape %s, expected first dimension of %d" % (key, str(x.shape), N)
      result[key + "_train"] = x[indices[:split_idx]]
      result[key + "_test"] = x[indices[split_idx:]]
    return result

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma  

def discretize_label(labels, survival, step=1.0):
    assert labels.shape == survival.shape, "Labels and survival vectors must be same shape!"
    result = []
    vec_sz = int(max(labels)) + 1
    for i in range(labels.shape[0]):
      y = np.zeros(vec_sz)
      y[int(labels[i])] = float(survival[i]) # 0 if alive, 1 if dead
      result.append(y)
    return np.array(result)

