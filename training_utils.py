import numpy as np
from collections import defaultdict

def sparse_reorder_split(m, new_order, split_range):
  assert len(split_range) == 2, "Length of split range must be 2"
  assert split_range[0] >= 0 and split_range[1] >= 0, "Split range must have values > 0"
  assert split_range[0] <= len(new_order) and split_range[1] <= len(new_order), "Split range must be less than/equal length of array"
  assert split_range[0] <= split_range[1], "Max split range %d is less than min split range %d!" % tuple(split_range)
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
    if i >= split_range[0] and i < split_range[1]:
      test_indices.extend([np.array([i-split_range[0]])]*len(values))
      test_values.extend(values)
    else:
      mod_i = i
      if i >= split_range[1]:
        mod_i = i - (split_range[1] - split_range[0])
      train_indices.extend([np.array([mod_i])]*len(values))
      train_values.extend(values)
  return np.array(train_indices), np.array(train_values), np.array(test_indices), np.array(test_values)

def flatten_nested_list(d):
  result = []
  for i,row in enumerate(d):
    _row = [[i, x] for x in row]
    result.extend(_row)
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
        result[key + "-indices_train"], result[key + "-values_train"], result[key + "-indices_test"], result[key + "-values_test"] = sparse_reorder_split(x, indices, [split_idx, N])
        result[key + "-shape_train"] = np.array(result[key + "-values_train"].shape)
        result[key + "-shape_test"] = np.array(result[key + "-values_test"].shape)
        continue
      assert len(x) == N, "Dataset under key %s has shape %s, expected first dimension of %d" % (key, str(x.shape), N)
      result[key + "_train"] = x[indices[:split_idx]]
      result[key + "_test"] = x[indices[split_idx:]]
    return result

def cv_split(d, partitions=5, sparse_keys=[]):
    assert len(d) > 0, "Must have at least one key/entry for the cross validation dataset"
    for key in sparse_keys:
        assert key + "-indices" in d and key + "-values" in d and key + "-shape" in d, "Need both indices and shape for key %s in dataset!" % key
        d[key] = np.vstack([d[key + "-indices"].flatten(), d[key + "-values"]]).T
        print("new shape is", d[key].shape)
        del d[key + "-indices"]
        del d[key + "-shape"]
        del d[key + "-values"]
    for key in d:
        assert type(d[key]).__module__ == np.__name__, "Must pass in a numpy array type!"
        if key not in sparse_keys:
            N = d[key].shape[0]
            break
    assert N > 0, "Must have at least one key in the dataset with non zero, non sparse data"
    assert N > partitions, "Must have at least as many elements in dataset as partitions"
    indices = np.random.permutation(N)
    split_indices = [int(N*i/(partitions)) for i in range(partitions)]
    split_indices = split_indices + [N]
    result = []
    for partition_idx in range(partitions):
        _d = {}
        for key in d:
            if key in sparse_keys:
                _d[key + "-indices_train"], _d[key + "-values_train"], _d[key + "-indices_test"], _d[key + "-values_test"] = sparse_reorder_split(d[key], indices, [split_indices[partition_idx], split_indices[partition_idx+1]])
                _d[key + "-shape_train"], _d[key + "-shape_test"] = np.array(_d[key + "-values_train"].shape), np.array(_d[key + "-values_test"].shape)
            else:
                _d[key + "_test"]= d[key][indices[split_indices[partition_idx]:split_indices[partition_idx+1]]]
                _d[key + "_train"] = np.expand_dims(np.append(d[key][indices[:split_indices[partition_idx]]], d[key][indices[split_indices[partition_idx+1]:]]), axis=-1)
        result.append(_d)                
    return result

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma  

def discretize_label(labels, survival, step=1.0):
    assert labels.shape == survival.shape, "Labels and survival vectors must be same shape!"
    result = []
    vec_sz = int(max(labels)) + 1 + 1 # Add an extra label for alive + over prediction
    for i in range(labels.shape[0]):
      y = np.zeros(vec_sz)
      y[int(labels[i])] = float(survival[i]) # 0 if alive, 1 if dead
      if survival[i] == 0:
        y[-1] = 1.0 # set the last label, which is designated for survival, as 1
      result.append(y)
    return np.array(result)

