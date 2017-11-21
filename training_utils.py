import numpy as np

def train_test_split(d, split=0.7, sparse=False):
    assert len(d) > 0, "Must have at least one key/entry for the dataset"
    for key in d:
    	assert type(d[key]).__module__ == np.__name__, "Must pass in a numpy array type!"
    N = len(d.values()[0])
    indices = np.random.permutation(N)
    result = {}
    for key in d:
      x = d[key]
      assert len(x) == N, "Dataset under key %s has shape %s, expected first dimension of %d" % (key, str(x.shape), N)
      result[key + "_train"] = x[indices[:N*split]]
      result[key + "_test"] = x[indices[N*split:]]
    return result

def flatten_nested_list(d):
  result = []
  for i,row in enumerate(d):
    _row = [[i, tup[1]] for tup in row]
    result.extend(_row)
  return np.array(result)

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma  
