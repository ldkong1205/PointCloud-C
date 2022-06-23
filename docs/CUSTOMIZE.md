<img src="../figs/logo.png" align="right" width="25%">


# Customize Evaluation for Your Own Codebase

We have designed utilities to make evaluation on ModelNet-C easy. 
You may already have an evaluation code for the standard ModelNet.
It takes three simple steps to make it work on ModelNet-C.

### Step 1: Install and Import ModelNet-C Utility
Install our utility by:
```bash
git clone https://github.com/jiawei-ren/ModelNet-C.git
cd ModelNet-C
pip install -e modelnetc_utils
```
Import our utility in your evaluation script for the standard ModelNet40:
```python
from modelnetc_utils import eval_corrupt_wrapper, ModelNetC
```

### Step 2: Modify the Test Function
The test function on the standard ModelNet should look like:
```python
def test(args, model):
    '''
    Arguments:
        args: necessary arguments like batch size and number of workers
        model: the model to be tested
    Return:
        overall_accuracy: overall accuracy (OA)
    '''
    # Create test loader
    test_loader = DataLoader(ModelNet40(...), ...)
    
    # Run model on test loader to get the results
    overall_accuracy = run_model_on_test_loader(model, test_loader)
    
    # return the overall accuracy (OA)
    return overall_accuracy
```
where `run_model_on_test_loader` is usually a for-loop that iterates through all test batches.

To test on ModelNetC, we need an additional argument `split` to indicate the type of corruption. The modified test function should look like:
```python
def test_corrupt(args, model, split):
    '''
    Arguments:
        args: necessary arguments like batch size and number of workers
        model: the model to be tested
        split: corruption type
    Return:
        overall_accuracy: overall accuracy (OA)
    '''
    # Replace ModelNet40 by ModelNetC
    test_loader = DataLoader(ModelNetC(split=split), ...)
    
    # Remains unchanged
    overall_accuracy = run_model_on_test_loader(model, test_loader)
    
    # Remains unchanged
    return overall_accuracy
```

### Step 3: Call Our Wrapper Funcion
The calling of the test function for the standard ModelNet40 should be:
```python
overall_accuracy = test(args, model)
print("OA: {}".format(overall_accuracy))
```
For ModelNet-C, we provide a wrapper function to repeatedly call the test function for every corruption type and aggregate the results. 
We may conveniently use the wrapper function by:
```python
eval_corrupt_wrapper(model, test_corrupt, {'args': args})
```

### Example
An example evaluation code for ModelNet-C is provided in [GDANet/main_cls.py](https://github.com/jiawei-ren/ModelNet-C/blob/main/GDANet/main_cls.py#L312). 

Example output:
```bash
# result on clean test set
{'acc': 0.9359805510534847, 'avg_per_class_acc': 0.9017848837209301, 'corruption': 'clean'}
{'OA': 0.9359805510534847, 'corruption': 'clean', 'level': 'Overall'}

# result on scale corrupted test set
{'acc': 0.9258508914100486, 'avg_per_class_acc': 0.8890872093023254, 'corruption': 'scale', 'level': 0}
...
{'acc': 0.9047811993517018, 'avg_per_class_acc': 0.8646802325581395, 'corruption': 'scale', 'level': 4}
{'CE': 0.9008931342460089, 'OA': 0.9153160453808752, 'RCE': 1.0332252836304725, 'corruption': 'scale', 'level': 'Overall'}
...
# final result
{'RmCE': 1.207452747764862, 'mCE': 1.1023796740168037, 'mOA': 0.7303542486686734} 
```
