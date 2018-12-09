# =============================================================================
# Created By: bpatter5
# Updated By: bpatter5
# Created On: 12/3/2018
# Updated On: 12/8/2018
# Purpose: 
# =============================================================================

import pyarrow as pa
from pyarrow import parquet as pq
import numpy as np



def write_test(path, test, context):
    '''
    Description
    -----------
    Method to serialize a test and write it to disk
    
    Parameters
    ----------
    path : string
        File path for which to write the serialized object to
    
    test : ann_inference.testing.xxxx_test
        Test to serialize and write to disk
    
    context : pa.SerializationContext
        Custom serialization context that defines how the test is to be written
    
    Returns
    -------
        : void 
        Writes a serialized pyarrow buffer to disk        
    
    '''
    
    buf = context.serialize(test).to_buffer()
    
    with open(path, 'wb') as f:
        f.write(buf)
        
def read_test(path, context):
    '''
    Description
    -----------
    Read a serialized pyarrow buffer from disk and return a test object
    
    Parameters
    ----------
    path : string
        File path to serialized test
    
    context: pa.SerializationContext
        Custom serialization context to read the test back into memory
        
    Returns
    -------
        : ann_inference.testing.xxxx_test
        Desearialized test using the given context
    '''
    mmap = pa.memory_map(path)
    
    return(context.deserialize(mmap.read_buffer()))


def gen_parquet_batch(test_stat, fill_col, epoch_num, test_num, col_names):
    '''
    Description
    -----------
    Generate a pyarrow.RecordBatch from a numpy array of PyTorch model parameters.
    
    Parameters
    ----------
    test_stat : np.array
       Array of parameters to be written to batch and eventually saved to disk for later testing
    
    fill_col : string
        Name of the parameter being tested which will later become a partition on disk
    
    epoch_num : int
        Current epoch
    
    test_num : int
        Current test assuming multiple model runs
        
    col_names : list[string]
        Column names to assign to fields in the current batch
    
    Returns
    -------
        : pyarrow.RecordBatch
        Record batch with nrows(test_stat) x ncols(test_stat) + 3
        
    '''
    data = []
        
    
        
    data.append(pa.array(np.full(test_stat.shape[0], fill_col)))
    data.append(pa.array(np.full(test_stat.shape[0], test_num)))
    data.append(pa.array(np.full(test_stat.shape[0], epoch_num)))
        
    for j in np.arange(0, test_stat.shape[1]):
        data.append(pa.array(test_stat[:,j]))
        
    return(pa.RecordBatch.from_arrays(data, col_names))

def array_to_parquet(path, test_stat, fill_col, test_num, col_names):
    
    '''
    Description
    -----------
    Generate a Parquet dataset from a numpy array of PyTorch model parameters.
    
    Parameters
    ----------
    test_stat : np.array
       Array of parameters to be written to batch and eventually saved to disk for later testing
    
    fill_col : string
        Name of the parameter being tested which will later become a partition on disk
    
    test_num : int
        Current test assuming multiple model runs
        
    col_names : list[string]
        Column names to assign to fields in the current batch
    
    Returns
    -------    
        : void
        writes to parquet store
    '''
    data = []
    
    data.append(pa.array(np.full(test_stat.shape[0], fill_col)))
    data.append(pa.array(np.full(test_stat.shape[0], test_num)))
    data.append(pa.array(np.arange(0, test_stat.shape[0])))
    
    data.append(pa.array(test_stat))
    
    error_tbl = pa.Table.from_arrays(data, col_names)
    
    pq.write_to_dataset(error_tbl, path, partition_cols=['stat'])

def results_to_parquet(path, batch_list):
    '''
    Description
    -----------
    Function turns a list of batches into a parquet dataset on disk for future analysis.
    
    Parameters
    ----------
    path : string
        Base path to create parquet dataset at
    
    batch_list : list[pyarrow.RecordBatch]
        Batches with matching schema definitions to be written to disk
    
    Returns
    -------
        : void
        Writes a parquet dataset to disk
    
    '''
    
    tbl = pa.Table.from_batches(batch_list)
    
    pq.write_to_dataset(tbl, path, partition_cols=['stat'])
    

def read_parquet_store(path, nthreads=5):
    '''
    Description
    -----------
    Read dataset at the given path into a pandas dataframe for analysis
    
    Parameters
    ----------
    path : string
        path to parquet data store to read into memory for analysis
        
    nthreads : int, 5
        number of threads to use for reading the parquet store into memory
        
    Returns
    -------
     : pandas.DataFrame
         pandas frame at the given path
    '''
    return(pq.read_table(path, nthreads=nthreads).to_pandas())