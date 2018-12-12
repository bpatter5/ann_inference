# Data

## arrow_helper

* _write_test(path, test, context)_:
    * Parameters:
        * path : string to write serialized test to
        * test : RegressionTester to create serialized object of
        * context: SerializationContext for test object
    * Description:
        * Method to serialize a test and write it to disk
        
* _read_test(path, context)_:
    * Parameters:
        * path : string to read serialized test from
        * context: SerializationContext for test object
    * Description:
        * Read a serialized pyarrow buffer from disk and return a test object
        
* _gen_parquet_batch(test_stat, fill_col, epoch_num, test_num, col_names)_:  
    * Description
        * Generate a pyarrow.RecordBatch from a numpy array of PyTorch model parameters.
    * Parameters
        * test_stat : np.array of parameters to be written to batch and eventually saved to disk for later testing
        * fill_col : name of the parameter being tested which will later become a partition on disk
        * epoch_num : current epoch
        * test_num : current test assuming multiple model runs        
        * col_names : list(string) of column names to assign to fields in the current batch
    * Returns
        * pyarrow.RecordBatch with nrows(test_stat) x ncols(test_stat) + 3
        
* _array_to_parquet(path, test_stat, fill_col, test_num, col_names)_:
    * Description:
        * generate a Parquet dataset from a numpy array of PyTorch model parameters
    * Parameters:
        * test_stat : np.array of parameters to be written to batch and eventually saved to disk for later testing
        * fill_col : string parameter being tested which will later become a partition on disk
        * test_num : int id of current test assuming multiple model runs
        * col_names : list(string) of column names to assign to fields in the current batch
    
    * Returns
        * void return but writes to parquet store
        
* _results_to_parquet(path, batch_list)_:
    * Description:
        * Turns a list of batches into a parquet dataset on disk for future analysis.
    * Parameters:
        * path : string path to create parquet dataset at 
        * batch_list : list(pyarrow.RecordBatch) with matching schema definitions to be written to disk
    * Returns:
        * void but writes a parquet dataset to disk
        
* _read_parquet_store(path, nthreads=5)_:
    * Description:
        * Read dataset at the given path into a pandas dataframe for analysis
    * Parameters:
        * path : string to parquet data store to read into memory for analysis
        * nthreads : int(5) number of threads to use for reading the parquet store into memory
    * Returns
        * pandas.DataFrame from the Parquet files at the given path (must have same schema)