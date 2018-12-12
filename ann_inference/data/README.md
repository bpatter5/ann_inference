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
        

## load_data

Methods for loading canned data for testing the properties of a network.

* _gen_regression(samples, features, inform, bias=0.0, noise=1.0, random_state=None)_:
    * Description:
        * Generate datasets for regression problems
    * Parameters:
        * samples : int number of samples to generate
        * features : int number of independent variables in the dataset
        * inform : int number of informative features
        * bias : float bias term in generated dataset
        * noise : float variance of noise to inject into samples
        * random_state : RandomState to seed the random number generator for samples
        
    * Returns:
        * tuple(X, y) of independent and dependent variables
        
* _gen_friedman(i, samples, noise=1.0, random_state=None, features=20)_:
    * Description:
        * Generate datasets for regression problems    
    * Parameters:
        * i : int dataset to generate from datasets dict
        * samples : int number of samples to generate
        * features : int number of independent variables in the dataset
        * noise : float variance of noise to inject into samples
        * random_state : RandomState to seed the random number generator for samples
    * Returns:
        * tuple(X, y) of independent and dependent variables
        

#### ModelData

* Description: 
    * Class for data generation to fit a PyTorch model, see help(ModelData) for function definitions
* Parameters:
    * X : numpy.array of inputs for testing
    * y : numpy.array of actual y values
    * seed : int seed for random number generator
    * train_pct : float, [0.0-1.0] percentage of dataset to save for training vs testing
    
    