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