# Authorship Attribution in a forensic context
All code as used in the thesis project with the same name. To run this code first use df_creator2.py to create a dataframe with all the test of the corpus. Currently the corpora FRIDA and abc_nl1 are supported.
The saved dataframe can then be used for either Computational Authorship Attribution (CAA) using CAA_feature_based.py or CAA_BERT_based.py or for Forensic Authorship Attribution (FAA), using LRsystem.py.
The helper_function folder contains all functions needed for the different methods to work. In config.json all variables of the models can be specified, which are either strings, numbers or booleans represented as 0 or 1.

In Toy.py a toy example of likelihood ratio results are used to show how various plotting methods work.

