TFLite Analyser
===============

### Summary

This is a simple command line utility to analyse and print details of TFLite models. It is usefull to quickly be able to inspect various types of high level information of the model.

#### Usage

| tflite_micro_utils.py | [Options]... [flatbuffer file name] |
| :-: | --- |
| **positional arguments:** | |
| file_name | Name of the tflite flatbuffer file to load. |
| **optional arguments:** | |
| -h, --help | show this help message and exit |
| -i INDEX, --index INDEX | Index of the subgraph to analyse. Defaults to 0 |
| -o, --operations |Print a summary of the operations used in this model |
| -ot, --op_types | Print a summary of the operation types used in this model |
| -b, --buffers | Print a summary of all memory buffers definied in this model |
| -md, --meta_data | Print a summary of all metadata entries in this model |