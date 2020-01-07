TFLite Micro Utilities
======================

### Summary

This is a simple command line utility to analyse, print, and modfiy details
of TFLite models. It is usefull to quickly be able to inspect various types of
high level information of the model.

#### Usage

| tflite_micro_utils.py | [Options]... [flatbuffer file name] |
| :-: | --- |
| **positional arguments:** | |
| file_name | Name of the tflite flatbuffer file to load. |
| **optional arguments:** | |
| -h, --help | show this help message and exit |
| -i Index, --index Index | Index of the subgraph to analyse. Defaults to 0 |
| -o, --operations | Print a summary of the operations used in this model |
| -md, --meta_data | Print a summary of all metadata entries in this model |
| -pm Output_File, --pre_allocate_memory Output_File | (In progress) pre-allocate intermediate tensors and safe positions into metadata table |