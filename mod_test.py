from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import math
import array

import tensorflow as tf
import hashlib as hl
from shutil import which
import subprocess as subproc

import EnsureSchemaCompiled

import flatbuffers as fb
import tflite.Model as tfl_model
import tflite.BuiltinOptions as BuiltinOptions
import tflite.Conv2DOptions as Conv2DOptions
import tflite.Pool2DOptions as Pool2DOptions
import tflite.DepthwiseConv2DOptions as DepthwiseConv2DOptions
import tflite.BuiltinOperator as BuiltinOperator
import tflite.Metadata as Metadata


def main():

  #ensure_schema_compiled()

  tflite_file = None
  file_name = FLAGS.file_name
  base_name = file_name
  if base_name[-7:] == ".tflite":
    base_name = base_name[:-7]

  try:
    tflite_file = open(file_name, 'rb')
  except IOError:
    print("Failed to open file \"%s\"." % file_name)
    quit()

  print("=" * (len(file_name) + 14 + 21))
  print("====== Reading flatbuffer \"%s\" ======" % file_name)
  print("=" * (len(file_name) + 14 + 21))
  flatbuffer = tflite_file.read()
  buf = bytearray(flatbuffer)
  print("Done.")

  print("Generate flatbuffer using original API")
  model_old = tfl_model.Model.GetRootAsModel(buf, 0)
  print("Loaded model with version %d" % model_old.Version())

  print("Generate flatbuffer object API instance of model")
  # model = tfl_model.ModelT()
  n = fb.encode.Get(fb.packer.uoffset, buf, 0)
  print("n = %d" % n)
  model = tfl_model.ModelT.InitFromBuf(buf, n)
  print("Loaded model with version %d" % model.version)

  print("Metadata type is : %s" % str(type(model.metadata)))

  metadata = model.metadata
  if metadata is None:
    metadata = []
  print("Model contains %d meta data entries" % len(metadata))
  for i, m in enumerate(metadata):
    print("element type : %s" % str(type(m)))
    print("[%d] - %s" % (i, m.name))

  print("Add metadata item to model")
  new_metadata = Metadata.MetadataT()
  print("type of new_metadata object : %s" % str(type(new_metadata)))
  new_metadata.name = "Test Metadata"
  if model.metadata is None:
    model.metadata = []
  model.metadata.append(new_metadata)

  metadata = model.metadata
  print("Model contains %d meta data entries" % len(metadata))
  for i, m in enumerate(metadata):
    print("element type : %s" % str(type(m)))
    print("[%d] - %s" % (i, m.name))

  print("Rebuild and save modified model")
  builder = fb.Builder(1024)

  if builder.finished:
    print("builder finished")
  else:
    print("builder not finished")
  print("Builder %d bytes" % len(builder.Bytes))

  packed_model = model.Pack(builder)
  print("Packed model")
  if builder.finished:
    print("builder finished")
  else:
    print("builder not finished")
  print("Builder %d bytes" % len(builder.Bytes))

  print("Builder head is [%s]" % builder.head)

  builder.Finish(packed_model, file_identifier=bytearray("TFL3", 'utf-8'))
  new_model_buffer = builder.Output()
  # completed_buffer = buf[0:24] + new_model_buffer

  modified_name = base_name + "_new.tflite"
  new_tflite_file = open(modified_name, 'wb')
  new_tflite_file.write(new_model_buffer)
  print("Completed writing modifiled flatbuffer to \"%s\"" % modified_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_name', type=str,
                      help='Name of the tflite flatbuffer file to load.')
  parser.add_argument('-i', '--index',
                      type=int, default=0,
                      help='Index of the subgraph to analyse. Defaults to 0')
  parser.add_argument('-a', '--all', action="store_true",
                      help='Print out all details of this model.')
  parser.add_argument('-sg', '--sub_graphs',
                      action="store_true",
                      help='Print a list of all the graphs stored in this tflite flatbuffer.')
  parser.add_argument('-d', '--meta_data',
                      action="store_true",
                      help="Prints any meta-data members stored in this tflite flatbuffer.")
  parser.add_argument('-o', '--operations',
                      action="store_true",
                      help='Print a summary of the operations used in this model.')
  parser.add_argument('-ot', '--op_types',
                      action="store_true",
                      help='Print a summary of the operation types used in this model.')
  parser.add_argument('-w', '--weights', action="store_true",
                      help='Print detail of the weights of this model.')
  parser.add_argument('-m', '--memory',
                      action="store_true",
                      help='Print details of memory allocation required by this model.')
  parser.add_argument('-t', '--tensors',
                      action="store_true",
                      help='Print details of the tensors used in this model.')
  parser.add_argument('-s', '--save_csv',
                      action="store_true",
                      help='Write the selected detail to a set of CSV files.')
  parser.add_argument('-om', '--optimise_memory',
                      action="store_true",
                      help='Uses various algorithms to precalculate an optimal memory useage pattern.')
  parser.add_argument('-p', '--add_preallocations',
                      action="store_true",
                      help='Test option to add the TFL micro pre-allocation meta-data.')

  FLAGS, unparsed = parser.parse_known_args()

  main()