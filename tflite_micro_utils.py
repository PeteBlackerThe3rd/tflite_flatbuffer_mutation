from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard python imports
import argparse
import os
import sys
import numpy as np
import math
import array
import hashlib as hl
from shutil import which
import subprocess as subproc

# local module which compiles the provided flatbuffer schema into it's
# python interface if it doesn't exist or needs updating
import EnsureSchemaCompiled

# import flatbuffers and tflite schema interface
import flatbuffers as fb
import tflite.Model as TFLModel
import tflite.BuiltinOptions as BuiltinOptions
import tflite.Conv2DOptions as Conv2DOptions
import tflite.Pool2DOptions as Pool2DOptions
import tflite.DepthwiseConv2DOptions as DepthwiseConv2DOptions
import tflite.BuiltinOperator as BuiltinOperator
import tflite.Metadata as Metadata
import tflite.Buffer as Buffer


def load_flatbuffer():
  file_name = FLAGS.file_name
  # base_name = file_name
  # if base_name[-7:] == ".tflite":
  #  base_name = base_name[:-7]

  try:
    tflite_file = open(file_name, 'rb')

    print("=" * (len(file_name) + 35))
    print("====== Reading flatbuffer \"%s\" ======" % file_name)
    print("=" * (len(file_name) + 35))
    buf = bytearray(tflite_file.read())
    print("Read %d bytes okay." % len(buf))

    # print("Generate flatbuffer using original API")
    # model_old = TFLModel.Model.GetRootAsModel(buf, 0)
    # print("Loaded model with version %d" % model_old.Version())

    # print("Generate flatbuffer object API instance of model")
    # model = tfl_model.ModelT()
    n = fb.encode.Get(fb.packer.uoffset, buf, 0)
    print("n = %d" % n)
    model = TFLModel.ModelT.InitFromBuf(buf, n)
    print("Loaded version %s model" % model.version)

    return model

  except IOError:
    print("Failed to open file \"%s\"." % file_name)
    quit()


def get_builtin_operator_name(builtin_code):
  """
  Horrific function to reverse the crazy way that flatbuffers stores enums
  it works great for checking specific values, but you have to do this to
  get a value string from an index!
  :param builtin_code: index of builtin operator type
  :return: string of the operator name
  """
  keys = list(BuiltinOperator.BuiltinOperator.__dict__.keys())
  values = list(BuiltinOperator.BuiltinOperator.__dict__.values())
  builtin_name = keys[values.index(builtin_code)]
  return builtin_name


def list_operations(model):

  print("Model subgraph(%d) contains %d operations" %
        (FLAGS.index,
         len(model.subgraphs[FLAGS.index].operators)))

  for i, op in enumerate(model.subgraphs[FLAGS.index].operators):
    operator_code = model.operatorCodes[op.opcodeIndex]
    operator_description = get_builtin_operator_name(operator_code.builtinCode)
    operator_description = operator_description.title()
    if operator_code.customCode is not None:
      operator_description = operator_code.customCode.decode('utf-8')
      operator_description += " (custom)"

    operator_description += " v%d" % operator_code.version

    print("[%3d] %s" % (i, operator_description))


def pre_allocate_memory(model):

  output_filename = FLAGS.pre_allocate_memory[0]
  print("Allocating memory and saving flatbuffer to [%s]" % output_filename)

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

  # create new buffer to hold the metadata content
  metadata_buffer = Buffer.BufferT()
  metadata_buffer.data = bytearray("This is the test metadata. honest!", 'utf8')
  if model.buffers is None:
    model.buffers = []
  new_buffer_idx = len(model.buffers)
  model.buffers.append(metadata_buffer)
  new_metadata.buffer = new_buffer_idx

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

  new_tflite_file = open(output_filename, 'wb')
  new_tflite_file.write(new_model_buffer)
  print("Completed writing modifiled flatbuffer to \"%s\"" % output_filename)


def list_meta_data(model):

  metadata = model.metadata
  if metadata is None:
    metadata = []
  print("Model contains %d meta data entries" % len(metadata))
  for i, m in enumerate(metadata):
    # print("element type : %s" % str(type(m)))
    print("[%d] \"%s\" (%d bytes of data stored in buffer %d)" %
          (i,
           m.name.decode("utf-8"),
           len(model.buffers[m.buffer].data),
           m.buffer))


def main():

  model = load_flatbuffer()

  if FLAGS.operations:
    list_operations(model)

  if FLAGS.pre_allocate_memory:
    pre_allocate_memory(model)

  if FLAGS.meta_data:
    list_meta_data(model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file_name', type=str,
                      help='Name of the tflite flatbuffer file to load.')
  parser.add_argument('-pm', '--pre_allocate_memory',
                      nargs=1, default='',
                      help='Pre allocate intermediate tensor buffers, save'
                           'in the metadata table and write to new flatbuffer')
  parser.add_argument('-i', '--index',
                      type=int, default=0,
                      help='Index of the subgraph to analyse. Defaults to 0')
  parser.add_argument('-o', '--operations',
                      action="store_true",
                      help='Print a summary of the operations used in '
                           'this model.')
  parser.add_argument('-ot', '--op_types',
                      action="store_true",
                      help='Print a summary of the operation types used in '
                           'this model.')
  parser.add_argument('-md', '--meta_data',
                      action="store_true",
                      help="Prints any meta-data members stored in this "
                           "tflite flatbuffer.")
  parser.add_argument('-b', '--buffers',
                      action="store_true",
                      help="Prints a summary of all memory buffers defined "
                           "in this model.")
  """parser.add_argument('-w', '--weights', action="store_true",
                      help='Print detail of the weights of this model.')
  parser.add_argument('-m', '--memory',
                      action="store_true",
                      help='Print details of memory allocation required by'
                           ' this model.')
  parser.add_argument('-t', '--tensors',
                      action="store_true",
                      help='Print details of the tensors used in this model.')
  parser.add_argument('-s', '--save_csv',
                      action="store_true",
                      help='Write the selected detail to a set of CSV files.')
  parser.add_argument('-om', '--optimise_memory',
                      action="store_true",
                      help='Uses various algorithms to precalculate an optimal '
                           'memory useage pattern.')"""

  FLAGS, unparsed = parser.parse_known_args()

  main()
