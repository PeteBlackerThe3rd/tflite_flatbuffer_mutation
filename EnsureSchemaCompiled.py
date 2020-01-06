
import os
import array
import hashlib as hl
from shutil import which
import subprocess as subproc

def compile_schema(schema_filename, checksum_filename):
  # check flatcc tool is installed
  flatcc_installed = which('flatc') is not None

  print("One time compilation of flatbuffer schema, python interface.")

  if flatcc_installed:
    print("flatbuffer compiler flatc found okay.")
  else:
    # TODO add installation instructions
    print("flatbuffer compiler not installed. Add installation instructions.")

  compile_schema_cmd = 'flatc --python --gen-object-api -o test/ %s' % \
                       schema_filename

  try:
    subproc.check_output(compile_schema_cmd, shell=True)

    # update schema checksum so that this is only re-compiled
    # if the schema has changed.
    checksum = hl.md5(open(schema_filename, 'rb').read()).hexdigest()
    checksum_bytes = bytearray(checksum, 'utf8')
    open(checksum_filename, 'wb').write(checksum_bytes)

  except subproc.CalledProcessError as e:
    print("Error running flatc tool: %s" % format(e))


def ensure_schema_compiled():
  schema_filename = 'schema/schema.fbs'
  checksum_filename = "tflite_schema.md5"
  compile = False

  # if the schema checksum file doesn't exist then it definitely needs compiling
  if not os.path.isfile(checksum_filename):
    compile = True
  else:
    # calculate the md5 checksum of the current schema and compare to
    # checksum file
    checksum = hl.md5(open(schema_filename, 'rb').read()).hexdigest()
    checksum_bytes = bytearray(checksum, 'utf8')
    checksum_file = open(checksum_filename, 'rb').read()
    if checksum_bytes != checksum_file:
      compile = True

  if compile:
    compile_schema(schema_filename, checksum_filename)

# call the function on library import.
ensure_schema_compiled()