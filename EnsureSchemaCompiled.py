
import os
import array
import hashlib as hl
from shutil import which
import subprocess as subproc

schema_filename = 'schema/schema.fbs'
checksum_filename = 'tflite/schema.md5'


def compile_schema():
  # check flatcc tool is installed
  flatcc_installed = which('flatc') is not None

  print("One time compilation of tflite flatbuffer schema, python interface.")

  if flatcc_installed:
    print("flatbuffer compiler flatc found okay.")
  else:
    print("Error: Flatbuffers not installed, schema cannot be compiled.\n"
          "Please follow the installation instructions at "
          "https://google.github.io/flatbuffers/flatbuffers_guide_building.html"
          "\nto install the flatbuffers package then re-try.")
    quit(1)

  compile_schema_cmd = 'flatc --python --gen-object-api %s' % \
                       schema_filename

  try:
    subproc.check_output(compile_schema_cmd, shell=True)

    # update schema checksum so that this is only re-compiled
    # if the schema has changed.
    checksum = hl.md5(open(schema_filename, 'rb').read()).hexdigest()
    checksum_bytes = bytearray(checksum, 'utf8')
    open(checksum_filename, 'wb').write(checksum_bytes)

    print("successfully compiled tflite flatbuffer schema.")

  except subproc.CalledProcessError as e:
    print("Error running flatc tool: %s" % format(e))


def ensure_schema_compiled():
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
    compile_schema()


# call the function on library import.
ensure_schema_compiled()
