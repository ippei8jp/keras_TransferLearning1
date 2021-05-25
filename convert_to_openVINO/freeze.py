'''
saved_modelをfrozen-graphに変換するスクリプト
https://github.com/openvinotoolkit/openvino/issues/4830
'''

import sys
import os

# コマンドラインパラメータ
args = sys.argv

if len(sys.argv) != 3 :
    print( "==== USAGE ====")
    print(f"    python {sys.argv[0]} <SAVED_MODEL_DIR> <FROZEN_MODEL_FILENAME>")
    sys.exit(1)

# "simple_frozen_graph.pb"

SAVED_MODEL_DIR       = args[1]
FROZEN_MODEL_DIR      = os.path.dirname(args[2])
FROZEN_MODEL_FILENAME = os.path.basename(args[2])

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model = tf.saved_model.load(SAVED_MODEL_DIR)
graph_func = model.signatures['serving_default']

# Get frozen ConcreteFunction 
frozen_func = convert_variables_to_constants_v2(graph_func)
frozen_func.graph.as_graph_def()

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def = frozen_func.graph, logdir = FROZEN_MODEL_DIR, name = FROZEN_MODEL_FILENAME, as_text = False)


