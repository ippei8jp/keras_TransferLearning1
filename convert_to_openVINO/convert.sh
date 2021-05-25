#!/usr/bin/bash

# pyenvでpythonのバージョンを切り替えたときの問題の対策  ==================================
## 現在のpythonのバージョン取得
# TMP_PYVER=`python -V |  sed -e "s/^.*\(3.[0-9]\{1,\}\).*$/\1/g"`
TMP_PYVER=`python -c "import sys; v = sys.version_info; print(f'{v[0]}.{v[1]}')"`
## PYTHONPATHの該当箇所を置換
export PYTHONPATH=`echo $PYTHONPATH  | sed -e "s/\/python3\.[0-9]\{1,\}/\/python${TMP_PYVER}/g"`
# =========================================================================================

MODEL_NAME=cifar10
SAVED_MODEL=../saved_model
# エラーになるときはFrozen modelに変換して試してみてください
### FROZEN_MODEL=./cifar10_forzen_graph.pb

INPUT_NODE='mobilenetv2_1_00_160_input'
INPUT_SHAPE="[1,160,160,3]"
#### OUTPUT_NODE="dense/Softmax:0"      # これじゃない。どうやって調べる？

# frozen modelへの変換
# (saved model からの変換がエラーになる場合はfrozen modelに変換してから実行を試す)
if [[ -v ${FROZEN_MODEL} ]] ; then          # FROZEN_MODELが定義されている
    if [ ! -e ${FROZEN_MODEL} ]; then       # FROZEN_MODELファイルが存在してない
        python freeze.py ${SAVED_MODEL} ${FROZEN_MODEL}     # SavedModelからFrozenModelに変換
        if [ $? -ne 0 ]; then
          # エラー
          exit 1
        fi
    fi
fi

# モデルオプティマイザオプション設定
OPTIONS=" --framework=tf"
# OPTIONS+=" --log_level=DEBUG"       # logレベル
OPTIONS+=" --data_type=FP16"
OPTIONS+=" --reverse_input_channels"
OPTIONS+=" --input_shape=${INPUT_SHAPE}"
OPTIONS+=" --input=${INPUT_NODE}"
## OPTIONS+=" --output=${OUTPUT_NODE}"          # 設定しなければモデルから自動に設定される
OPTIONS+=" --model_name=${MODEL_NAME}"
if [[ -v ${FROZEN_MODEL} ]] ; then
    OPTIONS+=" --input_model=${FROZEN_MODEL}"       # FROZEN_MODELが定義されていればFrozenModelから変換
else
    OPTIONS+=" --saved_model_dir=${SAVED_MODEL}"    # そうでなければSavedModelから変換
fi
OPTIONS+=" --output_dir=./_IR/${MODEL_NAME}/FP16"   # 出力先

# 実行
echo ${OPTIONS}
${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py ${OPTIONS}
