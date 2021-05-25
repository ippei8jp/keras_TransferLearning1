import tensorflow as tf

import sys
import numpy as np
import pprint

# 実行毎に結果が変わらないように乱数の種を指定
tf.random.set_seed(0)

# CIFAR-10データのダウンロードと保存
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# データのフォーマットと個数を確認
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
    # (50000, 32, 32, 3) (50000, 1)
    # (10000, 32, 32, 3) (10000, 1)

# リサイザを入力層に
inputs = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (160, 160)))(inputs)
x = tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)(x)

# ベースモデルの生成
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    weights='imagenet', input_tensor=x, input_shape=(160, 160, 3),
    include_top=False, pooling='avg'
)

# モデルの生成
# 出力層は10ノードのDenseレイヤー、活性化関数はsoftmax
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
    # Model: "sequential"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #   
    # =================================================================
    # mobilenetv2_1.00_160 (Model) (None, 1280)              2257984   
    # _________________________________________________________________
    # dense (Dense)                (None, 10)                12810     
    # =================================================================
    # Total params: 2,270,794
    # Trainable params: 2,236,682
    # Non-trainable params: 34,112
    # _________________________________________________________________

# ベースモデルは学習しない(Denseレイヤのみ)
base_model.trainable = False

model.summary()
    # 学習しない設定をしたので、Trainable paramsが減っている
    # Model: "sequential"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #   
    # =================================================================
    # mobilenetv2_1.00_160 (Model) (None, 1280)              2257984   
    # _________________________________________________________________
    # dense (Dense)                (None, 10)                12810     
    # =================================================================
    # Total params: 2,270,794
    # Trainable params: 12,810
    # Non-trainable params: 2,257,984
    # _________________________________________________________________

# モデルのコンパイル 
#      最適化関数： RMSprop  学習率：0.0001    
#      損失関数：交差エントロピー誤差   
#      評価関数:正解率
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# とりあえず学習せずにテストしてみる
test_result = model.evaluate(x_test, y_test, verbose=1)
print(test_result)
    # デタラメな感じの結果がでる
    # [2.922475576400757, 0.11320000141859055]

# 学習の実行(評価データは学習データを分割して使用する)
model.fit(x_train, y_train, epochs=6, validation_split=0.2, batch_size=256)

# テスト
test_result = model.evaluate(x_test, y_test, verbose=1)
print(test_result)
    # こんな感じ、正解率を もうちょっと良くしたい
    # [0.5055754780769348, 0.8306000232696533]

# ベースモデル内のレイヤをリストアップ
layer_names = [l.name for l in base_model.layers]
# block_12_expandレイヤのインデックスを取得
idx = layer_names.index('block_12_expand')
print(idx)

# ベースモデル全体を学習許可に
base_model.trainable = True

# ベースモデルの先頭からblock_12_expandの前までを学習しない設定に
# (block_12_expandレイヤ以降を学習する)
for layer in base_model.layers[:idx]:
    layer.trainable = False

# trainableを変更したので再コンパイル
# 学習率は小さくしておく
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
    # 学習するレイヤを変更したので、Trainable paramsが変わっている
    # Model: "sequential"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #   
    # =================================================================
    # mobilenetv2_1.00_160 (Model) (None, 1280)              2257984   
    # _________________________________________________________________
    # dense (Dense)                (None, 10)                12810     
    # =================================================================
    # Total params: 2,270,794
    # Trainable params: 1,812,426
    # Non-trainable params: 458,368
    # _________________________________________________________________

# 再度学習
model.fit(x_train, y_train, epochs=6, validation_split=0.2, batch_size=256)

# テスト
test_result = model.evaluate(x_test, y_test, verbose=1)
print(test_result)
    # ちょっと良くなった？ま、手順の確認なので...
    # [0.3642473518848419, 0.8762999773025513]

# モデルの保存 ===================================================================
print("Save model...")
# HDF5形式で保存(モデル/重み)
model.save("cifar10.h5")

# Tensorflow SavedModel形式で保存
model.save("saved_model", save_format = "tf")
print("Done.")
# ================================================================================
