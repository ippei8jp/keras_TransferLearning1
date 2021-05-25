# from operator import iconcat
# from numpy.core.records import array
import tensorflow as tf

import sys
import os
import numpy as np

import time

# 画像ディレクトリ
IMG_DIR = './img'

# モデルファイル
MODEL_FILE = './cifar10.h5'

# 動作パラメータ
DO_TEST = False
SAVE_IMG = False

DISP   = False
BATCH = False

# メンドクサイからこんなんでいっか。
if len(sys.argv) > 1 :
    # パラメータが何か指定されていたらバッチモード
    BATCH = True

if DISP :
    import  cv2

# 画像ディレクトリがなけれな作成してファイルを保存
if not os.path.exists(IMG_DIR) :
    os.makedirs(IMG_DIR)
    SAVE_IMG = True

# 実行毎に結果が変わらないように乱数の種を指定
tf.random.set_seed(0)

# モデルの読み込み
print('Loading model...')
model = tf.keras.models.load_model(MODEL_FILE)
print('Done.')

# モデル構成を表示してみる
model.summary()

if SAVE_IMG or DO_TEST :
    # テストデータの準備
    # 面倒なので、CIFAR-10データを使う
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

if SAVE_IMG :
    from PIL import Image
    # 画像データを保存
    print("Saving Image file...")
    for x in range(100) :
        file_name = os.path.join(IMG_DIR, f'{x}.png')
        print (file_name)
        Image.fromarray(x_test[x]).save(file_name)

if DO_TEST :
    # テスト
    test_result = model.evaluate(x_test, y_test, verbose=1)
    print(test_result)
    # 学習時の最終テストの値を同じになるはず
    # 大体こんな感じ [0.3642473518848419, 0.8762999773025513]

# 外部画像の認識 =================================================================
# 結果ラベル
labels = [
    "飛行機",
    "自動車",
    "鳥",
    "猫",
    "鹿",
    "犬",
    "カエル",
    "馬",
    "船",
    "トラック"
]

# 認識結果のデコード
def decode_predects(predicts, top=5) :
    results = []
    for predict in predicts:
        # ソートして最後の値が最大値なので、top～最後を取り出して降順に並べ替え
        top_indices = predict.argsort()[-top:][::-1]
        # result = [(labels[i], predict[i]) for i in top_indices]
        result = [(i, predict[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

if not BATCH :
    # 単体評価
    print("==== SINGLE MODE ====")
    for i in range(100) :
        # 画像ファイルの読み込み
        file_name = os.path.join(IMG_DIR, f'{i}.png')
        img_pil = tf.keras.preprocessing.image.load_img(file_name)
        
        # 前処理
        img_array = tf.keras.preprocessing.image.img_to_array(img_pil)
        # 転移学習したモデルなのでpreprocess_input()は やらない
        # img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[tf.newaxis, ...])
        img_preprocessed = img_array[tf.newaxis, ...]
        
        # 認識
        pred_start_time = time.time()
        predicts = model.predict(img_preprocessed)
        pred_time = time.time() - pred_start_time           # 時間計測
        
        # 結果表示
        results = decode_predects(predicts, top=5)

        print(f'\n{i}\t', end='')
        for k, result in enumerate(results[0]) :
            print(f'{labels[result[0]]:＿<5},  {result[1]:.5f}    ',end='')
        print(f'    time: {pred_time}',end='')

        if DISP :
            # 表示してみる
            img_cv = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)     # RGB→BGR
            cv2.imshow(f"IMG", img_cv)                                        # 表示
            
            # キー入力待ち
            while True :
                key = cv2.waitKey(1)
                if key != -1 :
                    break
                if key == 27:
                    # ESCキー
                    break
    # 後片付け
    print('')
    if DISP :
        cv2.destroyAllWindows()
else :
    print("==== BATCH MODE ====")
    for j in range(10) :            #画像ファイルをバッチサイズで分割してループ
        img_batch = None
        for i in range(10) :        # バッチサイズ10で0枚まとめて実行
            # 読み込み
            file_name = os.path.join(IMG_DIR, f'{i + j*10}.png')
            # print(file_name)
            img_pil   = tf.keras.preprocessing.image.load_img(file_name)
            # 配列化
            img_array = tf.keras.preprocessing.image.img_to_array(img_pil)
            img_tmp = img_array[tf.newaxis, ...]
            # バッチリスト化
            if img_batch is None :
                img_batch = img_tmp
            else :
                img_batch = np.append(img_batch, img_tmp, axis=0)
            
        # 前処理
        # 転移学習したモデルなのでpreprocess_input()は やらない
        # img_preprocessed   = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        img_preprocessed   = img_batch
        
        # 認識
        pred_start_time = time.time()
        predicts = model.predict(img_preprocessed)
        pred_time = time.time() - pred_start_time           # 時間計測
        
        # 結果表示
        results = decode_predects(predicts, top=5)
        for i in range(10) :
            print(f'\n{i + j*10}\t', end='')
            for k, result in enumerate(results[i]) :
                print(f'{labels[result[0]]:＿<5},  {result[1]:.5f}    ',end='')
        print(f'    time: {pred_time}',end='')
    
    print('')

# ================================================================================
