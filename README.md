# 概要
Keras(on Tensorflow2)環境での転移学習の手順を試してみた。  

# 参考
以下のサイトの前半部分をトレースしています。  
<https://note.nkmk.me/python-tensorflow-keras-transfer-learning-fine-tuning/>  

Jupter notebookだと何かと不便なので、pythonプログラムで実行してます。  

# 環境準備


## このリポジトリをcloneする

## python仮想環境設定
```
pyenv virtualenv 3.7.10 tensorflow1_py37
pyenv local tensorflow1_py37
pip install --upgrade pip setuptools
pip install tensorflow
pip install pillow
```
pythonのバージョンはローカルでのみ実行する場合は最新でも良いですが、  
Google Colabで学習を実行する場合はGoogle Colabで使用しているpythonのバージョンに合わせてください。  
異なるバージョンで実行するとKeras modelロード時にエラーが発生します。  
(SavedModelを使用すればバージョンが異なっても問題なく実行できます)  

# 転移学習の実行
## ローカル環境で実行する場合
ローカル環境で``cifar10_train.py``を実行すると、mobilenetV2のimagenetでの学習済みモデルをcifar-10で転移学習し、  
結果を``cifar10.h5``ファイルと``saved_model``ディレクトリに出力します。  

## Google Colabで実行する場合
ローカル環境で実行しても、何日もかかるようなことはありませんが、やはり さくっと終わるようなことはありません。  
そんなときは、Google Colabで実行すると短時間で実行できます。  
Google Colabで実行するには、```colab/keras_TransferLearning1.ipynb` を Google Colab にアップロードし、実行します。  
中身は、このリポジトリをcloneし、``cifar10_train.py``を実行、結果ファイルをzipファイルに圧縮しているだけです。  
実行完了したらzipファイルをダウンロードして展開して使用します。  

試したときは、ローカルで1.5時間程度かかったものが、Colab環境では10数分で終わりました。  
(スペックや環境によって異なるので、目安でしかありませんが)  

# ローカル環境でのテスト

## ダウンロードしたモデルファイルを展開する
Google Colabで実行した場合はダウンロードしたファイルを展開しておきます。  
モデルファイルのpathはプログラム内で固定なので、展開位置に注意してください。  

```
unzip result.zip
```

## テスト

モデル学習時に使用したテストデータから100個を画像ファイルとして保存(初回のみ)し、  
それらを使用して認識処理を実行してみます。  

```
# ファイル1個ずつ認識実行(KerasModel使用)
python cifar10_predict.py

# ファイル10個をまとめて認識実行(KerasModel使用)
python cifar10_predict.py 1

# ファイル1個ずつ認識実行(SavedModel使用)
python cifar10_predict_saved_model.py

# ファイル10個をまとめて認識実行(SavedModel使用)
python cifar10_predict_saved_model.py 1

```
# openVINOモデルへの変換

## 準備
python 仮想環境はopenVINO環境に切り替えておく。  
(Tensorflow2を使用するpython3.8版を使用)  
```
cd convert_to_openVINO
pyenv local openvino_py38
```

## モデル変換

```
bash convert.sh
```
_IRディレクトリ以下に変換されたモデルが出力される。  

## テスト

```
python _test.py
```

