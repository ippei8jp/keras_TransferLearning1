#!/usr/bin/env python
import sys
import os
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
import numpy as np

# 環境変数設定スクリプトが実行されているか確認
if not "INTEL_OPENVINO_DIR" in os.environ:
    print("/opt/intel/openvino/bin/setupvars.sh が 実行されていないようです")
    sys.exit(1)
else:
    # 環境変数を取得するには os.environ['INTEL_OPENVINO_DIR']
    # これを設定されてない変数に対して行うと例外を吐くので注意
    pass

# openvino.inference_engine のバージョン取得
from openvino.inference_engine import get_version as ov_get_version
ov_vession_str = ov_get_version()
# print(ov_vession_str)               # バージョン2019には '2.1.custom_releases/2019/R～'という文字列が入っている
                                    # バージョン2020には '～-releases/2020/～'という文字列が入っている
                                    # バージョン2021には '～-releases/2021/～'という文字列が入っている

# バージョン判定
if "/2019/R" in ov_vession_str :
    ov_vession = 2019           # 2019.*
elif "releases/2020/" in ov_vession_str :
    ov_vession = 2020           # 2020.*
else :
    ov_vession = 2021           # デフォルト2021

from openvino.inference_engine import IENetwork, IECore

if ov_vession >= 2021 : 
    # バージョン2021以降はngraphを使用
    import ngraph

# ラベルマップ
label_map = [
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


# コマンドラインパーサの構築 =====================================================
def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    parser.add_argument("-d", "--device", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer on; \n"
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.\n"
                             "The demo will look for a suitable plugin \n"
                             "for device specified.\n"
                             "Default value is CPU")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None, 
                        help="Optional.\n"
                             "Required for CPU custom layers. \n"
                             "Absolute path to a shared library\n"
                             "with the kernels implementations.\n"
                             "以前はlibcpu_extension_avx2.so 指定が必須だったけど、\n"
                             "2020.1から不要になった")
    return parser
# ================================================================================
# 結果の解析と表示
def decode_result(net, res, num_result=5) :
    out_blob = list(net.outputs.keys())[0]
    
    # for obj in res[out_blob][0][0]:     # 例：このループは200回まわる
    if hasattr(res[out_blob], 'buffer') :
        # # 2021以降のバージョン
        res_shape = res[out_blob].buffer.shape
        # (1,10)のはず
        res_array = res[out_blob].buffer[0]
    else :
        res_shape = res[out_blob].shape
        res_array = res[out_blob][0]
    # 結果を降順に並べ替え
    sorted_res = np.argsort(res_array)[-num_result:][::-1]
    results = [(i, res_array[i]) for i in sorted_res]

    return results
# ================================================================================

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    model_xml = "./_IR/cifar10/FP16/cifar10.xml"                # モデルファイル名(xml)(決め打ちで)
    model_bin = os.path.splitext(model_xml)[0] + ".bin"         # モデルファイル名(bin)
    
   # 指定されたデバイスの plugin の初期化
    log.info("Creating Inference Engine...")
    ie = IECore()
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading Extension Library...")
        ie.add_extension(args.cpu_extension, "CPU")
    
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    log.info(f"Loading model files:\n\t{model_xml}\n\t{model_bin}\n")
    # 2020.2以降、IENetwork()は非推奨となったため、ie.read_network()に差し替え
    if hasattr(ie, 'read_network') :        # 2020.2以降のバージョン(IECore.read_networkメソッドがあるかで判定)
        net = ie.read_network(model=model_xml, weights=model_bin)
    else :
        net = IENetwork(model=model_xml, weights=model_bin)
    
    # 未サポートレイヤの確認
    if "CPU" in args.device:
        # サポートしているレイヤの一覧
        supported_layers = ie.query_network(net, "CPU")
        ### # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
        ### not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # netで使用されているレイヤ一覧
        if "ngraph" in sys.modules :            # ngraphがインポート済みかで判定
            # バージョン 2021.x以降
            used_layers = [l.friendly_name for l in ngraph.function_from_cnn(net).get_ordered_ops()]
        else :
            # バージョン 2020.x以前
            used_layers = list(net.layers.keys())
        # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
        not_supported_layers = [l for l in used_layers if l not in supported_layers]
        # サポートされていないレイヤがある？
        if len(not_supported_layers) != 0:
            # エラー終了
            log.error(f"Following layers are not supported by the plugin for specified device {args.device}:\n {', '.join(not_supported_layers)}")
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    
    # 入力層の情報出力
    log.info("Preparing inputs")
    img_info_input_blob = None
    if hasattr(net, 'input_info') :        # 2021以降のバージョン
        inputs = net.input_info
    else :
        inputs = net.inputs

    for blob_name in inputs:
        if hasattr(inputs[blob_name], 'shape') :        # 2020以前のバージョン
            input_shape = inputs[blob_name].shape
        else :                                          # 2021以降のバージョン
            input_shape = inputs[blob_name].input_data.shape
        print(f'==== {blob_name}   {input_shape}')
        if len(input_shape) == 4:
            # 入力層でshapeが4のものを取り出す
            input_blob = blob_name
        else:
            # なければエラー
            raise RuntimeError(f"Unsupported {len(input_shape)} input layer '{ blob_name}'. Only 4D input layers are supported")
    
    # 入力画像情報の取得
    if hasattr(inputs[input_blob], 'input_data') :
        input_n, input_colors, input_height, input_width = inputs[input_blob].input_data.shape
    else :
        input_n, input_colors, input_height, input_width = inputs[input_blob].shape
    
    # データ入力用辞書
    feed_dict = {}
    
    # 静止画なので同期モードしか使用しない
    cur_request_id = 0          # 同期モードではID=0のみ使用
    next_request_id = 0         #     〃
    wait_key_code = 0           # 永久待ち
    
    # プラグインへモデルをロード
    log.info("Loading model to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    
    # 推論開始
    log.info("Starting inference...")
    
    print("==== SINGLE MODE ====")
    for file_number in range(100) :
        # 画像の前処理 =============================================================================
        # 入力画像の読み込み
        file_name = f'../img/{file_number}.png'
        # print(f'FILE: {file_name}')
        img = cv2.imread(file_name)
        
        # 入力用フレームの作成
        in_frame = cv2.resize(img, (input_width, input_height))         # リサイズ
        in_frame = in_frame.transpose((2, 0, 1))                        # HWC →  CHW
        in_frame = in_frame.reshape(input_shape)                        # CHW → BCHW
        feed_dict[input_blob] = in_frame
        
        inf_start = time.time()                         # 推論処理開始時刻          --------------------------------
        # 推論予約 =============================================================================
        exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        
        # 推論結果待ち =============================================================================
        while exec_net.requests[cur_request_id].wait(-1) != 0:
            continue
        
        inf_end = time.time()                           # 推論処理終了時刻          --------------------------------
        inf_time = inf_end - inf_start                  # 推論処理時間
        
        # 検出結果の解析 =============================================================================
        if hasattr(exec_net.requests[cur_request_id], 'output_blobs') :        # 2021以降のバージョン
            res = exec_net.requests[cur_request_id].output_blobs
        else :
            res = exec_net.requests[cur_request_id].outputs
        # print(res)
        
        # 結果のデコード
        results = decode_result(net, res, 5)
        # 結果の表示 =============================================================================
        print(f'\n{file_number}\t', end='')
        for k, result in enumerate(results) :
            print(f'{label_map[result[0]]:＿<5},  {result[1]:.5f}    ',end='')
        print(f'    time: {inf_time}',end='')
    
    # 後片付け
    print('')

if __name__ == '__main__':
    sys.exit(main() or 0)
