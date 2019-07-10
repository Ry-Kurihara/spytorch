# SpyTorchの日本語解説&実装
## 代理勾配法によるSNNの学習
### 要項
Tutorial1～Tutorial3までコメントアウトなどで細かく動作を確認しつつ進めていく。基本的には自分が後から読んでわかるように書いている。

#### 時系列処理
Tutorial4として、クラス分類問題以外に、系列データの学習を実装する予定。</br>
PyTorchでの一般的な系列データ入力方法は、Tensorを（バッチサイズ、時系列シーケンス、入力次元数）として入力する。LSTMクラスを採用したネットワークに対する入力として代表的である。LSTMCellクラスを使用した場合は、時間ステップごとに入力する必要があり、ループを用いて適した時系列シーケンスの配列を取り出していくため、（バッチサイズ、入力次元数）での入力になる。</br>
しかし、SNNではひとつのデータに対してひとつのスパイクパターンの配列が用意されるため、次元数が1次元増える問題がある。</br>
Tutorial2の入力層への入力は（256, 100, 784）となっている。ここでの100は時系列長では無く、タイムステップ（時間窓）である。つまりクラス分類問題であれば時系列長とタイムステップを同等とみなすことで、LSTMクラスのときと同様の配列を入力することができる。</br>
しかし系列データを扱う場合、SNNでは4次元必要になる。このときの入力方法をどうするかを思案中

### リポジトリ内ファイル
- notebooks/
  - Spytorchチュートリアル。日本語コメントアウト。
- SGLtrans.md
  - 元論文の日本語訳&個人的注釈

### 進捗
Tutorial2: 計算グラフの伝搬</br>
Tutorial1,2: ctxオブジェクトによるautogradの保存</br>
Tutorial3: 読み込み中</br>
SGLtrans.md: 読み込み&まとめ中

## ↓↓↓フォーク元↓↓↓

## SpyTorch
A tutorial on surrogate gradient learning in spiking neural networks

Version: 0.2

This repository contains tutorial files to get you started with the basic ideas
of surrogate gradient learning in spiking neural networks using PyTorch.

Feedback and contributions are welcome.

> For more information on surrogate gradient learning please refer to:
> Neftci, E.O., Mostafa, H., and Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks.
> https://arxiv.org/abs/1901.09948

Also see https://github.com/nmi-lab/dcll/tree/master/tutorials

## Copyright and license

Copyright 2019 Friedemann Zenke, https://fzenke.net

This work is licensed under a Creative Commons Attribution 4.0 International License.
http://creativecommons.org/licenses/by/4.0/
