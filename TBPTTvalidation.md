# Trancated BPTT in SNN & RNN

## 概要
学習手法としてTrancated BPTTを使用して、正弦波の学習、予測を行う。SNNとRNNどちらも実装してみて学習性能の比較検証を行う。

## TBPTT in RNN
### ネットワーク&学習手法etc
ネットワークは3層全結合フィードフォワード型のNNを採用する。実験中に変更するパラメータはVariableと表記している。

- ネットワーク

|変数名|意味|値|
|---|---|---|
|units_input|入力層ユニット|3|
|units_hidden|隠れ層ユニット|10|
|units_output|出力層ユニット|1|

- 学習手法

|変数名|意味|値|
|---|---|---|
|optimizer|重み更新手法|Adam|
|learning_rate|学習率|Variable|
|epoch|学習回数|Variable|
|time_size|展開時間サイズ|Variable|

- 学習データ&予測データ

|変数名|意味|値|
|---|---|---|
|steps_per_cycle|1周期ごとのデータ点数|100|
|number_of_cycles|学習データの総合周期数|20|
|batch_size|バッチサイズ|32|
|init_cycles|初期点として与える周期数|1|
|pre_cycles|初期点以降の予測周期|2|

#### 補足
- input_datanum（入力総データ数）
  - 入力総データ数は（steps_per_cycles*number_of_cycles-units_input）によって1997と求められる。</br>
  総データのうち、最初の入力分の3データは教師値として用いることは無く、最後のデータは入力値として用いることはない。
- sin_input（入力データ）
  - この実験条件での入力データのサイズはtorch.Size([62, 32, 3])である。[系列、バッチサイズ、入力次元]の並びとなっている。これより、time_sizeは最大でも62までしか設定できないことがわかる。現状のプログラムでは、例えばバッチサイズが10の場合、最後に残った2系列分は学習を行わないようになっているため、入力系列に対して余りのないように展開時間サイズを設定することが推奨される。

### 実験結果

パラメータを変えて変化を検証してみる。

|変数名|意味|値|
|---|---|---|
|learning_rate|学習率|2e-4|
|epoch|学習回数|1000|
|time_size|展開時間サイズ|Variable|

<div align="center">
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/1e-3.png" alt="autogradの計算グラフ" title="autogradの計算グラフ">
</div>
