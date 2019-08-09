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
  - この実験条件での入力データのサイズはtorch.Size([62, 32, 3])である。[系列、バッチサイズ、入力次元]の並びとなっている。これより、time_sizeは最大でも62までしか設定できないことがわかる。現状のプログラムでは、例えばtime_sizeが10の場合、最後に残った2系列分は学習を行わないようになっているため、入力系列に対して余りが0、または少なくなるように展開時間サイズを設定することが推奨される。
- 誤差関数（MSE）
  - 各時系列に対してMSEで誤差を取る。逆伝搬を行う誤差loss_valは、time_sizeが10の場合、10系列の誤差の合計に対してbackward()を実行する。実行後、グラフプロット用の変数total_lossに足したあと、loss_valは0にリセットする。全系列が62の場合、この操作が6回行われる。total_lossは、残りの2系列分のロスが切り捨てられていることになる。そのため、切り捨て系列に違いがあるとtotal_lossグラフによる比較は、正確にはフェアな条件ではない。以下のLossグラフはそうなってしまっているため、あとで直すかも。

### 実験結果

パラメータを変えて変化を検証してみる。

#### 学習率変化
|変数名|意味|値|
|---|---|---|
|learning_rate|学習率|Variable|
|epoch|学習回数|1000|
|time_size|展開時間サイズ|6|

#### 学習
1e-2が一番収束が早いが、最終的なlossは1e-3が最も小さい。

<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/loss0-1000.png" alt="0-1000">

<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/loss700-1000.png" alt="700-1000">

#### 推論
- lr=1e-2(0.01)：予測周期以降少し振幅と周波数が大きくなってしまいズレている。
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/1e-2.png" alt="lr=1e-2" title="lr=1e-2">

- lr=1e-3(0.001)：一番真値に近い予測ができている。
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/1e-3.png" alt="lr=1e-3" title="lr=1e-3">


- lr=1e-4(0.0001)：振幅がかなり小さくなっている。
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/lr1e-4.png" alt="lr=1e-4" title="lr=1e-4">

#### 学習回数変化
|変数名|意味|値|
|---|---|---|
|learning_rate|学習率|1e-3|
|epoch|学習回数|Variable|
|time_size|展開時間サイズ|6|

#### 推論

- epoch=10：ほとんど周期や振幅を記憶できていない。
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/epoch10.png" alt="epoch10">

- epoch=100：このあたりで大まかな傾向は学習できているっぽい。
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/epoch100.png" alt="epoch100">

#### 展開時間サイズ変化
|変数名|意味|値|
|---|---|---|
|learning_rate|学習率|1e-3|
|epoch|学習回数|1000|
|time_size|展開時間サイズ|Variable|

#### 学習
全体的に時間展開サイズが大きくなるほど必要な学習回数は増えていく傾向がある。時間展開サイズが大きくなれば学習する必要があるネットワークが大きくなるのと同義なので当たり前の結果と言えそう。

<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/learn_1-60_0-1000.png" alt="1-60_0-1000">

<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/learn_1-10_1-1000.png" alt="1-10_0-1000">

<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/learn_1-30_700-1000.png" alt="1_30_700-1000">

<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/learn_1-10_400-1000.png" alt="1-10_400-1000">

#### 推論
ノイズもほとんど入っていない（整数に直すときに少数が切り落とされるのが強いていえばノイズ）正弦波予測なので、実は時系列的な記憶が必要ない簡単な問題ということもありどの時間展開サイズでもあまり変わらない。time_size=1では完全に収束したはずだがそこまで予測波形が綺麗ではないことを考えると、簡単とはいえ多少時間的な記憶をしたほうが良いみたい。time_size=10やtime_size=30あたりが一番正確な予測となっている。time_size=60では単純にEpoch1000ではまだ学習が足りてないかもしれない。

- time_size=1
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/pre_time1.png" alt="pre_time1">

- time_size=2
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/pre_time2.png" alt="pre_time2">

- time_size=10
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/pre_time10.png" alt="pre_time10">

- time_size=30
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/pre_time30.png" alt="pre_time30">

- time_size=60
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/pre_time60.png" alt="pre_time60">
