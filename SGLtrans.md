# Surrogate Gradient Learning in Spiking Neural Networks

## Abstract
スパイキングニューラルネットワーク（SNN）は、効率的で多用途に用いられるソリューションである。これらのモデルをハードウェアに実装するために、様々な研究が行われているが、現実世界の入力をどのように扱い学習するかにおいての課題がいくつか存在する。従来のニューラルネットワーク（ANN）と同様に、現実世界のデータを訓練することは可能だが、バイナリ形式で行われる入出力の特性などから、最適な学習方法や実装方法が一意に定まっていないのが現状である。</br>
この論文では、SNNをトレーニングする際に一般的に遭遇する問題を段階的に説明しシナプス可塑性（Synaptic Plasticity）とオンラインデータ学習法について説明する。その後、これらの問題に対応する画期的な方法である代理勾配法についての説明に移る。

#### 個人的注釈
- SNN
  - 近年になって研究が盛んにされてきているニューラルネットワークのモデルのひとつ。RNNやSNNなどを第２世代NNと呼ぶのに対してこちらは第３世代NNと呼ばれることもある。
- シナプス可塑性
  - あるニューロンが発火することにより、次のニューロンが発火した場合、そのニューロン間どうしの結びつきがさらに強くなるという法則。実際の人間の脳はこの法則性を持っているとされる。これをSNNに適応させた学習速としてSTDP（Spike-Timing Dependent Plasticity）がある。（https://scrapbox.io/AGI/STDP）</br>
  SNNの教師なし学習則として広く用いられている。2018年12月頃にリリースされたSNN用のフレームワークBindsNetにもこの学習則は実装されている。しかしBindsNetにはなぜか教師あり学習則が実装されていない・・・

## Introduction
機械学習の分野では、内部状態が時間とともに変化するNNの一種であるRNNがリアルタイムのパターン認識とノイズの多い系列データの学習に非常に効果的であることがわかっています。この類似性に基づき、RNNとSNNの積分発火モデル（LIFモデル）を組み合わせたニューロンモデルが提案されてきている。
1. F. Zenke and S. Ganguli, “SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks,”
Neural Computation, vol. 30, no. 6, pp. 1514–1541, Apr. 2018.
2. G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass, “Long short-term memory and
Learning-to-learn in networks of spiking neurons,” in Advances in Neural Information ProcessingSystems 31, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds. Curran Associates, Inc., 2018, pp. 795–805.
3. J. Kaiser, H. Mostafa, and E. Neftci, “Synaptic plasticity for deep continuous local learning,” arXiv preprint arXiv:1812.10766, 2018.
4. A. Tavanaei, M. Ghodrati, S. R. Kheradpisheh, T. Masquelier, and A. Maida, “Deep learning in
spiking neural networks,” Neural Networks, Dec. 2018.

大規模RNNによる大規模データ学習は、時系列ノイズや空間的依存性により困難な場面が多い。SNNやバイナリRNNにおいても同様であり、出力のバイナリ性質によってさらに困難になる。</br>
隠れ層ユニットなしの2層SNNでは、多くの効率的なトレーニングによる結果が報告されているが、大規模データの学習には隠れ層ユニットや層の深さは重要な要素であるため、隠れ層ユニットを持つSNNにおける学習課題を克服することは不可欠である。</br>
ネットワークモデルが大きくなり、組み込み型アプリケーションや自動車アプリケーションにそれらを移すことを考えるとき、そのモデルの電力効率はますます重要になってくる。こういった大規模ハードウェア開発が進んできていることもあり、エネルギー効率の良いSNNやバイナリRNNの研究需要というのは大きくなってきている。
次の項より、SNNの隠れ層を学習する際に困難になる理由、それらをうまく実装するために使用される様々な戦略や近似について紹介していく。

#### 個人的注釈
- LIF（Leaky Integrate and Fire）モデル
  - SNNのニューロンモデルのひとつ。時々刻々と膜電位が上昇していきニューロンの閾値を超えた場合に次のニューロンへの発火を伝える様子を簡略化したモデル。</br>
  この他にもHH（Hodgkin-Haxley）モデル＜一番複雑！＞や、Izhikevichモデル＜あまり使われない＞などがある。
- ハードウェア実装
  - 多くはハードウェア言語によってFPGAなどに学習モデルを載せることを指す。バイナリで計算する利点として、ハードウェアで使用する乗算器が少なく済むという利点があるらしい。

## Understanding SNNs As RNNs
SNNをRNNの一種としてマッピングすることから始める。RNNとしてSNNを定式化することは、RNNの既存の学習則をSNNに適用させることにおいて重要な作業となる。また、概念的な学習則の理解のためにも役立つ。筆者らは、今後RNNという用語を広い意味で利用していく。具体的には、その状態が時間的に変化し、内部状態が回帰的な動的方程式を用いて表せるネットワーク全般を指すこととする。一般的なRNNの理解は、再帰結合を持ったネットワークがRNNであるというケースが多い。しかし、本質的な「再帰（リカレンス）」とは、再帰結合がない場合にも起こり得る。例えば、動的な内部状態を持つニューロンモデルを考えてみると、特定の時刻の状態（膜電位）は、前時刻の状態に依存したものになっている。本論文ではこのどちらのモデルにおいてもRNNという用語を適用することとする。（つまり一般的なSNNもRNNと呼ぶこともあるよということ）さらに、SNNとRNNの区別のために、再帰結合を持つネットワークはRCNN（Reccurrently Connected Neural Network）よ呼ぶことを提案する。