# Pytorch 実装

## 概要
- 自動微分
- 関数

## ネットワーク
- nn.Module

### nn.RNN
(seq_len, batch, input_size)の形状のテンソルを入力する。通常のnn.Linearのみだけで構成させるネットワークと比較すると、系列をまとめて入力できる利点がある。入力系列が時々刻々と変化していく場合にはnn.RNNは適さないかも。また、引数batch_firstをTrueにすると(batch, seq_len, input_size)として入力でき、Tensor.transpose(n,m)でわざわざ次元入れ替えをする必要がなくなる。

### パラメータの更新
- for文を用いた直接更新

nn.Moduleを継承して作成したネットワークのインスタンスに対してparametersメソッドを使うと、ネットワークの学習パラメータのみ抜き出すことが出来る。内部的にどんな処理が行われているかの可読性が上がるメリットがある。SGDなどの最適化手法であれば簡潔に記述できるが、Adamなど複雑な最適化手法ではかえって可読性が下がる可能性もある。

```
SGDを用いたパラメータ更新例

model = LSTM(4, 5, 3)

例1
for param in model.parameters():
  param.data -= learnng_rate * param.grad.data

例2
for param in model.parameters():
  param.data.add(-learning_rate, param.grad.data)
### param.data = param.data + -learning_rate*param.grad.data
```

- torch.optimを使った更新

torch.optim.（Adam, Adamax, etc）クラス？にmodel.parameters()を引数としてインスタンスを作成し、更新を行っていく。こちらの書き方のほうが良く用いられている。

```
Adamを用いたパラメータ更新例

model = LSTM(4,5,3)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for e in range(Epoch):
  ---
  ---
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```

## 自動微分
- torch.autograd

Pytorchにおける自動識別用のコアパッケージ。順方向フェーズで実行する操作を記憶し、逆方向フェーズで操作を再生する。
Torchクラスの引数requieres_grad=Trueと指定することで追跡が可能になる。

```
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#tensor([1., 2., 3.], requires_grad=True)

b = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
#tensor([2., 3., 4.], requires_grad=True)

f = torch.dot(a, b)
#tensor(20., grad_fn=<DotBackward>)

f.backward()

print(a.grad)
#tensor([2., 3., 4.])

print(b.grad)
#tensor([1., 2., 3.])

print(f.grad)
#None
```

この流れを図で表すと以下のような感じ。手書きですが、、、この計算グラフの描き方はゼロから作るDeepLearningを参考にしています。

- 加算ノード
  - 上流をそのまま下流に流す
- 乗算ノード
  - 分岐ルートの値を上流からきた値と乗算して下流に流す

<div align="center">
<img src="https://github.com/Ry-Kurihara/spytorch/blob/images/pytorchgrad1.png" alt="autogradの計算グラフ" title="autogradの計算グラフ">
</div>

## 関数
#### torch.squeeze
要素が1の次元を削減する

### 詰まったところ

- torch.max

torch.maxでは、2要素の返り値があるが、indicesプロパティのほうには計算履歴が残らない。

```
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

>>>torch.max(a, 0)
#torch.return_types.max(values=tensor(3., grad_fn=<MaxBackward0>), indices=tensor(2))
```

- torch.Tensor.numpy()

この関数では、計算グラフを持ったTensorに適用することができない。以下のようなエラー文が出る。なので、Tensor.detach()によって計算グラフの情報を消去してからnumpyへ変換することが推奨される。

```
>>>tensor1
#tensor([ 71.,  79.,  84.,  91.,  97., 101., 104., 107., 108., 109., 108., 106.,
        106., 103., 100.,  95.,  90.,  84.,  77.,  70.,  64.,  54.,  47.,  39.,
         34.,  29.,  22.,  12.,   9.,   5.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   2.,   7.,  15.,  21.,  25.,  35.,  43.,  48.,
         54.,  64.,  71.,  79.,  84.,  91.,  97., 101., 104., 107., 108., 109.,
        108., 106., 106., 103., 100.,  95.,  90.,  84.,  77.,  70.,  64.,  54.,
         47.,  39.,  34.,  29.,  22.,  12.,   9.,   5.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   2.,   7.,  15.,  21.,  25.,  35.,
         43.], grad_fn=<CopyBackwards>)

>>>tensor1.numpy()
#RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
```

- torch.Tensor.backward()

1回呼び出した情報をもう一回呼び出すとエラーになる。エラーを回避するためには引数にretain_graph=Trueを設定する必要がある。

```
>>>f.backward()
#Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

###純粋に足されていく
>>>f.backward(retain_graph=True)
>>>w.grad
#tensor(8.0)
>>>f.backward(retain_graph=True)
>>>w.grad
#tensor(16.0)
```

- 特定の範囲の値が入ったインデックスに対して操作する

```
inpt = torch.randn(3,5, dtype=dtype)
>>>inpt
#tensor([[-0.0678, -1.1650,  0.4382, -0.0260,  1.3923],
        [-0.7038,  0.3360,  1.0169, -1.1304,  0.3695],
        [-0.0491, -1.4775, -2.2848,  0.9594, -0.2495]])

outen = torch.zeros_like(inpt)
outen[[-5.0<inpt] and [inpt<-1.0]] = 8.0 #-5.0 < inpt < -1.0とは指定できない
>>>outen
#tensor([[0., 8., 0., 0., 0.],
        [0., 0., 0., 8., 0.],
        [0., 8., 8., 0., 0.]])
```

- torch.topk()

Tensor中の最大値とそのインデックスを返す。引数で昇順に数値をいくつ見つけるかを指定する。

```
fi = torch.randn(4,3)
>>>fi
#tensor([[-0.0879,  1.2384,  0.3714],
        [ 0.6582,  0.4486,  1.1292],
        [ 0.7334, -1.0504,  0.9428],
        [ 1.9731, -0.2247,  0.6539]])

>>>fi.topk(2)
#torch.return_types.topk(values=tensor([[1.2384, 0.3714],
        [1.1292, 0.6582],
        [0.9428, 0.7334],
        [1.9731, 0.6539]]), indices=tensor([[1, 2],
        [2, 0],
        [2, 0],
        [0, 2]]))
```

- torch.cat()

Tensorの連結dimで連結する次元を指定。dim=0で0次元要素、dim=1で1次元要素がそれぞれ一致している必要がある。

```
tensor1 = torch.tensor([[1,1,1],[1,1,1],[1,1,1]])
tensor2 = torch.tensor([[2,2],[2,2],[2,2]])

>>>print("1size{}and2size{}".format(tensor1.size(), tensor2.size()))
#1sizetorch.Size([3, 3])and2sizetorch.Size([3, 2])
```

dimの考え方として、Tensor.Sizeの右側から数えていくとわかりやすい。dim=0だと3と2で一致しないのでエラーになる。dim=1で3と3になり一致するので連結できる。

```
>>>torch.cat((tensor1, tensor2), dim=0)
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 3 and 2 in dimension 1 at /pytorch/aten/src/TH/generic/THTensor.cpp:711

>>>torch.cat((tensor1, tensor2), dim=1)
tensor([[1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2]])
```
