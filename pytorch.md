# Pytorch 実装

## 概要
Pytorchにおける自動識別用のコアパッケージ。順方向フェーズで実行する操作を記憶し、逆方向フェーズで操作を再生する。
Torchクラスの引数requieres_grad=Trueと指定することで追跡が可能になる。

### 実践

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
torch.maxでは、2要素の返り値があるが、indicesプロパティのほうには計算履歴が残らない。

```
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

>>>torch.max(a, 0)
#torch.return_types.max(values=tensor(3., grad_fn=<MaxBackward0>), indices=tensor(2))
```
