## 起源：Entropy

起源系列意在记录一些关于`Computer Science`并且有实际应用的有意思的东西，所收集的东西很零散，但是每一篇都是能够自我阐明的。

### Entropy

对于`Entropy`的定义，是对一个随机变量的不确定性的一个量化表示。用`x`表示一个在域`X`里离散分布的随机变量，随机分布函数定义为`p(x) = Pr{ x = m}, m ∈ X`。则`Entropy`为：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/entropy.png?raw=true" width=250px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/entropy_ex1.png?raw=true" width=540px/>
</p>

### Joined Entropy and Conditional Entropy

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/joined_entropy.png?raw=true" width=540px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/conditional_entropy.png?raw=true" width=540px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/entropy_chain_rule.png?raw=true" width=540px/>
</p>

### Relative Entropy

`relative entropy`是对两个随机分布的距离的一个量化表示，有很多种不同的命名，例如`Kullback–Leibler distance`、`cross entropy`、`information divergence`、`information for discrimination`等等。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/relative_entropy.png?raw=true" width=540px/>
</p>

对于`CrossEntropyLoss`计算如下：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/cross_entropy_loss.png?raw=true" width=340px/>
</p>

`CrossEntropyLoss`在Machine Learning或者Deep Learning的分类问题中经常用到，其目的就是尽量减少模型拟合的数据分布`q(x)`和数据的真实分布`p(x)`之间的距离，对于单条数据来说，真实分布`p(x=right class) = 1, p(x = wrong class) = 0`，所以`D(p || q) = Sum(log(1/q(x))) = -Sum(log(q(x)))`。其中`q(x)`用softmax计算，意在放大`q(x=right class)`的值，甚至加入较低temperature来使该效果更佳显著。

### Mutual Information

对于`Mutual Information`的定义是：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/mutual_information.png?raw=true" width=540px/>
</p>

参考`Relative Entropy`的定义，不难理解`Mutual Information`的含义，即对于`x,y`的真实分布`p(x,y)`和假定的`x,y`的独立分布`p(x)p(y)`之间的距离。或者对于`x`，给定`y`，对于`x`的`Entropy`的减少的作用性有多大，或者对于`y`，给定`x`，对于`y`的`Entropy`减少的作用性有多大。

`Mutual Information`在Machine Learning里被用来做特征筛选，例如对于早期的文本分类，筛选分类的前N个`unigram`、`bigram`特征。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/mi_feature_selection1.png?raw=true" width=640px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/mi_feature_selection2.png?raw=true" width=640px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/mi_feature_selection3.png?raw=true" width=640px/>
</p>