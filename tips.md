## Tips

### Linear Regression

Linear Regression的输出是连续的，适合做连续的值的结果预测，如果用在分类任务上则需要一个阈值，但从Linear Regression的Cost函数来看，Linear Regression适合做连续值预测。

**formula**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/linear_regression_formula.png?raw=true" height=45/>
</p>

**cost**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/linear_regression_cost.png?raw=true" height=50/>
</p>

### Logistic Regression

Logistic Regression输出值在`[0, 1]`之间，适合做分类任务的置信度预测。

**formula**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_formula.png?raw=true" height=90/>
</p>

**cost**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_cost.png?raw=true" height=90/>
</p>

**cost explain**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_cost1.png?raw=true" height=540/>

<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_cost2.png?raw=true" height=330/>
</p>

**cost simplify**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_cost_simple.png?raw=true" height=120/>
</p>

**maximum likelihood estimation explain**
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_cost_explain.png?raw=true" height=540/>
</p>

**my understanding**

这里是一个很不错的关于`MLE`的解释，[A Gentle Introduction to Maximum Likelihood Estimation for Machine Learning](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/)。

对于每一个样本来说，他的概率为：

<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/mle_sample_prob.png?raw=true" height=45/>
</p>

即当`y=1`时，`p(y=1|x;h) = f(x)`，当`y=0`时，`f(x)`尽可能趋向于0，所以`p(y=0|x;h) = 1 - f(x)`。

MLE假定所有的样本都是彼此独立的，所以对于整个分布来说，整体概率为：

<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/logistic_regression_cost_mle.png?raw=true" height=150/>
</p>

#### reference

1. [Logistic Regression — Detailed Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)