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

### Decision Tree

#### C4.5 Algorithm

 - 特征选择Split条件：**Information Gain**，**Gain Ratio**，**Gini Index**

<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/entropy.png?raw=true" height=50 />
</p>

对于每一次split，计算split之后的entropy，与父节点的entropy之差便是**Information Gain**，每次取最大的information gain做split。Information Gain更偏向于数值分布比较广泛的特征。

C4.5 使用**Gain Ratio**做split决策。

<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/gain_ratio.png?raw=true" height=450 />
</p>

**Gini Index**

<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/gini_index.png?raw=true" height=150 />
</p>

**Gini Index**只做二分split。

 - prune

**prune**移除那些Information Gain比较小的分支，进而避免overfitting。

#### Pros

 - Decision trees are easy to interpret and visualize.
 - It can easily capture Non-linear patterns.
 - It requires fewer data preprocessing from the user, for example, there is no need to normalize columns.
 - It can be used for feature engineering such as predicting missing values, suitable for variable selection.
 - The decision tree has no assumptions about distribution because of the non-parametric nature of the algorithm. 

#### Cons

 - Sensitive to noisy data. It can overfit noisy data.
 - The small variation(or variance) in data can result in the different decision tree. This can be reduced by bagging and boosting algorithms.
 - Decision trees are biased with imbalance dataset, so it is recommended that balance out the dataset before creating the decision tree.
 
#### reference

 1. [Decision Tree Classification in Python](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)


