# Tips

## Machine Learning

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

### Random Forest

**Random Forest**是Decision Tree的一个Ensemble，通过voting决定最终结果。

#### Bagging
 
**Bagging (Bootstrap Aggregation)** is used when our goal is to reduce the variance of a decision tree. Here idea is to create several subsets of data from training sample chosen randomly with replacement. Now, each collection of subset data is used to train their decision trees. As a result, we end up with an ensemble of different models. Average of all the predictions from different trees are used which is more robust than a single decision tree.

树可以并发生成，训练数据随机选取，特征也是随机选取，以此保证树的多样性。

### Gradient Boosting Decision Tree

#### Boosting
 
**Boosting** is another ensemble technique to create a collection of predictors. In this technique, learners are learned sequentially with early learners fitting simple models to the data and then analyzing data for errors. In other words, we fit consecutive trees (random sample) and at every step, the goal is to solve for net error from the prior tree.

树是顺序生成的，每一棵树都是一个Weak Learner，后来生成的树有更高的权重，前提假设是每棵树不具有相关性，每棵树对于部分数据拟合的很好。

#### reference

 1. [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

### SVM

A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

#### Kernel

The learning of the hyperplane in linear SVM is done by transforming the problem using some linear algebra. This is where the kernel plays role.

 - linear kernel: 当数据是线性分割的时候，linear kernel比较合适，当特征比较多的时候，用linear kernel比较合适，当然可以通过PCA降维之后用其他kernel
 - rbf kernel: 把特征往高维映射，让数据变得线性可分割，适用于非线性可分割数据
 - poly kernel：计算量比较大，一般不使用

 选取哪个Kernel还是通过GridSearch比较好，因为不能事先对数据的分布做假定。

#### reference

 1. [Chapter 2 : SVM (Support Vector Machine) — Theory](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)

### KNN
 
算法比较简单，就是找到最近的K个训练集样本，最终预测结果可以用voting的方式获取，计算量随着数据量线性增长。随着K值的增大，边界会越来越平滑。当K趋向于无穷大的时候，哪个分类样本占比最大，就永远出那个分类。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/knn1.png?raw=true" />

<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/knn2.png?raw=true" />

<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/knn3.png?raw=true" />
</p>

### K-Means

K-Means属于unsupervised learning. 算法也比较简单，就是初始化K个中心点，每次迭代中心点都向对应的位置移动，直到最终稳定下来，得到最小MSE。

#### 终止条件

 - The datapoints assigned to specific cluster remain the same (takes too much time)
 - Centroids remain the same (time consuming)
 - The distance of datapoints from their centroid is minimum (the thresh you’ve set)
 - Fixed number of iterations have reached (insufficient iterations → poor results, choose max iteration wisely)

#### Evaluate Cluster Quality: Silhouette Coefficient

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/silhouette_coefficient.png?raw=true" />
</p>

#### reference

 1. [Silhouette (clustering)](https://en.wikipedia.org/wiki/Silhouette_(clustering))

### 特征工程

对于机器学习来说，最主要的不是模型，而是特征的构造和选取，一般对于特征的选取有以下方式。

 - **Mutual Information**：给定分类以及特征，通过mutual information就可以计算出该分类对应不同特征的的互通信息量。
 
 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/mutual_information.png?raw=true" />
</p>

 - **chi-square**: [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
 - **decision tree**也可以用来做特征选取，因为decision tree不需要对数据做过多的处理，所以也很方便。

## Deep Learning

### Word2Vec

`Word2Vec`输入和输出都是Vocab的大小，隐层大小即是embedding的维数。画是这么画，其实实现并没有这么复杂，因为每次gradient都只更新一个词，参考fasttext里面wordembedding的实现。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/word2vec.png?raw=true" />
</p>

#### Cbow & Skip-Gram

**CBOW**

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/cbow.png?raw=true" />
</p>

**Skip-Gram**

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/skip-gram.png?raw=true" />
</p>

skip gram不需要比较大的稀疏矩阵相乘，所以速度上比较快。

训练目标，maximize log probability：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/skip_gram_logprob.png?raw=true" />
</p>

在整个词表上计算softmax非常耗时，所以有了以下两种方法：`Hierarchical Softmax` 和 `Negtive Sampling`。

**两者的区别**

**Skip-gram**: works well with small amount of the training data, represents well even rare words or phrases.

**CBOW**: several times faster to train than the skip-gram, slightly better accuracy for the frequent words.

#### Hierarchical Softmax

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/hierarchical_softmax.png?raw=true" />
</p>

hierarchical softmax的实现一般使用Huffman Tree，这样频率越高的词，路径越短，进而减少计算的时间复杂度。

#### Negative Sampling

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/negtive_sampling1.png?raw=true" />
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/negtive_sampling2.png?raw=true" />
</p>

#### Subsampling of frequent words

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/subsampling_frequent_words.png?raw=true" />
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/subsampling_frequent_words2.png?raw=true" />
</p>

#### Subword Information

fasttext里面word embedding的实现用到了subword information，即词素的概念，英文词有前缀、词根、后缀，例如`delight + ful`，这种想法也是符合语言学的。通过这种方式对词做表义，能够更好的适用于词表之外的词，即对于`Unknown`的词也可以有一个embedding表示。

#### reference

 1. [Distributed Representations of Words and Phrases
and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### Back Propagation

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/computation_graph.png?raw=true" />
</p>

这里对`Back Propagration`做一个简单的推导。

对于给定`f(x)`，在指定`x`的倒数可以计算为：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/derivative.png?raw=true" />
</p>
 
 所以对于forward pass，一是计算`f(x)`，二是计算`f'(x)`，这样在后续back propagation的时候会使用到。假定最后的loss为`Loss`，则：
 
 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/back_propagation1.png?raw=true" />
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/back_propagation2.png?raw=true" />
</p>

有推导可见，loss会随着偏导数无限增大或者无限减少，分别叫做`vanishing gradient`和`gradient explosion`，这也是deep neural netword难训练的原因。对于`vanishing gradient`，一般用highway connection就能解决，包括LSTM的Cell机制，也可以理解为highway connection。对于后者，一般用`clip gradient`的方式解决，即当gradient达到一个阈值之后，将其置为阈值来避免其爆发。

### TextCNN

TextNN因为其并发性比较好，训练比较快，适用于线上的文本分类任务。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/textcnn.png?raw=true" />
</p>

```python
class TextCNN(nn.Module):
    """
        TextCNN model
    """
    def __init__(self, vocab_size, emb_dim, emb_droprate, seq_len, filter_count, kernel_size, conv_droprate, num_class):
        super().__init__()
        self.vocab_size = vocab_size        # vocab size
        self.emb_dim = emb_dim              # embedding dimension
        self.emb_droprate = emb_droprate    # embedding dropout rate
        self.seq_len = seq_len              # sequence length
        self.filter_count = filter_count    # output feature size
        self.kernel_size = kernel_size      # list of kernel size, means kGram in text, ex. [1, 2, 3, 4, 5 ...]
        self.conv_droprate = conv_droprate  # conventional layer dropout rate
        self.num_class = num_class          # classes
        pass

    def build(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_dropout = nn.Dropout(self.emb_droprate)
        self.conv1 = nn.Conv2d(1, self.filter_count, (self.kernel_size[0], self.emb_dim))
        self.conv2 = nn.Conv2d(1, self.filter_count, (self.kernel_size[1], self.emb_dim))
        self.conv3 = nn.Conv2d(1, self.filter_count, (self.kernel_size[2], self.emb_dim))
        self.pool1 = nn.MaxPool2d((self.seq_len - self.kernel_size[0] + 1, 1))
        self.pool2 = nn.MaxPool2d((self.seq_len - self.kernel_size[1] + 1, 1))
        self.pool3 = nn.MaxPool2d((self.seq_len - self.kernel_size[2] + 1, 1))
        self.conv_dropout = nn.Dropout(self.conv_droprate)
        self.fc = nn.Linear(3 * self.filter_count, self.num_class)
        pass

    def forward(self, input_):
        batch_size = input_.shape[0]

        x = self.embedding(input_)
        x = self.emb_dropout(x)

        x = x.view(batch_size, 1, x.shape[1], x.shape[2])
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(batch_size, 1, -1)           # shape: [batch_size, 1, filter_count * conv_layer_count]
        x = self.conv_dropout(x)

        x = self.fc(x)
        x = x.view(-1, self.num_class)          # shape: [batch_size, num_class]

        return x
```

个人对TextCNN的理解是，可以通过`kernel size`来获取uni-gram, bi-gram, tri-gram甚至更高的n-gram特征，通过不同的filter和pooling来取得更复杂的特征表示，即使有stride，textcnn获取的上下文信息也是比较短的。虽如此，textcnn在文本分类任务上还是取得了比较好的效果。

#### reference

 1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

### LSTM

LSTM因为是时序的，所以在训练和预测上会慢很多，但是能够将长一些的上下文信息做encoding。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/lstm_cell.png?raw=true" />
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/lstm_cell_formula.png?raw=true" />
</p>

LSTM有三个门，分别是遗忘门，忘掉上个Cell状态的部分；输入门，将当前的计算结果的部分存入当前状态；输出门，将当前状态输出并更新隐层状态。

一般在Language Model之外，使用bi-LSTM，即前向encoding和后向encoding。因为LSTM的时序越长，则较长的上文更可能被遗忘掉，这样做可以保留较远的上文信息。

#### reference

 1. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Attention

除self-attention之外，常用的attention机制有以下几种。

 - **Bahdanau Attention**，也叫additive attention。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bahdanau_attention.png?raw=true" />
</p>

1. Producing the Encoder Hidden States - Encoder produces hidden states of each element in the input sequence
2. Calculating Alignment Scores between the previous decoder hidden state and each of the encoder’s hidden states are calculated(Note: The last encoder hidden state can be used as the first hidden state in the decoder)
3. Softmaxing the Alignment Scores - the alignment scores for each encoder hidden state are combined and represented in a single vector and subsequently softmaxed
4. Calculating the Context Vector - the encoder hidden states and their respective alignment scores are multiplied to form the context vector
5. Decoding the Output - the context vector is concatenated with the previous decoder output and fed into the Decoder RNN for that time step along with the previous decoder hidden state to produce a new output
6. The process (steps 2-5) repeats itself for each time step of the decoder until an token is produced or output is past the specified maximum length

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bahdanau_attention_flow.png?raw=true" />
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bahdanau_attention_score.png?raw=true" />
</p>

假定`H(decoder)`和`H(encoder)`都是300维的，即使不是也可以通过linear转换成300维的。则`tanh(decoder + encoder).shape`是`[sequence_length, 300]`，`W(combined).shape`是`[300, 1]`，则最终Score的shape是`[1,13]`，softmax之后与output序列相乘即`[1,300]`的context表示。

 - **Luong Attention**根据attention的计算方式，分为三种类型。

 假定encoder是`[seqlen, 300]`, decoder是`[1,300]`
 
 1. **Dot**
   
 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention_dot.png?raw=true" height=60/>
</p>

`score.shape = [1, seqlen]`

2. **General**

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention_general.png?raw=true" height=60/>
</p>

`score.shape = [1, seqlen]`

3. **Concat**

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention_concat.png?raw=true" height=60/>
</p>

`W.shape = [1, hidden], score.shape=[1, seqlen]`
 
这种传统的attention机制需要指定query和values，而self-attention不需要，因为self-attention算的是词与词之间的alignment，然后以此计算出query。

#### reference

 1. [Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)