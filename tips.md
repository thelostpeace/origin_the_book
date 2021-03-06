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

  <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention.png?raw=true"/>
</p>

 假定encoder是`[seqlen, 300]`, decoder是`[1,300]`
 
 1. **Dot**
   
 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention_dot.png?raw=true" height=60/>
</p>

`score.shape = [1, seqlen]`

2. **General**

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention_general.png?raw=true" height=50/>
</p>

`score.shape = [1, seqlen]`

3. **Concat**

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/luong_attention_concat.png?raw=true" height=50/>
</p>

`W.shape = [1, hidden], score.shape=[1, seqlen]`
 
这种传统的attention机制需要指定query和values，而self-attention不需要，因为self-attention算的是词与词之间的alignment，然后以此计算出query。

#### reference

 1. [Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)

### Bert

#### pretrain

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bert_pretrain_sentence_pair.png?raw=true"/>
</p>

bert pretrain的训练数据，每行一个完整句子，document之间以空行分割。因为bert预训练里有一个子任务是next sentence prediction，所以训练数据长这样。

预处理之后，数据变为：

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bert_pretrain_data.png?raw=true"/>
</p>

#### Bert Model

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bert_model.png?raw=true"/>
</p>

这里可能画的有一些歧义性，多层transformer之间，本层transformer的输入是上一层transformer的输出，即`input->transformer->transformer->transformer->...->transformer->output`。

经过多层transformer之后，输出结果为`[batch size, sequence length, hidden size]`，即对整个句子做了encoding，通过`[CLS]`做上下句相关的分类，通过对应`[MASK]`的词的encoding做对应正确的词的预测。

关于scaling，因为两个矩阵Query和Key相乘的时候肯能会导致数值变得异常大，进而导致softmax的gradient变得非常小，所以加scaling。

#### Bert的优势

Bert本身是一个unsupervised MLM(Masked Language Model)，对句对预测和mask词预测做联合训练，在很多NLP任务上都取得了SOTA。预训练好之后的Bert相当于一个Encoding Layer，在后续的任务上做finetune拟合也很快，但由于庞大的参数量，实际使用中往往需要加缓存才能保证延时上是可用的。

#### reference

 1. [BERT Explained: A Complete Guide with Theory and Tutorial](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)

### CRF(Conditional Random Fields)

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/crf1.png?raw=true"/>
</p>

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/crf2.png?raw=true"/>
</p>

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/crf3.png?raw=true"/>
</p>

对于biLSTM + CRF来说，biLSTM层对label的预测相当于发射矩阵，CRF层需要学习状态转移矩阵。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf1.png?raw=true"/>
</p>

上图红框标明的部分是发射矩阵，每个label到label的路径有对应的权重，便是状态转移矩阵。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf2.png?raw=true"/>
</p>

状态转移矩阵是随机初始化的，会在训练的过程中学习到。

CRF的`LossFunction = P(realpath) / (P1 + P2 + ... + Pn), Pi = e^Si, Si = 路径上的节点score + 路径score`
P(readpath)根据训练数据能直接算得score。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step1.png?raw=true"/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step2.png?raw=true"/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step3_1.png?raw=true"/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step3_2.png?raw=true"/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step3_3.png?raw=true"/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step3_4.png?raw=true"/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bilstm_crf_step3_5.png?raw=true"/>
</p>

当学到状态转移矩阵之后，推导最佳路径，是一个简单当DP问题，也叫Viterbi算法。

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/viterbi_algo.png?raw=true"/>
</p>

`Score[l][w] = max(Score[i=1->n][w-1] + weight_path(i,w-1 => w) + weight_node(l, w)), l means label, w means words`, 最大Score是`max(Score[i=1-n][w])`，最大Score的路径需要记录每一个转移的上一个label信息，即可得到最大路径。

#### reference

 1. [What Are Conditional Random Fields?](https://prateekvjoshi.com/2013/02/23/what-are-conditional-random-fields/)
 2. [Why Do We Need Conditional Random Fields?](https://prateekvjoshi.com/2013/02/23/why-do-we-need-conditional-random-fields/)
 3. [Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
 4. [通俗理解BiLSTM-CRF命名实体识别模型中的CRF层](https://www.cnblogs.com/createMoMo/p/7529885.html)
 5. [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

### TF-IDF

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/tf-idf.png?raw=true"/>
</p>
 
 `tf-idf`是对词的权重的一个衡量，常被用来做停用词挖掘，在information retrieval里面被用来做Rank，即Query和Document的匹配度。
 
## C++

### 函数重载

定义如下两个重载函数：

```
int foo(int a, int b) {
    return 0;
}

int foo(int a, double b) {
    return 0;
}
```

c++编译后的符号表：

```
69:    56: 00000000004005ad    17 FUNC    GLOBAL DEFAULT   13 _Z3fooii
73:    60: 00000000004005be    19 FUNC    GLOBAL DEFAULT   13 _Z3fooid
```

因为c++对函数的命名方式涵盖了namespace，函数名和参数类型，所以同函数名不同参数类型，在elf表里面生成的函数名是不一样的，所以c++支持重载这个概念。

c编译之后的符号表：

```
66:    56: 00000000004004ed    19 FUNC    GLOBAL DEFAULT   13 foo
```

符号表里面只有函数名，所以对于c来说，不支持函数重载。

### C++ Virtual Table

```c++
#include <iostream>

class B
{
public:
  virtual void bar();
  virtual void qux();
};

void B::bar()
{
  std::cout << "This is B's implementation of bar" << std::endl;
}

void B::qux()
{
  std::cout << "This is B's implementation of qux" << std::endl;
}

class C : public B
{
public:
  void bar() override;
};

void C::bar()
{
  std::cout << "This is C's implementation of bar" << std::endl;
}

B* b = new C();
// calling C::bar()
b->bar();

```

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/vtable.png?raw=true"/>
</p>

新建一个实例的时候，vpointer指向该实例对应的vtable，例如`B *b = new C()`，`b->vpointer => vtable of class C`，然后`b->bar()`其实是在C的vtable找函数定义。基于此，有虚函数的类，析构函数也得声名为virtual。

#### reference

 1. [Understandig Virtual Tables in C++](https://pabloariasal.github.io/2017/06/10/understanding-virtual-tables/)

## 开放性问题

### 关键词挖掘，Text Summarization

该问题的定义如下，给定一个document，可能含有title、正文、脚注等等，需要挖掘哪些词是该document的关键词。该场景适用于Query Document匹配、以文搜文等。

 - TF-IDF对词做Ranking，TF-IDF算是最早的对词做weighting的方法
 - TextRank（Unsupervised）

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/text_rank1.png?raw=true"/>
</p>

对于Text Summariztion来说，需要对句子做ranking，句对的分值可以通过cosine或者其他distance函数计算，这样对于整个document来说就是一个VxV的图，其中V是句子总数。参考Markov Model，以任意状态开始，无限步后最终停在某个状态的概率是固定的，所以每个sentence都会有一个最终的score。同理，可以把关键词当作句子，只不过词与词之间构成边只限于词对在指定window size之内。在预处理的时候可以结合POS tag过滤部分词，其他过滤手段取决于任务关注的点。

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/text_rank2.png?raw=true height=45"/>
</p>

 <p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/textrank_steps.png?raw=true"/>
</p>

 - 一些Supervised的模型，这种模型对训练集要求比较高，而训练集是比较难获取的。
 - 一些Unsupervised的模型，通过一些unsupervised的任务对句子做encoding，这一部分模型很多，也很多样化。然后对所有的句子做一个clustering，选取各cluster距离中心点最近的句子当作summary的句子。其实相比TextRank，只是换了一种方式而已。

#### reference

 1. [TextRank: Bringing Order into Texts](https://www.aclweb.org/anthology/W04-3252.pdf)
 2. [Unsupervised Text Summarization Using Sentence Embeddings](https://www.cs.utexas.edu/~aish/ut/NLPProject.pdf)
 3. [https://pkghosh.wordpress.com/2019/06/27/six-unsupervised-extractive-text-summarization-techniques-side-by-side/](https://pkghosh.wordpress.com/2019/06/27/six-unsupervised-extractive-text-summarization-techniques-side-by-side/)
 4. [Text Summarization Techniques: A Brief Survey](https://arxiv.org/pdf/1707.02268.pdf)
 

### How to handle imbalanced data sets for classification

 - **oversampling**: 对于训练数据偏少的数据，随机选取，重复复制(2x, 3x, 5x, 10x etc.)
   + **SMOTE**(Synthetic Minority Over-sampling Technique) algorithm: 

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/smote_1.png?raw=true" width=700/>
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/smote_2.png?raw=true" width=700/>
   </p>

   简述就是，首先确认需要扩充多大的数据量，然后随机在要扩充的样本里找一条训练数据`Si`，找到这个数据的knn`(S1-Sk)`，
   从`1-k`随机选取一个值`m`，计算`Si`和`S1-Sm`之间的向量差`D1-Dm`，然后得到扩充数据为`Si + D1-m * random(0, 1)`，然后再随机找一条训练数据，重复该操作直到达到需要扩充的数据量。

   + **ADASYN**(Adaptive Synthetic sampling):

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/adasyn_1.png?raw=true" width=600/>
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/adasyn_2.png?raw=true" width=600/>
   </p>
   
   相比SMOTE对于每个训练样本随机生成1-k个扩充样本，ADASYN根据knn里的majority class样本的占比去扩充样本，这样生成的样本会更加均衡，对于难分类的样本生成更多的扩充样本。

 - **undersampling**:  对于训练数据比较多的数据，随机删除部分数据
   + **NearMiss**: 计算所有majority class和minority class之间的距离，取n个离minority class最近的majority样本删除。Version1，与minority class之间的平均距离最近的被选取；Version2，与minority class之间平均距离最远的被选取；Version3，对于minority class选取M个近邻，majority class离N近邻平均距离较大的被选取（这里对N的描述不清楚）。
   + **CNN**(Condensed Nearest Neighbor): 1. 从训练数据T随机选择一个点，放入U 2. 从T-U选取第一个近邻在U但是与该近邻不在一个分类的样本放入U 3. 重复2直到U不再增大。该算法太耗时，不可取。
   + **ENN**(Edited Nearest Neighbor): 取knn，k里面做voting选出majority class label，移除不符合majority class label的数据，这个效果上挺不错，相当于清理出明确的分类边界。
   + **RENN**(Repeated ENN): 重复ENN清理数据，直到没有数据可以清理，该方法优于ENN，整体准召提升最明显。
   + **Tomek links**: Tomek link定义为A，B两个点，离A最近的是B，离B最近的是A，如果A、B对应的分类不一样，则可以擦除A、B中属于majority class的,原理上还是清理有overlapping的样本。
   + **Easy Ensemble**: 随机从majority sample里面选取样本，使得majority sample和minority sample的大小一致，然后基于此训练N个分类器，基于这N个分类器的结果训练出一个分类器。(准召优于以上所有，但是模型参数量大的话训练和部署都比较占资源)
   + **BalanceCascade**: 相比Easy Ensemble后一次抽样都移除掉之前抽样中预测正确的样本。(效果优于Easy Ensemble)

 - 当分类较多的时候，loss可以用negative sampling

#### reference

 1. [oversampling & undersampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)
 2. [SMOTE — Synthetic Minority Over-sampling Technique](https://medium.com/erinludertblog/smote-synthetic-minority-over-sampling-technique-caada3df2c0a)
 3. [SMOTE: Synthetic Minority Over-sampling Technique (Paper)](https://arxiv.org/pdf/1106.1813.pdf)
 4. [ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf)
 5. [Survey of resampling techniques for improving classification performance in unbalanced datasets](https://arxiv.org/pdf/1608.06048.pdf)

### The Loss Functions

 - **MSE**(Mean-Squared Loss): 适用于连续值预测

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/loss_mse.png?raw=true" height=50/>
   </p>

 - **binary cross entropy(MLE)**: 适用于二分类

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/loss_binary_cross_entropy.png?raw=true" height=50/>
   </p>

 - **cross entropy**: 适用于多分类

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/loss_cross_entropy.png?raw=true" height=50/>
   </p>

 - **Hinge Loss**: penalize不对的预测和置信度低的正确预测

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/loss_hinge.png?raw=true" height=50/>
   </p>

 - 更多loss functions参见reference[2]

#### reference

 1. [Picking Loss Functions - A comparison between MSE, Cross Entropy, and Hinge Loss](https://rohanvarma.me/Loss-Functions/)
 2. [On Loss Functions for Deep Neural Networks in Classification](https://arxiv.org/pdf/1702.05659.pdf)

### Bias & Variance tradeoff

#### What is Bias

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

#### What is Variance

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/bias_variance.png?raw=true" width=600/>
   </p>

在Supervised Learning里面，underfitting意指high bias，low variance；overfitting意指low bias，high variance。

#### The Tradeoff

Dimensionality reduction and feature selection can decrease variance by simplifying models. Similarly, a larger training set tends to decrease variance. Adding features (predictors) tends to decrease bias, at the expense of introducing additional variance. Learning algorithms typically have some tunable parameters that control bias and variance; for example,

 - linear and Generalized linear models can be regularized to decrease their variance at the cost of increasing their bias
 - In artificial neural networks, the variance increases and the bias decreases as the number of hidden units increase,[3] although this classical assumption has been the subject of recent debate.[8] Like in GLMs, regularization is typically applied
 - In k-nearest neighbor models, a high value of k leads to high bias and low variance.
 - In instance-based learning, regularization can be achieved varying the mixture of prototypes and exemplars
 - In decision trees, the depth of the tree determines the variance. Decision trees are commonly pruned to control variance

One way of resolving the trade-off is to use mixture models and ensemble learning. For example, boosting combines many "weak" (high bias) models in an ensemble that has lower bias than the individual models, while bagging combines "strong" learners in a way that reduces their variance.

Model validation methods such as cross-validation (statistics) can be used to tune models so as to optimize the trade-off.

#### reference

 1. [Understanding the Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
 2. [Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

### Gradient Boosting

Gradient Boosting属于ensemble learning其中之一，就是顺序生成weak learner，每一个weak learner拟合目标是之前的模型的预测残差，即`y(current model) = y(target) - y(previous model)`。

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/gradient_boosting.png?raw=true" width=600/>
   </p>

#### reference
 1. [Gradient Boosting from scratch](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
 2. [How to explain gradient boosting](https://explained.ai/gradient-boosting/index.html)
 3. [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)

### XGBoost

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost.png?raw=true" width=400/>
   </p>
   
   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost1.png?raw=true"/>
   </p>

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost2.png?raw=true"/>
   </p>

   `gi`和`hi`可以用MSE为loss函数推导一下，但是这种形式适用于任意loss函数。而`ft(x)`正是要学习的函数，而`欧姆(ft(x))`是对函数复杂度的一个惩罚，相比gbdt而言，xgboost有不一样的学习方式，而且能有效防止overfitting，gbdt很容易overfitting，boosting机制使然。

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost3.png?raw=true"/>
   </p>

`欧姆(ft)`定义为对区域划分个数和L2 norm惩罚，也就是算法更喜欢简单的区域划分以防止过拟合。

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost4.png?raw=true"/>
   </p>

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost5.png?raw=true"/>
   </p>

   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost5.png?raw=true"/>
   </p>
   
   <p align="center">
   <img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/xgboost5.png?raw=true"/>
   </p>

这篇论文很不错，易读易理解，比Reference里的Blog更清晰易懂。

#### booster

 - gbtree，使用regression tree做weak learner
 - gblinear，使用linear regression做weak learner
 - dart，借用neural network里的dropout思想，在每次迭代里丢弃一定比例的树。

#### reference

 1. [Introduction to Boosted Trees(Original)](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
 2. [Introduction to Boosted Trees(doc)](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
 3. [XGBoost Algorithm: Long May She Reign!](https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d)
 4. [XGBoost: A Scalable Tree Boosting System(Paper)](https://arxiv.org/pdf/1603.02754.pdf)
 5. [Higgs Boson Discovery with Boosted Trees(Paper)](http://proceedings.mlr.press/v42/chen14.pdf)
 6. [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
