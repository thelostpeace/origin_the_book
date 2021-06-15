## Talking

回顾知识点的时候，可能心里想明白了，但是表达出来就和想的有一些差异性。所以，事先做做剧本。

1. 怎么做关键词挖掘？怎么做summary？怎么以文搜文？

关键词挖掘，传统的办法是tf-idf做词的ranking，对于多文档，词表大小为V，文档数为N，用词的tf-idf值去填充NxV的矩阵，然后做SVD降维，然后取前
K个特征值和特征向量的乘积做关键词中心点的表示，相当于K个聚类。然后分别选取一个离这K个聚类最近的词做关键词。

第二种办法是TextRank，TextRank借鉴的是PageRank的思路，最先用作Text Summary，对于一个Document来说有V个句子，每个句子彼此相连得到一个VxV的完全图。
对每一个句子用word2vec或者其他word embedding的方式以求和取均值的方式得到句子的embedding表示，或者用Language Model以及其他模型直接对句子做encoding。
然后计算每个句对的相似度即为边的分值，整个图其实是一个Markov Chain，最终整体状态会趋向于稳定。一般几十次迭代之后，就差不多稳定了，每个句子的权重初始为1，
稳定后，即可得到所有句子的权重。对于关键词的挖掘也是一样，只不过关键词需要通过pos tag或者ner或者其他方式过滤一些不关注的词，取决于任务的目的。

第三种办法也是通过模型对句子或者词做embedding，然后做聚类，例如k-means，然后分别选取离中心点最近的词做关键词。

第四种办法是supervise的模型，现实中不太可取，训练数据难标注。

2. 说一说CRF

早些时候CRF其实是严重依赖于特征工程的，有一堆feature function，对于文本标注来说，例如当前词是A的时候特征值是1，否则0，就是unigram，
这样可以构造出更复杂的特征，例如bigram，trigram，也可以把上文的label当作特征。这样就是一个NxF的矩阵，加一个linear层，输入纬度是特征数量，
输出纬度是label数量，然后加softmax就可以得到每个label的置信度。现在机遇deep learning，不需要手工构造特征，词的encoding，句子的encoding，
label的encoding都是特征。

3. Loss Functions

`y`是观测值，`ÿ`是预测值。

 - **MSE**: 用于linear regression，`L = sum(y - ÿ)^2`
 - **MLE**: 用于logistic regression，二分类loss `L = -ylog(ÿ) - (1-y)log(1-ÿ)`
 - **Cross Entropy**: 用于多分类，`L = -Sum(y * log(ÿ))`，和**Negative Log Likelihood**是一个东西
 - **Negative Log Likelihood**: 用于多分类
 
<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/NLL_1.png?raw=true" height=100/>
</p>

<p align='center'>
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/NLL_2.png?raw=true" height=150/>
</p>

