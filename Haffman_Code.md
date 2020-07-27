## Huffman Code

### huffman code

Huffman Code是一种对数据分布做最小长度编码的一种方式，其结果接近于熵值。

Huffman Code把最小概率的D个合并成一个形成父节点，以此类推，不够以Dummy去补充。其中`D`为编码数，`p(Dummy) = 0.0`。

例如，我们将`0.25, 0.25, 0.2, 0.15, 0.15`，分别对应`A，B，C，D，E`：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/haffman_code.png?raw=true" width=480px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/haffman_code_ex1.png?raw=true" width=570px/>
</p>

huffman code常被用来做数据压缩，对于高频的数据，用较短的编码，对于低频数据，用较长的编码。对于Huffman Code的相关论证如下：

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/haffman_code_lemma1.png?raw=true" width=570px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/haffman_code_lemma2.png?raw=true" width=570px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/haffman_code_lemma3.png?raw=true" width=570px/>
</p>

<p align="center">
<img src="https://github.com/thelostpeace/origin_the_book/blob/master/image/haffman_code_lemma4.png?raw=true" width=570px/>
</p>

Huffman Code其实是一种贪心算法，通过不断合并最小概率的D个值来获取局部最优解，进而得到全局最优解。


