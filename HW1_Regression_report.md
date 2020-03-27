# Regression  - 从预测PM2.5说起
HW#1主要任务就是学习如何通过足量的数据进行regression的学习，进行学习的任务是利用气象测算相关数据来测算PM2.5的值。
## 学习目标
- 编写简单的regression的训练模型（数据整理，损失计算，参数更新）
- 编写简单的优化策略：Gradient Descent 和 Adagrad
## 数据详细说明
```./Data/hw1_data```中包含了训练集和测试集。

原始训练数据的大小是：4320 x 24, 4320行中，包含了240天的18个特征值，24列，表示一天中每隔一小时各个特征的测量结果。
我们希望根据每9个小时的所有特征值来估计第10个小时的PM2.5的特征值（前9个小时的特征量中是包含了PM2.5这一特征的）。
所以4320 x 24矩阵将变成12个8 x 480的矩阵，每一个矩阵代表一个月前20天各个小时的18个气象特征。

若取每9列（每9个小时）的18个特征和第10列的PM2.5的值为一组训练数据，我们每个月将会有 471 组训练数据，一共就有 471 x 12 组训练数据。对于每组训练数据来说，
拿来预测PM2.5的特征值一共有18 x 9 + 1个，后面的 +1 代表的是bias。

这样我们就可以整理出所有的训练数据和验证数据。
## 结果分析

结果挺有意思的，为了保证自己做出的预测估计的趋势是正确的，我有参考[这位朋友](https://github.com/dafish-ai/NTU-Machine-learning/tree/master/%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E4%BD%9C%E4%B8%9A/HW1)
main file的运行结果。（这位同学收集和整理李宏毅老师的资料是很全的） 
1. 记录和对比了不同的Optimization下，训练的收敛结果。结果表明标准的Gradient Descent收敛最慢，
如果加上一些正则化来加以控制，收敛的速度会变慢。
![img1](Data/hw1_results/cost_compare.png)
2. Gradient Descent 和 Adagrad 都可以达到相似的损失函数值。
![img2](Data/hw1_results/opt_compare.png)
3. 这我这里对比Gradient Descent加正则化和不加正则化收敛的速度其实差不多，分析认为是前期数据进行了归一化的处理，
所以在数据收敛上，看不太出来两者的差别。但是如果不做数据的归一化处理，可以看出在损失函数加上正则化L2的限制，收敛会稍微平稳些。
![img3](Data/hw1_results/w_norm_gd.png)
![img4](Data/hw1_results/wo_norm_gd.png)
4.Adagrad的数据很难调，不知道是我归一化之后数据的问题还是啥的。如果lr很小的话，无法达到和GD一样的cost function的值，
但如果加大学习率，在更新的前几步，cost会变得很大，然后急速下降。在图一中，GD的learning rate都是1e-6，但Adagrad的lr是5.