# Classification - 如何进行二元分类

第二次作业主要是focus在二分类问题，如何利用已有的数据对人的收入在50000以上还是50000以下做判断。

## 数据分析

训练输入数据是 X_train， 输出数据是 Y_train，测试输入数据是X_test

输入数据的每个人都有自己的id，每一行都是一个1010的vector，1代表是，0代表否，第一行标注了每一个是否项的具体内容，输出数据Y_train是每一个id的收入是否在50000以上，也是用01表示的。

## 方法分析

这里的方法分为两类，总的来说就是generative的方法和discriminative的方法。

生成（generative）模型的方法就是用已知的数据来构建生成不同类别的模型，我们在有了模型之后可以随便生成某一类别的数据，所以用已知数据来估计模型的方法叫生成方法。

另一种判别(discriminative)模型是利用已知的数据来拟合分界条件，也就是说我们不去猜测数据怎么来的，我们希望对数据进行参数的建模，在最小化误差的同时确定参数值，并以此作为下次判定分类的边界条件。

- Generative Model

  一般情况下，我们对数据进行高斯模型的拟合，然后计算不同分类的均值 $\mu$ 和协方差矩阵 $\Sigma$。然后根据公式可以得到分类的参数值

- Discriminative Model

  一般情况下，我们对数据直接logistic regression的拟合，然后估计参数值。

## 结果分析

- Method1做的是判别模型，可以看到训练和validate数据的准确率图。用的是最简单的gradient descent的方法，lr= 0.0002， batchsize=8

  ![Figure_1](../../Blog/source/images/Figure_1.png)

- Method2做的是生成模型，直接用每个模型本身的均值和方差来估计模型的参数，然后再放入logistic regression中进行判别。

  ![Figure_2](../../Blog/source/images/Figure_2.jpeg)



