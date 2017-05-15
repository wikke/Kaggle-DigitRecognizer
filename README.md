# Digit Recognizer

[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)

# SVM

**这里仅仅验证了5000个train dataset的预测结果**

- sharpen数据，经过测试，以32为临界值，可以得到最佳的预测结果

# CNN

**这里仅仅验证了2000个train dataset的预测结果，利用全部的数据预测得到了0.98886的Kaggle竞赛结果**

- 输入数据被划分为4维(dataset size, width, height, image channel)
- 2层卷积网络，包括ReLU Activation和Max Pooling
- 在softmax前Dropout(0.5)
