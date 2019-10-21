# DRL-tensorflow
My DRL library with tensorflow1.14
core codes based on https://github.com/openai/spinningup

My job is wrap the algorithms functions into classes in order to easy to call.
Maintain the performance in gym environments of the original codes.

越来越丰富了，基本上将主流的深度强化学习的off-policy的三个主要算法都打包成功了。
**目前都是最简单的模式，直接进入algo_class.py的文件中，run就完事儿了。**
对于结果的显示，以及性能的对比，目前做的还不够，因为我还没吃透spinning-up的log类，没有办法更方便的将这个功能嵌入进去。
还有画图的功能，目前只能用乞丐版的matplotlib画个图。

等我有时间了再加点功能~
大家要是用起来有什么bug，欢迎开issues~
要是有帮助的话，希望能给个star。

过段时间看看能不能加个LSTM的，我已经看到有大佬的实现了，整合一下到我这个包里~