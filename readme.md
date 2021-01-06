文件命名示例：
sac_sp.py：凡是带下划线sp的文件，都是spinup中的封装形式，即把强化算法封装成函数；
sac_class.py 凡是带class的文件，都是封装成类的形式，便于大家直接调用；
sac_auto_per_class 凡是带per的文件，都是可以选择是否调用优先经验回放的class，但是有时候优先经验回放并不一定好使。
另外关于sac_auto，也被称作sac2，或者自适应sac，因为alpha超参数是放到网络中进行学习，一般来说要比sac更容易调用。

--

2020-12-09

发现这是我获得star最多的一个项目了。

刚才过了一遍所有的文件，发现优先经验回放PER没有单独抽取出来，还是和DDPG打包在一起的，这样会导致不能和TD3，SAC兼容。

另外sac-auto也没有提交。

事后经验回放HER没有实现（我到现在还没有调出最好的参数，贼烦，我本以为her是灵丹妙药，没想到不过如此）。

另外继续搞tf1感觉有种49年入国军的错觉。

难顶

--



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

----
已经更新了logger和plot功能，功能实现代码在sp_utils文件夹中，直接抽调了spinup的代码，做了稍许修改。
在run_in_gym这个文件夹中可以直接试用该功能，非常方便。
spinup的这两个功能可以抽调到大家自己开发的包当中，比自己实现要省事儿很多。


另外，个人感觉我封装的这三个算法，好像不是特别的完美，在gym中测试好像没有问题，但是在机器人环境中无法收敛。
要是有人测试出bug的话，恳请告知~


----




大家要是用起来有什么bug，欢迎开issues~
要是有帮助的话，希望能给个star。

过段时间看看能不能加个LSTM的，我已经看到有大佬的实现了，整合一下到我这个包里~
