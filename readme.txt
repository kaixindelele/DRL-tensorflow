1、先运行data_utils中的create_train_dataset.py
这里需要设置的是
    1、os.environ["CUDA_VISIBLE_DEVICES"] = "3"  ----> 0
    2、MAX_EPISODES = 2000 ---->1600比较合适
    3、Origin_size换成256最好。或者之后要对应起来。
    4、然后run就行了。
    5、出现暂停的时候，在console那儿，输入c,然后输入，就可以继续运行了
Git is a version control system.
Git is free software.