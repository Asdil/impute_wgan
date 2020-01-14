### 基于GAN的Impute模型
 生成器（Generator）: 结合Full Connection（全局信息）和 CNN（局部信息），以及全量数据的位点统计先验
 判别起（Discriminator）: 结合Full Connection 和 CNN 对生成数据和原始数据进行区分
 两者迭代训练，每次训练5个Discriminator后再进行Generator训练，以保证Discriminator模型足够强
 目前数据长度使用的是500bp， 目前的效果(基于位点的统计结果)：

### 运行方式
1. git clone到本地，在目录中创建error和ckpt目录
2. 运行bash train.sh, 传递相关参数（查看args.py)

### 实验效果
>=0.99|(0.15 ~ 0.20)  >=0.96|(0.41 ~ 0.5)   >=0.9|(1.00)

### Notice
1.对于低频位点，在训练时保留，但不进入最后的评估统计中
2.目前模型的瓶颈在于对于某些低频变异位点的impute中
