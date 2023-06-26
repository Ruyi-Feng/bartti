# bartti

This model based on BART which using Transformer structure and with a detaily designed training task.
Our bartti is one of the derives of BART model that aiming applied in Traffic Prediction.

# Paper Profile
Vehicle trajectory prediction preforms an essential basement of several related work. Such as trajectory compensation, traffic condition judgement, future path generation as well as vehciles simulation. Actually, the well-learned driving strategy is one of the keys to realize these tasks. This paper aims to build a pre-training task, based on the Transformer structure, to learn the performance of vehicles in different traffic flow states, and then connect to the above-mentioned downstream tasks. Transformer is good at learning the interaction between vehicles and predicting overall features.  Humm.. It would be even better if it could incorporate driving information.

## Following work
- func trans_to_array [done]
  - 删掉帧号
  - 增加帧间隔符
  - 把没有连续出现的车删去，只保留连续五帧都在的车
- 挖空
  - _noise_one 选r%的车挖掉一帧（1/2 改ID） [done]
  - _noise_two 选r%的车mask(替换掉) [done]
  - _noise_three直接删掉一整帧 （1/2 改ID） [done]
  - _noise_four交换两帧位置 [done]
- 有embed要改一下 [done]
- 大数据集制作，需要注意归一化一下车辆出现的位置 [done]
- main [done]
- Problem fix
  - 训练集长度超过最大时的截取。
  - 针对于local环境的滑窗，不用大范围预测
  - 训练资源


