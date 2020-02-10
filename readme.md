# 基于Resnet构建的分类器
#### Power by Ruis 2020.
## 运行环境
Python >3.6 且可以运行Pytorch Cuda版本的环境,如果仅测试模型,Pytorch Cpu版本也可运行。
## 目录结构
- dataset 存放数据集
- models 存放训练好的模型
- resnet_classifier 源代码
    - config 配置文件
    - utils 通用工具
## 使用方法
1. 安装requirements.txt 所示软件包
    ```shell
    pip install -r requirements.txt
    ```
2. 在dataset建立如下目录结构
    - dataset
        - train
            - dog
            - cat
        - val
            - dog
            - cat
3. 酌情修改 config/config.yaml中内容
4. 在scrpits同级目录运行
    ```shell
    sh scripts/train.sh
    ```
5. 训练完成
6. 修改scripts/test.yaml中模型路径,待处理图像路径
7. 使用scripts/test.py 测试模型
    ```shell
    sh scripts/test.sh
    ```