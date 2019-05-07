BERT/ERNIE进行一些NLP实验，暂实现词性标注
===========================

### Docker相关

#### 1. 词性标注
   1.   build

       docker/pos/build.sh

   2.   run

       docker/pos/run.sh

   3.   save

       docker/pos/save.sh

   4.   load

       docker/pos/load.sh

#### 2. NER
这里为王轩实现，因此只提供了镜像加载和运行。见docker/ner目录

### 部署
分为两部分:
1.  分布式部署pos-bert或ner-bert到集群中。见docker/deploy
2.  haproxy负载均衡部署（暂单节点）。见docker/haproxy






-------------------------------------------
参考：<https://github.com/sberbank-ai/ner-bert>