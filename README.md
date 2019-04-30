BERT进行一些NLP实验，暂实现词性标注
=====================

### Docker相关
   1.   build

       docker build -t pos-bert . && docker image prune -f

   2.   run(新建)

       docker run --shm-size=300m -p 50001:50001 --memory=500m --restart=always pos-bert


-------------------------------------------
参考：<https://github.com/sberbank-ai/ner-bert>