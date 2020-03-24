# fo-sent

## 安装

pip install -r requirements.txt

## 数据
从https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip下载，解压

## 训练
需要修改bertfo.py第5行的路径为前面解压的路径。
```
python bertfo.py
```

## 预测服务
```
export FLASK_APP=pred.py
flask run
```
