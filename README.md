# Sensitive classification 
  目標為比較加入半監督式學習的方法是否能讓模型預測力提升

# Overview
實際情況有標籤的資料非常少，而為了解決標籤的資料過少問題，採用半監督式的方法進行訓練(SSL)，並將一般遷移學習訓練與採用半監督式訓練方法進行結果比較，其中半監督式方法是利用
https://arxiv.org/pdf/1904.12848.pdf 這篇論文進行 Loss function 的改寫，模型部份是利用Roberta模型進行遷移學習訓練
* 加入半監督式學習
    
    將原本的訓練資料翻譯成英文再翻譯回中文進行 data augmentation，然後將翻譯前資料的loss 與翻譯後資料的loss 計算 KL Divergence loss，
    然後再將KL Divergence loss 與原本資料的 Cross Entropy Loss 進行加總，如下：
![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/%E6%88%AA%E5%9C%96%202021-09-05%20%E4%B8%8B%E5%8D%886.33.09.png)

    訓練架構如下：
![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/structure.jpeg)

#Data
利用酒店評論資料進行訓練，讓模型評斷留言是正面還是負面

數據來源 : https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb

正面評論與負面評論資料筆數的比率約為 7:3
# Result
    在測試資料上，F1 Score有上升3個百分點，在抓取負面評論(少樣本)有提升效果
   
   * Test Data on F1 Score
     * General Train 
       
       F1 Score(neg) : 0.84
       
       ![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/roberta.png)
     * Train with SSL
        
       F1 Score(neg) : 0.87
       
       ![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/roberta_with_back_translate.png)