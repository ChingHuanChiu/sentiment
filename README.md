# Sensitive classification 

# Overview
為了解決標籤的資料過少問題，因此採用半監督式的方法進行訓練(SSL)，並將一般遷移學習訓練與融合半監督式訓練方法進行結果比較，其中半監督式方法是利用
https://arxiv.org/pdf/1904.12848.pdf 這篇論文進行 Loss function 的改寫，模型部份是利用Roberta模型進行遷移學習訓練
* 加入半監督式學習
    
    將原本的訓練資料翻譯成英文再翻譯回中文進行 data augmentation，然後將翻譯前資料的loss 與翻譯後資料的loss 計算 KL Divergence loss，
    然後再將KL Divergence loss 與原本資料的 Cross Entropy Loss 進行加總，如下：
![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/%E6%88%AA%E5%9C%96%202021-09-05%20%E4%B8%8B%E5%8D%886.33.09.png)

    訓練架構如下：
![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/%E5%9C%96%E7%89%87%201.png)

# Result
    在測試資料上，F1 Score有上升1個百分點
   * Validation data on F1 Score : 
    
     經過半監督式訓練後的結果較佳，其中水藍色為有融入半監督式學習，藍色則無
     
     ![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/%E6%88%AA%E5%9C%96%202021-09-05%20%E4%B8%8B%E5%8D%886.38.43.png)
     
   * Test Data on F1 Score
     * General Train 
       
       F1 Score : 0.85
       
       ![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/%E6%88%AA%E5%9C%96%202021-09-05%20%E4%B8%8B%E5%8D%886.48.41.png)
     * Train with SSL
        
       F1 Score : 0.86
       
       ![image](https://github.com/ChingHuanChiu/sensitive/blob/master/img/%E6%88%AA%E5%9C%96%202021-09-05%20%E4%B8%8B%E5%8D%886.49.23.png)