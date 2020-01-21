这是 ICACI2020 已接收论文《InvUnet:Inverse the Unet for Nuclear Segmentation in
H&E Stained Images》的核心代码, 包括：

`InvUnet.py`: 对 **InvUnet** 的实现；

`weight_map.py`: 对 **权重图** 的实现.

----

There are the key codes of paper "InvUnet:Inverse the Unet for Nuclear Segmentation in
H&E Stained Images" accepted by ICACI2020, including:

`InvUnet.py`: implementation of **InvUnet**；

`weight_map.py`: implmentation of **weight map** .


---

TABLE I. Pixel-level segmentation results(IoU:%) for Unet and InvUnet

| Models  | Params | breast | kidney | liver  | prostate | bladder | colon  | stomach | A\_avg | B\_avg |
|---------|--------|--------|--------|--------|----------|---------|--------|---------|--------|--------|
| Unet    | 7\.76M | 71\.58 | 68\.95 | 67\.85 | 69\.98   | 72\.09  | 65\.63 | 74\.35  | 69\.59 | 70\.69 |
| InvUnet | 1\.26M | 72\.59 | 68\.46 | 68\.07 | 71\.52   | 73\.40  | 66\.09 | 75\.06  | 70\.16 | 71\.52 |

---

TABLE II. Instance-level segmentation results(F1:%) for Unet-3c, InvUnet-3c, Unet-w and InvUnet-w. 

| Models      | Params | breast | kidney | liver  | prostate | bladder | colon  | stomach | A\_avg | B\_avg |
|-------------|--------|--------|--------|--------|----------|---------|--------|---------|--------|--------|
| Unet\-3c    | 7\.76M | 46\.07 | 26\.37 | 56\.73 | 64\.50   | 46\.08  | 26\.37 | 56\.73  | 48\.42 | 43\.06 |
| InvUnet\-3c | 1\.26M | 41\.54 | 20\.53 | 44\.84 | 61\.96   | 41\.54  | 20\.03 | 44\.84  | 42\.22 | 35\.47 |
| Unet\-w     | 7\.76M | 41\.86 | 26\.54 | 61\.30 | 64\.57   | 45\.34  | 53\.63 | 71\.94  | 48\.57 | 56\.97 |
| InvUnet\-w  | 1\.26M | 47\.34 | 34\.73 | 64\.48 | 69\.85   | 52\.19  | 59\.65 | 75\.19  | 54\.10 | 62\.31 |
