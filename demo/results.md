# MOT Results:

#### Hockey 5 mins (30 secs). Yolo3 using the yolov3_deepsort.py implementation.

```
            num_frames      mota        motp
acc         748             0.636252    227.407214
      IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM  MOTA    MOTP
full 45.1% 51.8% 39.9% 70.8% 91.8% 25 12 11  2 1179 5451 163  376 63.6% 227.407
part 83.1% 94.9% 74.0% 74.0% 94.9% 25 18  1  6    2   13   0    0 70.0% 175.162
```


#### Hockey 5 mins (all data). Yolo3 using the yolov3_deepsort.py implementation.

```
            num_frames      mota        motp
acc         7497            0.641164    301.817385
      IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML    FP    FN  IDs    FM  MOTA    MOTP
full 16.9% 19.3% 15.0% 71.3% 91.8% 28 12 13  3 11877 53179 1530  3633 64.1% 301.817
part 83.1% 94.9% 74.0% 74.0% 94.9% 25 18  1  6     2    13    0     0 70.0% 175.162
```


#### Netball 5 mins (30 secs). Yolo3 using the yolov3_deepsort.py implementation.

```
            num_frames      mota        motp
acc         748             0.180398    715.354839
      IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM  MOTA    MOTP
full 20.2% 24.3% 17.2% 45.1% 63.6% 16  1 14  1 3085 6574 150  568 18.0% 715.355
part 48.1% 59.1% 40.6% 40.6% 59.1% 16  6  1  9    9   19   0    0 12.5% 534.462
```


#### Netball 5 mins (all data). Yolo3 using the yolov3_deepsort.py implementation.

```
            num_frames      mota        motp
acc         7497            0.172844    715.089561
      IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML    FP    FN  IDs    FM  MOTA    MOTP
full  6.4%  7.4%  5.6% 46.9% 62.2% 16  0 16  0 34119 63722 1378  5579 17.3% 715.090
part 48.1% 59.1% 40.6% 40.6% 59.1% 16  6  1  9     9    19    0     0 12.5% 534.462
```