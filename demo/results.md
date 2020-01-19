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