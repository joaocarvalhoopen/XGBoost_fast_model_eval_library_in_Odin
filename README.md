# XGBoost fast model eval library in Odin
This is a simple library that on a single low GHz CPU only takes 2.45 us micro seconds (now 796 ns nano-seconds) to make the example prediction.

## Description
This is a library to evaluate XGBoost models in odin without dependencies. The example model that I provide generates a model for 13 input variables of Boston Housing Prices and as 100 trees. **It evaluates / predicts in 796 ns nano-seconds** on a AMD Ryzen 4700G 65 Watt's in single thread mode. <br>
**So it's very fast!**<br>
It reads and generates the internal representation of the 228 KByte model from a python text model dump file in 1.8 ms. After that it makes any number of prediction in 2.45 micro-seconds ( now 796 ns), after the first evaluation to make the fill the caches with the model data. This can also be multi-threaded ex: 16 cores, I estimate that it would bringing the mean time of a prediction to something like 40 ns nano seconds.  

## The output of the example program is as follows

First this was 2.45 us but then I increased the number of things that I was making in the hot loop, because the CPU is super scaler and can do 4 things at once, so the total time for a prediction dropped from 2.45 us (micro-seconds) to 796 ns (nano seconds). 


``` bash 
./xgboost_fast_model_eval.exe

xgboost fast model evaluator in Odin begin...

Execution duration xme.load_model_from_txt() : 1.896404ms  

target_predicted_value_correct_1 : 24.019
Execution duration xme.xgb_predict() [1] : 3.67µs  
predicted_value_1 : 24.019

target_predicted_value_correct_2 : 41.766
Execution duration xme.xgb_predict() [2] : 2.45µs 
predicted_value_2 : 41.766

Execution duration xme.xgb_predict_IPC() [2] 4 instruction per cycle,
mean time 80 elements : 63.71µs each element : 796.375 nano seconds 
pred_1 : 41.766
pred_2 : 41.766
pred_3 : 41.766
pred_4 : 41.766

xgboost fast model evaluator in Odin end...
```
## License
MIT Open License

## Have fun
Best regards, <br>
João Nuno Carvalho

