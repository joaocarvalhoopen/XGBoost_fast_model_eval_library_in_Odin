// Project name : XGBoost fast model eval library in Odin
//
// Description  : This is a library to evaluate xgboost models in odin without
//                dependencies .
//                The example model that I provide generates a model for 13 input
//                variables of Boston Housing Prices and as 100 trees.
//                It evaluates / predicts in 2.45 micro-seconds on a AMD Ryzen 4700G
//                65 Watt's.
//                So it's very fast!
//                It reads and generates the internal representation of the 228 KByte
//                model from a Python text model dump file in 1.8 ms.
//                After that it makes any number of prediction in 2.45 micro-seconds,
//                after the first evaluation to make the fill the caches with the model
//                data.
// 
//                The output of the example program is as follows:
//
//                ``` 
//                ./xgboost_fast_model_eval.exe
//                xgboost fast model evaluator in Odin begin...
//                Execution duration xme.load_model_from_txt() : 1.896404ms  
//
//                target_predicted_value_correct_1 : 24.019
//                Execution duration xme.xgb_predict() [1] : 3.67µs  
//                predicted_value_1 : 24.019
//
//                target_predicted_value_correct_2 : 41.766
//                Execution duration xme.xgb_predict() [2] : 2.45µs 
//                predicted_value_2 : 41.766
//
//                xgboost fast model evaluator in Odin end...
//                ```
//
// Author       : João Nuno Carvalho
// License      : MIT Open License
// Date         : 2024.01.01


package main_examples

import xme"./xgboost_fast_model_eval"
import "core:fmt"
import t "core:time"
import "core:slice"

main :: proc () {
    fmt.println("xgboost fast model evaluator in Odin begin...")

    path := "./"
    filename := "model_xgboost.txt"
    
    stopwatch: t.Stopwatch
    t.stopwatch_start( & stopwatch )
    xgb_model : xme.XGB_Model_Eval
    err   : xme.XGB_Error
    xgb_model, err = xme.load_model_from_txt( path, filename )
    if err.type != xme.Error_Type.none {
        fmt.printf( "ERROR: %v\n\n", err)
        fmt.println("load_model_from_txt() failed\n\n")
        return
    }
    defer xme.xgb_model_eval_delete( & xgb_model )
    // time how long it took
    t.stopwatch_stop( & stopwatch )
    duration_1 := t.stopwatch_duration( stopwatch )
    t.stopwatch_reset( & stopwatch )

    fmt.printf("Execution duration xme.load_model_from_txt() : %v  \n", duration_1 )

    // Prints the model.
    // xme.xgb_model_print( xgb.model, xme.Order.desordered )
    // xme.xgb_model_print( xgb_model, xme.Order.ordered )   // <--- This is better for debugging.


    // Original data
    target_predicted_value_correct_1 := 24.019
    fmt.printf( "\ntarget_predicted_value_correct_1 : %v\n", target_predicted_value_correct_1 )

    input_slice_1 : []f32 = []f32{ 0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98 }
    
    t.stopwatch_start( & stopwatch )

    predicted_value_1 := xme.xgb_predict( & xgb_model, input_slice_1 )
    
    // time how long it took
    t.stopwatch_stop( & stopwatch )
    duration_2 := t.stopwatch_duration( stopwatch )
    t.stopwatch_reset( & stopwatch )

    fmt.printf("Execution duration xme.xgb_predict() [1] : %v  \n", duration_2 )


    fmt.printf( "predicted_value_1 : %v\n\n", predicted_value_1 )


    // Invented data    
    target_predicted_value_correct := 41.766
    fmt.printf( "target_predicted_value_correct_2 : %v\n", target_predicted_value_correct )

    input_slice_2 : []f32 = []f32{ 0.10632,19.00,3.310,0,0.5380,7.5750,66.20,5.0900,1,297.0,16.30,397.90,3.98 }
    
    t.stopwatch_start( & stopwatch )
    
    predicted_value_2 := xme.xgb_predict( & xgb_model, input_slice_2 )

    // time how long it took
    t.stopwatch_stop( & stopwatch )
    duration_3 := t.stopwatch_duration( stopwatch )
    t.stopwatch_reset( & stopwatch )

    fmt.printf("Execution duration xme.xgb_predict() [2] : %v \n", duration_3 )


    fmt.printf( "predicted_value_2 : %v\n\n", predicted_value_2 )



    // ############
    // Make 80 predictions to fill the caches.
    num_elem := 80
    input_dyn_array_3 : [dynamic]f32 = make( [dynamic]f32, len = 0, cap = num_elem * len(input_slice_2) )
    // TODO: Threat the errors and delete the dynamic array.
    for _ in 0 ..< num_elem {
        append_elems( & input_dyn_array_3, ..input_slice_2 )
        // for i in 0 ..< len(input_slice_2) {
        //     append( &input_dyn_array_3, input_slice_2[ i ] )
        // }
    }

    // fmt.printf( "==> input_slice_4 : %v\n", input_dyn_array_3 )

    input_slice_4 := input_dyn_array_3[ : ]

    stride : = len(input_slice_2 )

    t.stopwatch_start( & stopwatch )

    num_pred_at_a_time := 4

    pred_1, pred_2, pred_3, pred_4 : f64

    for i := 0; i < num_elem / num_pred_at_a_time; i += 1 {
        begin := i * stride * num_pred_at_a_time
        end   := begin + stride * num_pred_at_a_time
        pred_1, pred_2, pred_3, pred_4 = xme.xgb_predict_IPC( & xgb_model,
                    input_slice_4[ begin : end ],
                    stride )
//        if pred_1 != pred_2 || pred_1 != pred_3 || pred_1 != pred_4 {
//            fmt.println( "ERROR: IPC predictions are not equal." )
//        }
        
        // debug
        // fmt.printf( "begin : %v\n", begin )
        // fmt.printf( "end   : %v\n", end )
        // fmt.printf( "input_slice_4 : %v\n", input_slice_4[ begin : end ] )
    }
    // time how long it took
    t.stopwatch_stop( & stopwatch )
    duration_4 := t.stopwatch_duration( stopwatch )
    t.stopwatch_reset( & stopwatch )

    fmt.printf("Execution duration xme.xgb_predict_IPC() [2] 4 instruction per cycle, mean time 80 elements : %v each element : %v nano seconds \n", duration_4, f32(duration_4) / f32(num_elem) )

    fmt.printf( "pred_1 : %v\n", pred_1 )
    fmt.printf( "pred_2 : %v\n", pred_2 )
    fmt.printf( "pred_3 : %v\n", pred_3 )
    fmt.printf( "pred_4 : %v\n", pred_4 )





     // ############
    // Make 80 predictions to fill the caches.
    // num_elem := 80
    // input_dyn_array_3 : [dynamic]f32 = make( [dynamic]f32, len = 0, cap = num_elem * len(input_slice_2) )
    // // TODO: Threat the errors and delete the dynamic array.
    // for _ in 0 ..< num_elem {
    //     append_elems( & input_dyn_array_3, ..input_slice_2 )
    //     // for i in 0 ..< len(input_slice_2) {
    //     //     append( &input_dyn_array_3, input_slice_2[ i ] )
    //     // }
    // }

    // fmt.printf( "==> input_slice_4 : %v\n", input_dyn_array_3 )

    // input_slice_4 := input_dyn_array_3[ : ]

    // stride : = len(input_slice_2 )

    t.stopwatch_start( & stopwatch )

    num_pred_at_a_time_2 := 8

    pred_b_1, pred_b_2, pred_b_3, pred_b_4 : f64
    pred_b_5, pred_b_6, pred_b_7, pred_b_8 : f64

    for i := 0; i < num_elem / num_pred_at_a_time_2; i += 1 {
        begin := i * stride * num_pred_at_a_time_2
        end   := begin + stride * num_pred_at_a_time
        pred_b_1, pred_b_2, pred_b_3, pred_b_4,
        pred_b_5, pred_b_6, pred_b_7, pred_b_8 = xme.xgb_predict_IPC_2( & xgb_model,
                    input_slice_4[ begin : end ],
                    stride )
//        if pred_1 != pred_2 || pred_1 != pred_3 || pred_1 != pred_4 {
//            fmt.println( "ERROR: IPC predictions are not equal." )
//        }
        
        // debug
        // fmt.printf( "begin : %v\n", begin )
        // fmt.printf( "end   : %v\n", end )
        // fmt.printf( "input_slice_4 : %v\n", input_slice_4[ begin : end ] )
    }
    // time how long it took
    t.stopwatch_stop( & stopwatch )
    duration_5 := t.stopwatch_duration( stopwatch )
    t.stopwatch_reset( & stopwatch )

    fmt.printf("Execution duration xme.xgb_predict_IPC_2() 8 instructions per cycle [3] mean time 80 elements : %v each element : %v nano seconds \n", duration_5, f32(duration_5) / f32(num_elem) )

    fmt.printf( "pred_1 : %v\n", pred_b_1 )
    fmt.printf( "pred_2 : %v\n", pred_b_2 )
    fmt.printf( "pred_3 : %v\n", pred_b_3 )
    fmt.printf( "pred_4 : %v\n", pred_b_4 )
    fmt.printf( "pred_1 : %v\n", pred_b_5 )
    fmt.printf( "pred_2 : %v\n", pred_b_6 )
    fmt.printf( "pred_3 : %v\n", pred_b_7 )
    fmt.printf( "pred_4 : %v\n", pred_b_8 )



    fmt.println("xgboost fast model evaluator in Odin end...")
}

