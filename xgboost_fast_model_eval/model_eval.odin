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


package xgboost_fast_model_eval

import "core:fmt"
import "core:strings"
import "core:math"
import "core:math/bits"
import "core:os"
import "core:mem"
import "core:strconv"
import "core:slice"
import "core:intrinsics"


NUMBERS_STR : string : "-.0123456789"

XGB_Model_Eval :: struct {
    // num_trees : int,
    vec_trees : [dynamic]XGB_Tree,
}

XGB_TREE_NODE_LEAF :: bits.U16_MAX 

XGB_Tree :: struct {
    tree_num  : int,
    // num_nodes : int,
    vec_in_pid_n_or_leaf     : [dynamic]u16,   // = EMPTY;
    vec_in_cmp_or_leaf_value : [dynamic]f32,   // = 0.0;
    vec_jump_cmp_lt          : [dynamic]u16,   // = 0;
    vec_jump_cmp_ge          : [dynamic]u16,   // = 0;
}

// XGB_Tree :: struct {
//     tree_num  : int,
//     num_nodes : int,
//     nodes_vec : [dynamic]XGB_Tree_Node,
// }

// XGB_Tree_Node :: struct {
//     in_pid_n_or_leaf     : u16,   // = EMPTY;
//     in_cmp_or_leaf_value : f32,   // = 0.0;
//     jump_cmp_lt          : u16,   // = 0;
//     jump_cmp_ge          : u16,   // = 0;
// }

// In_Tree_Node_N_or_LEAF :: enum u8 {
//     empty     = 0,
//     node_id_n = 1,
//     leaf      = 2,
// }

XGB_Error :: struct {
    type : Error_Type,
    msg  : string,
}

Error_Type :: enum u8 {
    none       = 0,
    allocation = 1,
    io         = 2,
    parse      = 3,
}

xgb_model_eval_create :: proc ( trees_num_extimate : int = 100 ) -> ( model : XGB_Model_Eval, err : XGB_Error ) {
    vec_trees := make( [dynamic]XGB_Tree, 0, trees_num_extimate )
    if vec_trees == nil {
        return XGB_Model_Eval{},  // empty struct
               XGB_Error{
                    type = Error_Type.allocation,
                    msg  = fmt.aprintf( "xgb_model_eval_create() : make() failed" )
               }
    }   

    model = XGB_Model_Eval {
        // num_trees = 0,
        vec_trees = vec_trees,
    }

    return model, XGB_Error{ type = Error_Type.none, msg = "" }
}

// @private
xgb_model_tree_create :: proc ( tree_size_extimate : int = 1000 ) -> ( tree : XGB_Tree, err : XGB_Error ) {

    vec_in_pid_n_or_leaf     := make( [dynamic]u16, 0, tree_size_extimate )
    vec_in_cmp_or_leaf_value := make( [dynamic]f32, 0, tree_size_extimate )
    vec_jump_cmp_lt          := make( [dynamic]u16, 0, tree_size_extimate )
    vec_jump_cmp_ge          := make( [dynamic]u16, 0, tree_size_extimate )

    if vec_in_pid_n_or_leaf == nil     ||
       vec_in_cmp_or_leaf_value == nil ||
       vec_jump_cmp_lt == nil          ||
       vec_jump_cmp_ge == nil {
        return XGB_Tree{ }, // empty struct
               XGB_Error{
                    type = Error_Type.allocation,
                    msg  = fmt.aprintf( "xgb_model_tree_create() : one or more make() failed" )
               }
    }

    tree_num  := 0
    res := XGB_Tree{
        tree_num,
        vec_in_pid_n_or_leaf,
        vec_in_cmp_or_leaf_value,
        vec_jump_cmp_lt,
        vec_jump_cmp_ge,
    }
    return res, XGB_Error{ type = Error_Type.none, msg = "" }
} 

xgb_model_tree_clone :: proc ( model_tree : XGB_Tree ) -> ( tree : XGB_Tree, err : XGB_Error ) {

    // Clone the tree. Allocate.
    new_vec_in_pid_n_or_leaf, err_1     := slice.clone_to_dynamic( model_tree.vec_in_pid_n_or_leaf[ : ] )
    new_vec_in_cmp_or_leaf_value, err_2 := slice.clone_to_dynamic( model_tree.vec_in_cmp_or_leaf_value[ : ] )
    new_vec_jump_cmp_lt, err_3          := slice.clone_to_dynamic( model_tree.vec_jump_cmp_lt[ : ] )
    new_vec_jump_cmp_ge, err_4          := slice.clone_to_dynamic( model_tree.vec_jump_cmp_ge[ : ] )

    if err_1 != mem.Allocator_Error.None ||
       err_2 != mem.Allocator_Error.None ||
       err_3 != mem.Allocator_Error.None ||
       err_4 != mem.Allocator_Error.None {
        return XGB_Tree{ }, // empty struct
               XGB_Error{
                    type = Error_Type.allocation,
                    msg  = fmt.aprintf( "xgb_model_tree_clone() : one or more slice.clone_to_dynamic() failed" )
               }
    }

    res := XGB_Tree{
        tree_num                 = model_tree.tree_num,
        vec_in_pid_n_or_leaf     = new_vec_in_pid_n_or_leaf,
        vec_in_cmp_or_leaf_value = new_vec_in_cmp_or_leaf_value,
        vec_jump_cmp_lt          = new_vec_jump_cmp_lt,
        vec_jump_cmp_ge          = new_vec_jump_cmp_ge,
    }
    return res, XGB_Error{ type = Error_Type.none, msg = "" }
}

xgb_model_eval_delete :: proc ( model_model_eval : ^XGB_Model_Eval ) {
    // Delete the trees.
    for _, i in model_model_eval.vec_trees {
        xgb_model_tree_delete( & model_model_eval.vec_trees[ i ] )
    }

    // Clear the model.
    clear( & model_model_eval.vec_trees )

    // Delete the dynamic arrays.
    delete( model_model_eval.vec_trees )

    // Set dynamic arrays ptr to nil.
    model_model_eval.vec_trees = nil
}

xgb_model_tree_delete :: proc ( model_tree : ^XGB_Tree ) {
    // Delete the tree.
    model_tree.tree_num = 0
    
    // This is optional, but it's good to clear the tree before deleting it.
    // It only set's the size of the dynamic arrays to 0.
    clear( & model_tree.vec_in_pid_n_or_leaf)
    clear( & model_tree.vec_in_cmp_or_leaf_value )
    clear( & model_tree.vec_jump_cmp_lt )
    clear( & model_tree.vec_jump_cmp_ge )

    // Delete the dynamic arrays.
    delete( model_tree.vec_in_pid_n_or_leaf )
    delete( model_tree.vec_in_cmp_or_leaf_value )
    delete( model_tree.vec_jump_cmp_lt )
    delete( model_tree.vec_jump_cmp_ge )
    
    // Set dynamic arrays ptr to nil. 
    model_tree.vec_in_pid_n_or_leaf = nil
    model_tree.vec_in_pid_n_or_leaf = nil
    model_tree.vec_jump_cmp_lt      = nil
    model_tree.vec_jump_cmp_ge      = nil
}

xgb_model_tree_clear :: proc ( model_tree : ^XGB_Tree ) {
    // Clear the tree.
    model_tree.tree_num = 0
    // Set the len of the dynamic arrays to 0.
    clear( & model_tree.vec_in_pid_n_or_leaf )
    clear( & model_tree.vec_in_cmp_or_leaf_value )
    clear( & model_tree.vec_jump_cmp_lt )
    clear( & model_tree.vec_jump_cmp_ge )
}

xgb_model_tree_insert_node_normal :: proc ( model_tree   : ^XGB_Tree,
                                            node_index   : int, 
                                            input_num    : int,
                                            compare_num  : f32,
                                            yes_jump_num : int,
                                            no_jump_num  : int ) {

    // Curent len of tree.
    total_num_of_nodes := len( model_tree^.vec_in_pid_n_or_leaf )

    max_node_index_len := max( yes_jump_num, no_jump_num ) + 1
    
    if max_node_index_len > total_num_of_nodes {
        // Grow the tree to the max node index of the jump nodes, that are always
        // greater then the node_index.

        // Note that the buffer is reused so no other allocation is made after
        // achieving the max tree size.
        // for i := total_num_of_nodes; i <= max_node_index; i += 1 {
        for i in total_num_of_nodes ..< max_node_index_len {
            // Append the node to each vec of the tree.
            append( & model_tree^.vec_in_pid_n_or_leaf,     bits.U16_MAX )
            append( & model_tree^.vec_in_cmp_or_leaf_value, 0.0 )
            append( & model_tree^.vec_jump_cmp_lt,          bits.U16_MAX ) // Invalid value that will through out of bound if used.
            append( & model_tree^.vec_jump_cmp_ge,          bits.U16_MAX ) //    "      "
        }
    } 

    assert( node_index < len( model_tree^.vec_in_pid_n_or_leaf ) )

    // inject_at_elem( & model_tree^.vec_in_pid_n_or_leaf, node_index, u16( input_num ) )

    // Set the node to each vec of the tree.
    model_tree.vec_in_pid_n_or_leaf[ node_index ]     = u16( input_num )
    model_tree.vec_in_cmp_or_leaf_value[ node_index ] = compare_num
    model_tree.vec_jump_cmp_lt[ node_index ]          = u16( yes_jump_num )
    model_tree.vec_jump_cmp_ge[ node_index ]          = u16( no_jump_num ) 
}

xgb_model_tree_insert_node_leaf :: proc ( model_tree : ^XGB_Tree,
                                 node_index : int, 
                                 leaf_num   : f32 ) {

    // In the case of leaf nodes we have already grown the tree to the max node
    // index of the jump nodes.
    // So the leaf node index is already assured to exist.

    assert( node_index <= len( model_tree^.vec_in_pid_n_or_leaf ) )

    // Set the node to each vec of the tree.
    model_tree.vec_in_pid_n_or_leaf[ node_index ]     = XGB_TREE_NODE_LEAF
    model_tree.vec_in_cmp_or_leaf_value[ node_index ] = leaf_num
    model_tree.vec_jump_cmp_lt[ node_index ]          = bits.U16_MAX   // Invalid value that will through out of bound if used.
    model_tree.vec_jump_cmp_ge[ node_index ]          = bits.U16_MAX   //    "      "
}

// Assert all SOA - Structure of Arrays vectors inside the tree are of the same len.
xgb_model_tree_assert_soa_len :: proc ( model_tree : ^XGB_Tree ) -> bool {
    len_first := len( model_tree.vec_in_pid_n_or_leaf )
    assert( len_first == len( model_tree.vec_in_cmp_or_leaf_value ) )
    assert( len_first == len( model_tree.vec_jump_cmp_lt ) )
    assert( len_first == len( model_tree.vec_jump_cmp_ge ) )

    return true
} 

Order :: enum {
    desordered = 0,
    ordered    = 1,
}

xgb_model_print :: proc ( model : XGB_Model_Eval, order : Order ) {
    for _, i in model.vec_trees {
        if order == Order.ordered {
            xgb_model_tree_print_ordered( &model.vec_trees[ i ] )
        } else { 
            xgb_model_tree_print_desordered( &model.vec_trees[ i ] )
        }
    }
}

xgb_model_tree_print_desordered :: proc ( model_tree : ^XGB_Tree ) {
    fmt.printf( "Tree %d\n", model_tree.tree_num )
    for i in 0 ..< len( model_tree.vec_in_pid_n_or_leaf ) {
        fmt.printf( "  %d: ", i )
        if model_tree.vec_in_pid_n_or_leaf[ i ] == XGB_TREE_NODE_LEAF {
            fmt.printf( "%v:leaf=%.9v\n", i, model_tree.vec_in_cmp_or_leaf_value[ i ] )
        } else {
            fmt.printf( "%v:[f%v<%.9v] yes=%v,no=%v\n",
                        i,
                        model_tree.vec_in_pid_n_or_leaf[ i ],
                        model_tree.vec_in_cmp_or_leaf_value[ i ],
                        model_tree.vec_jump_cmp_lt[ i ],
                        model_tree.vec_jump_cmp_ge[ i ] )
        }
    }
}

xgb_model_tree_print_ordered :: proc ( model_tree : ^XGB_Tree ) {
    fmt.printf( "Tree %d\n", model_tree.tree_num )
    node_index : u16 = 0
    node_level := 0
    xgb_model_tree_print_ordered_node( model_tree, node_index, node_level )
 
}

xgb_model_tree_print_ordered_node :: proc ( model_tree : ^XGB_Tree, node_index : u16, node_level : int ) {
    for i in 0 ..< node_level {
        fmt.printf( "\t" )
    }
    if model_tree.vec_in_pid_n_or_leaf[ node_index ] == XGB_TREE_NODE_LEAF {
        fmt.printf( "%v:leaf=%.9v\n", node_index, model_tree.vec_in_cmp_or_leaf_value[ node_index ] )
    } else {
        fmt.printf( "%v:[f%v<%.9v] yes=%v,no=%v\n",
                    node_index,
                    model_tree.vec_in_pid_n_or_leaf[ node_index ],
                    model_tree.vec_in_cmp_or_leaf_value[ node_index ],
                    model_tree.vec_jump_cmp_lt[ node_index ],
                    model_tree.vec_jump_cmp_ge[ node_index ] )
        node_level := node_level 
        node_level += 1
        xgb_model_tree_print_ordered_node( model_tree,
                                           model_tree.vec_jump_cmp_lt[ node_index ],
                                           node_level )
        xgb_model_tree_print_ordered_node( model_tree,
                                           model_tree.vec_jump_cmp_ge[ node_index ],
                                           node_level )
    }
}

// Parameters example:
// path     = "./bla/"
// filename = "xgb_model_01.txt"
load_model_from_txt :: proc ( path : string, filename : string ) -> ( model : XGB_Model_Eval, err: XGB_Error ) {
    // Read the entire model TXT file from disk to a string.
    file_path_and_name := fmt.aprintf( "%v%v", path, filename )
    defer delete( file_path_and_name )
    model_byte_data, ok := os.read_entire_file_from_filename( file_path_and_name )
    if !ok {
        return XGB_Model_Eval{ }, // empty struct
               XGB_Error{
                    type = Error_Type.io,
                    msg  = fmt.aprintf( "load_model_from_txt() : os.read_entire_file_from_filename() failed for file: %v",
                            file_path_and_name )
               } 
    }
    defer delete( model_byte_data )

    // Extimation of the pre-allocation of the trees vector.
    trees_num_extimate := 100
    xgb_model, err_1 := xgb_model_eval_create( trees_num_extimate )
    if err_1.type != Error_Type.none {
        // Propagate the error.
        return XGB_Model_Eval{ }, // empty struct
               err_1, 
    }

    // Parse the TXT file into a collection of trees.
    res_slice, err_2 := strings.split( string( model_byte_data ), "\n" ) 
    if err_2 != mem.Allocator_Error.None {
        return XGB_Model_Eval{ }, // empty struct
               XGB_Error{
                    type = Error_Type.allocation,
                    msg  = fmt.aprintf( "load_model_from_txt() : strings.split() failed, allocation error, for file: %s",
                            file_path_and_name )
               } 
    }
    // Only deletes the array of strings, not the strings themselves.
    defer delete( res_slice )

    // Allocate the tree
    // that will be used to parse each line into a tree of nodes,
    // so that this tree grows to the maximum size fast and that it will not be
    // need to do many grows of dynamic arrays and that, the dynamic array that is
    // allocated for each final tree is always of the final len size.

    tree_size_extimate := 1000
    model_tree_tmp, err_3 := xgb_model_tree_create( tree_size_extimate )
    if err_3.type != Error_Type.none {
        // Propagate the error.
        return XGB_Model_Eval{ }, // empty struct
               err_1, 
    }
    defer xgb_model_tree_delete( & model_tree_tmp )

    // Parse each line of the TXT file string res_slice and construct M trees each with N nodes.

    line_num     := -1
    tree_acc_num := -1
    for line in res_slice {
        line_len := len( line )
        switch {
            case line_len == 0 || ( line_len == 1 && line[0] == '\n' ) :
                // Empty line.
                line_num += 1
                continue
        
            case line_len > 0 && line[ 0 ] == '#' :
                // New tree.
                // "#0\n"
                line_num += 1
                tree_num_str := line[ 1 : ]
                tree_num, ok := strconv.parse_int( tree_num_str )
                if !ok {
                    return XGB_Model_Eval{ }, // empty struct
                        XGB_Error{
                            type = Error_Type.parse,
                            msg  = fmt.aprintf( "load_model_from_txt() :Error while reading tree_num : %d line_num : %d [%s] , strconv.parse_int() failed, for file: %s",
                                        tree_acc_num + 1,
                                        line_num,
                                        tree_num_str, 
                                        file_path_and_name )
                        } 
                }
                assert( tree_num == tree_acc_num + 1 )
            
            case line_len > 0 && ( line[ 0 ] == '0' || line[ 0 ] == '\t' )  :
                // New tree node:
                //   "0:[f12<9.72500038] yes=1,no=2,missing=1\n"
                //   "\t1:[f5<6.94099998] yes=3,no=4,missing=3\n"
                //   "\t\t3:[f7<1.48494995] yes=7,no=8,missing=7\n"
                //   "\t\t\t7:leaf=11.8800001\n"
                //   "\t\t\t8:[f5<6.54300022] yes=15,no=16,missing=15\n"
                //   "\t\t\t\t15:leaf=6.64261389\n"
                //   "\t\t\t\t16:leaf=8.01750088\n"

                line_num += 1

                if line[ 0 ] == '0' {
                    tree_acc_num += 1
                    // Copy previous tree to the model.
                    if tree_acc_num > 0 {

                        assert( xgb_model_tree_assert_soa_len( & model_tree_tmp ) )

                        // Clone the tree.
                        final_model_tree, err := xgb_model_tree_clone( model_tree_tmp )
                        if err.type != Error_Type.none {
                            // Propagate the error.
                            return XGB_Model_Eval{ }, // empty struct
                                   err, 
                        }
                        // Append the tree to the model.
                        append( & xgb_model.vec_trees, final_model_tree )
                    }

                    // New tree.
                    // Clears the model_tree_tmp to fill with the parsing of the new tree.
                    xgb_model_tree_clear( & model_tree_tmp )
                    model_tree_tmp.tree_num = tree_acc_num
                }

                // ###############
                // Parse the node.
                
                // Eat all "\t" characters at the beginning of the line.
                line_2 : string
                line_2 = strings.clone( line ) 
                for line_2[ 0 ] == '\t' {
                    line_2 = string( line_2[ 1 : ] )
                }

                // Parse the node id.
                //   "0:[f12<9.72500038] yes=1,no=2,missing=1\n"
                //   "7:leaf=11.8800001\n"
                node_num_start := 0
                node_num_end   := strings.index(line_2, ":" )
                // TODO: I'm not validating that the line has found the ":", not testign for the -1 value.
                node_num_str   := line_2[ node_num_start : node_num_end ]
                node_num, ok   := strconv.parse_int( node_num_str )
                if !ok {
                    return XGB_Model_Eval{ }, // empty struct
                        XGB_Error{
                            type = Error_Type.parse,
                            msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d [%s] strconv.parse_int() failed, for file: %s",
                                        tree_acc_num + 1,
                                        line_num,
                                        node_num_str, 
                                        file_path_and_name )
                        } 
                }
                
                // Parse the node type.
                //   "0:[f12<9.72500038] yes=1,no=2,missing=1\n"
                //   "7:leaf=11.8800001\n"
                // TODO: I'm not validating that the line has size for the next access.
                line_2 = string( line_2[ node_num_end + 1 : ] )

                // If no type is normal node, parse it's fields.
                if line_2[ 0 ] == '[' {
                    // NORMAL NODE

                    // Parse the node field.
                    //   "0:[f12<9.72500038] yes=1,no=2,missing=1\n"
                    //      ^
                    //      |
                    start_index := strings.index_any( line_2, NUMBERS_STR )
                    end_index   := strings.index( line_2, "<" ) 
                    input_num_str := line_2[ start_index : end_index ]
                    input_num, ok_1 := strconv.parse_int( input_num_str )
                    if !ok_1 {
                        return XGB_Model_Eval{ }, // empty struct
                               XGB_Error{
                                    type = Error_Type.parse,
                                    msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d input_num_str : [%s] strconv.parse_int() failed, for file: %s",
                                                tree_acc_num,
                                                line_num,
                                                input_num_str, 
                                                file_path_and_name )
                               } 
                    }

                    line_2 = string( line_2[ end_index + 1 : ] )

                    // Parse the compare_num f32 to latter compare to the input_num.value.
                    //   "9.72500038] yes=1,no=2,missing=1\n"
                    //    ^
                    //    |
                    start_index = 0
                    end_index   = strings.index( line_2, "]" ) 

                    compare_num_str := line_2[ start_index : end_index ]
                    compare_num, ok_2 := strconv.parse_f32( compare_num_str )
                    if !ok_2 {
                        return XGB_Model_Eval{ }, // empty struct
                               XGB_Error{
                                    type = Error_Type.parse,
                                    msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d compare_num_str : [%s] compare_num : [%v] strconv.parse_f32() failed, for file: %s",
                                                tree_acc_num,
                                                line_num,
                                                compare_num_str,
                                                compare_num, 
                                                file_path_and_name )
                               } 
                    }

                    line_2 = string( line_2[ end_index : ] )

                    // Parse the yes jump to int number.
                    //   "] yes=1,no=2,missing=1\n"
                    //    ^
                    //    |
                    start_index = strings.index_any( line_2, NUMBERS_STR )
                    end_index   = strings.index( line_2, "," ) 
                    
                    yes_jump_num_str := line_2[ start_index : end_index ]
                    yes_jump_num, ok_3 := strconv.parse_int( yes_jump_num_str )
                    if !ok_3 {
                        return XGB_Model_Eval{ }, // empty struct
                               XGB_Error{
                                    type = Error_Type.parse,
                                    msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d yes_jump_num_str : [%s] yes_jump_num : [%v] strconv.parse_f32() failed, for file: %s",
                                                tree_acc_num,
                                                line_num,
                                                yes_jump_num_str,
                                                yes_jump_num, 
                                                file_path_and_name )
                               } 
                    }

                    line_2 = string( line_2[ end_index + 1 : ] )
                    // Parse the yes jump to int number.
                    //   "no=2,missing=1\n"
                    //    ^
                    //    |
                    start_index = strings.index_any( line_2, NUMBERS_STR )
                    end_index   = strings.index( line_2, "," )

                    no_jump_num_str := line_2[ start_index : end_index ]
                    no_jump_num, ok_4 := strconv.parse_int( no_jump_num_str )
                    if !ok_4 {
                        return XGB_Model_Eval{ }, // empty struct
                               XGB_Error{
                                    type = Error_Type.parse,
                                    msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d no_num_str : [%s] no_num : [%v] strconv.parse_f32() failed, for file: %s",
                                                tree_acc_num,
                                                line_num,
                                                no_jump_num_str,
                                                no_jump_num, 
                                                file_path_and_name )
                               } 
                    }

                    // NOTE : The missing_num is not used / needed in my implementation. 

                    // Note: The model_tree_tmp.tree_num is already correct when
                    //       we change to a new tree and at the end, when we finish
                    //       the last tree.

                    // Insert the node to each vec of the tree.
                    // Assures that the node has that number of elements.

                    node_index := node_num
                    xgb_model_tree_insert_node_normal( & model_tree_tmp,
                                                       node_index, 
                                                       input_num,
                                                       compare_num,
                                                       yes_jump_num,
                                                       no_jump_num )


                } else if line_2[ 0 ] == 'l' {
                    // LEADF NODE
                    // If type is a leaf node, parse it's fields.


                    // Parse the leaf field.
                    //   "7:leaf=11.8800001\n"
                    //      ^
                    //      |

                    start_index := strings.index_any( line_2, NUMBERS_STR )
                    // NOTE: The "\n" is automatically removed by the strings.split() function.
                    // end_index   := strings.index( line_2, "\n" )

                    leaf_num_str := line_2[ start_index : ]
                    leaf_num, ok_1 := strconv.parse_f32( leaf_num_str )
                    if !ok_1 {
                        return XGB_Model_Eval{ }, // empty struct
                               XGB_Error{
                                    type = Error_Type.parse,
                                    msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d leaf_num_str : [%s] leaf_num : [%v] strconv.parse_f32() failed, for file: %s",
                                                tree_acc_num,
                                                line_num,
                                                leaf_num_str,
                                                leaf_num, 
                                                file_path_and_name )
                               } 
                    }

                    // Note: The model_tree_tmp.tree_num is already correct when
                    //       we change to a new tree and at the end, when we finish
                    //       the last tree.

                    // Insert the node to each vec of the tree.
                    // Assures that the node has that number of elements.

                    node_index := node_num
                    xgb_model_tree_insert_node_leaf( & model_tree_tmp,
                                                     node_index, 
                                                     leaf_num )

                } else {
                    // INVALID NODE TYPE.

                    return XGB_Model_Eval{ }, // empty struct
                        XGB_Error{
                            type = Error_Type.parse,
                            msg  = fmt.aprintf( "load_model_from_txt() : Error while reading tree_num %d, line %d [%s] line_str[%s] failed the identification of the file of the node, for file: %s",
                                        tree_acc_num + 1,
                                        line_num,
                                        node_num_str,
                                        line_2, 
                                        file_path_and_name )
                        } 
                }                

        } // switch


    } // for

    // Copy the last tree to the model.
    // Clone the tree.
    if len(model_tree_tmp.vec_in_pid_n_or_leaf) > 0 {
        
        assert( xgb_model_tree_assert_soa_len( & model_tree_tmp ) )

        final_model_tree, err := xgb_model_tree_clone( model_tree_tmp )
        if err.type != Error_Type.none {
            // Propagate the error.
            return XGB_Model_Eval{ }, // empty struct
                   err, 
        }
        // Append the tree to the model.
        append( & xgb_model.vec_trees, final_model_tree )
    }

    // Success.
    return xgb_model, XGB_Error{ type = Error_Type.none, msg = "Success: Has read all the file." }
}

// Prediction function
xgb_predict :: proc ( xgb_model : ^XGB_Model_Eval, input_slice : []f32 ) -> f64 {
       // TODO: See if this is correct?
       //    R: This is correct!
       sum : f64 = 0.5 // 0.0
       // for _, tree_index in xgb_model.vec_trees {
       for tree_index in 0 ..< len( xgb_model.vec_trees ) {
            // start_node_index : u16 = 0
            
            // res := xgb_compute_tree( & xgb_model.vec_trees[ tree_index ],
            //                          start_node_index,
            //                          input_slice )
            
            res := xgb_compute_tree_fast( & xgb_model.vec_trees[ tree_index ],
                                          input_slice )

            sum += f64( res )
       }

       // sum = 1.0 / ( 1.0 + math.exp( -sum ) )

       return sum
}

// TODO: Implement with a stack without function call's to be faster.
//       This alredy works with the "c" function call format without passing the automatic context
xgb_compute_tree :: proc "c" ( model_tree : ^XGB_Tree, node_index : u16, input_slice : []f32 ) -> f32 {
    // Traverse Tree recursively
    if model_tree.vec_in_pid_n_or_leaf[ node_index ] != XGB_TREE_NODE_LEAF {
        input_index := model_tree.vec_in_pid_n_or_leaf[ node_index ]
        if input_slice[ input_index ] < model_tree.vec_in_cmp_or_leaf_value[ node_index ]  {
            next_node_index := model_tree.vec_jump_cmp_lt[ node_index ]
            return xgb_compute_tree( model_tree,
                                     next_node_index,
                                     input_slice )
        } else {
            next_node_index := model_tree.vec_jump_cmp_ge[ node_index ]
            return xgb_compute_tree( model_tree,
                                     next_node_index,
                                     input_slice )
        }
    } else {
        return model_tree.vec_in_cmp_or_leaf_value[ node_index ]
    }
}

// Note the time is the same as the recursive version. 
//
// This alredy works with the "c" function call format without passing the automatic context
xgb_compute_tree_fast :: proc "c" ( model_tree : ^XGB_Tree, input_slice : []f32 ) -> f32 {    
    // Traverse the tree in a tight FOR loop.
    node_index : u16 = 0
    for {
        input_index := model_tree.vec_in_pid_n_or_leaf[ node_index ]
        if input_index != XGB_TREE_NODE_LEAF {
            if input_slice[ input_index ] < model_tree.vec_in_cmp_or_leaf_value[ node_index ]  {
                node_index = model_tree.vec_jump_cmp_lt[ node_index ]
            } else {
                node_index = model_tree.vec_jump_cmp_ge[ node_index ]
            }
        } else {
            return model_tree.vec_in_cmp_or_leaf_value[ node_index ]
        }
    }
}

// Optimized version for IPC 4 instructions per cycle.

// Prediction function
xgb_predict_IPC :: proc ( xgb_model : ^XGB_Model_Eval, input_slice : []f32, stride : int ) -> ( sum_1, sum_2, sum_3, sum_4 : f64 ) {
    sum_1 = 0.5 // 0.0
    sum_2 = 0.5
    sum_3 = 0.5
    sum_4 = 0.5

    // Pre-fecthing the first tree from memory to cache.
    // In this case it is alredy on the cache and not in RAM, but this is nice to know how to do it.
    //  
    // address := & xgb_model.vec_trees[ 0 ]
    // ptr := rawptr( address )
    // locality ::  3 /* high */
    // intrinsics.prefetch_read_data( ptr, locality )

    for tree_index in 0 ..< len( xgb_model.vec_trees ) {

        // address := & xgb_model.vec_trees[ tree_index ]
        // // ptr := rawptr(uintptr(address) + offset)
        // ptr := rawptr( address )
        // locality ::  3 /* high */
        // intrinsics.prefetch_read_data( ptr, locality )
             
        res_1, res_2, res_3, res_4 := xgb_compute_tree_fast_IPC( & xgb_model.vec_trees[ tree_index ],
                                            input_slice,
                                            stride )

        sum_1 += f64( res_1 )
        sum_2 += f64( res_2 )
        sum_3 += f64( res_3 )
        sum_4 += f64( res_4 )
    }

    return sum_1, sum_2, sum_3, sum_4
}

// This alredy works with the "c" function call format without passing the automatic context
xgb_compute_tree_fast_IPC :: /* #force_inline */ proc "c" ( model_tree : ^XGB_Tree, input_slice : []f32, stride : int ) ->
           ( res_1, res_2, res_3, res_4 : f32 ) {    

    // Traverse the tree in a tight FOR loop.

    NO_RESULT_YET : f32 = max( f32 )

    // NO_RESULT_YET : f32 = math.F32_MAX

    res_1 = NO_RESULT_YET
    res_2 = NO_RESULT_YET
    res_3 = NO_RESULT_YET
    res_4 = NO_RESULT_YET

    node_index_1 : u16 = 0
    node_index_2 : u16 = 0
    node_index_3 : u16 = 0
    node_index_4 : u16 = 0

    stride_1 := 0
    stride_2 := stride
    stride_3 := 2 * stride
    stride_4 := 3 * stride

    for {

        if res_1 == NO_RESULT_YET {
            input_index_1 := model_tree.vec_in_pid_n_or_leaf[ node_index_1 ]
            if input_index_1 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_1 + int( input_index_1 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_1 ]  {
                    node_index_1 = model_tree.vec_jump_cmp_lt[ node_index_1 ]
                } else {
                    node_index_1 = model_tree.vec_jump_cmp_ge[ node_index_1 ]
                }
            } else {
                res_1 = model_tree.vec_in_cmp_or_leaf_value[ node_index_1 ]
            }
        }
        
        if res_2 == NO_RESULT_YET {
            input_index_2 := model_tree.vec_in_pid_n_or_leaf[ node_index_2 ]
            if input_index_2 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_2 + int( input_index_2 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_2 ]  {
                    node_index_2 = model_tree.vec_jump_cmp_lt[ node_index_2 ]
                } else {
                    node_index_2 = model_tree.vec_jump_cmp_ge[ node_index_2 ]
                }
            } else {
                res_2 = model_tree.vec_in_cmp_or_leaf_value[ node_index_2 ]
            }
        }

        if res_3 == NO_RESULT_YET {
            input_index_3 := model_tree.vec_in_pid_n_or_leaf[ node_index_3 ]
            if input_index_3 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_3 + int( input_index_3 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_3 ]  {
                    node_index_3 = model_tree.vec_jump_cmp_lt[ node_index_3 ]
                } else {
                    node_index_3 = model_tree.vec_jump_cmp_ge[ node_index_3 ]
                }
            } else {
                res_3 = model_tree.vec_in_cmp_or_leaf_value[ node_index_3 ]
            }
        }

        if res_4 == NO_RESULT_YET {
            input_index_4 := model_tree.vec_in_pid_n_or_leaf[ node_index_4 ]
            if input_index_4 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_4 + int( input_index_4 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_4 ]  {
                    node_index_4 = model_tree.vec_jump_cmp_lt[ node_index_4 ]
                } else {
                    node_index_4 = model_tree.vec_jump_cmp_ge[ node_index_4 ]
                }
            } else {
                res_4 = model_tree.vec_in_cmp_or_leaf_value[ node_index_4 ]
            }
        }

        if res_1 != NO_RESULT_YET && 
           res_2 != NO_RESULT_YET &&
           res_3 != NO_RESULT_YET &&
           res_4 != NO_RESULT_YET {
            break
        }
    } // for
    
    return res_1, res_2, res_3, res_4        
}


// Optimized version for IPC 8 instructions per cycle.

// Prediction function
xgb_predict_IPC_2 :: proc ( xgb_model : ^XGB_Model_Eval, input_slice : []f32, stride : int ) ->
           ( sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7, sum_8 : f64 ) {
    sum_1 = 0.5 // 0.0
    sum_2 = 0.5
    sum_3 = 0.5
    sum_4 = 0.5
    sum_5 = 0.5
    sum_6 = 0.5
    sum_7 = 0.5
    sum_8 = 0.5

    for tree_index in 0 ..< len( xgb_model.vec_trees ) {
             
         res_1, res_2, res_3, res_4, 
         res_5, res_6, res_7, res_8 := xgb_compute_tree_fast_IPC_2( & xgb_model.vec_trees[ tree_index ],
                                            input_slice,
                                            stride )

         sum_1 += f64( res_1 )
         sum_2 += f64( res_2 )
         sum_3 += f64( res_3 )
         sum_4 += f64( res_4 )

         sum_5 += f64( res_5 )
         sum_6 += f64( res_6 )
         sum_7 += f64( res_7 )
         sum_8 += f64( res_8 )

    }

    return sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7, sum_8
}

// This alredy works with the "c" function call format without passing the automatic context
xgb_compute_tree_fast_IPC_2 :: proc "c" ( model_tree : ^XGB_Tree, input_slice : []f32, stride : int ) ->
            ( res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8 : f32 ) {    

    // Traverse the tree in a tight FOR loop.

    NO_RESULT_YET : f32 = max( f32 )

    res_1 = NO_RESULT_YET
    res_2 = NO_RESULT_YET
    res_3 = NO_RESULT_YET
    res_4 = NO_RESULT_YET

    res_5 = NO_RESULT_YET
    res_6 = NO_RESULT_YET
    res_7 = NO_RESULT_YET
    res_8 = NO_RESULT_YET


    node_index_1 : u16 = 0
    node_index_2 : u16 = 0
    node_index_3 : u16 = 0
    node_index_4 : u16 = 0

    node_index_5 : u16 = 0
    node_index_6 : u16 = 0
    node_index_7 : u16 = 0
    node_index_8 : u16 = 0


    stride_1 := 0
    stride_2 := stride
    stride_3 := 2 * stride
    stride_4 := 3 * stride

    stride_5 := 4 * stride
    stride_6 := 5 * stride
    stride_7 := 6 * stride
    stride_8 := 7 * stride


    for {
        if res_1 != NO_RESULT_YET && 
           res_2 != NO_RESULT_YET &&
           res_3 != NO_RESULT_YET &&
           res_4 != NO_RESULT_YET &&
           res_5 != NO_RESULT_YET &&
           res_6 != NO_RESULT_YET &&
           res_7 != NO_RESULT_YET &&
           res_8 != NO_RESULT_YET
           {
            break
        }

        if res_1 == NO_RESULT_YET {
            input_index_1 := model_tree.vec_in_pid_n_or_leaf[ node_index_1 ]
            if input_index_1 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_1 + int( input_index_1 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_1 ]  {
                    node_index_1 = model_tree.vec_jump_cmp_lt[ node_index_1 ]
                } else {
                    node_index_1 = model_tree.vec_jump_cmp_ge[ node_index_1 ]
                }
            } else {
                res_1 = model_tree.vec_in_cmp_or_leaf_value[ node_index_1 ]
            }
        }
        
        if res_2 == NO_RESULT_YET {
            input_index_2 := model_tree.vec_in_pid_n_or_leaf[ node_index_2 ]
            if input_index_2 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_2 + int( input_index_2 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_2 ]  {
                    node_index_2 = model_tree.vec_jump_cmp_lt[ node_index_2 ]
                } else {
                    node_index_2 = model_tree.vec_jump_cmp_ge[ node_index_2 ]
                }
            } else {
                res_2 = model_tree.vec_in_cmp_or_leaf_value[ node_index_2 ]
            }
        }

        if res_3 == NO_RESULT_YET {
            input_index_3 := model_tree.vec_in_pid_n_or_leaf[ node_index_3 ]
            if input_index_3 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_3 + int( input_index_3 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_3 ]  {
                    node_index_3 = model_tree.vec_jump_cmp_lt[ node_index_3 ]
                } else {
                    node_index_3 = model_tree.vec_jump_cmp_ge[ node_index_3 ]
                }
            } else {
                res_3 = model_tree.vec_in_cmp_or_leaf_value[ node_index_3 ]
            }
        }

        if res_4 == NO_RESULT_YET {
            input_index_4 := model_tree.vec_in_pid_n_or_leaf[ node_index_4 ]
            if input_index_4 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_4 + int( input_index_4 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_4 ]  {
                    node_index_4 = model_tree.vec_jump_cmp_lt[ node_index_4 ]
                } else {
                    node_index_4 = model_tree.vec_jump_cmp_ge[ node_index_4 ]
                }
            } else {
                res_4 = model_tree.vec_in_cmp_or_leaf_value[ node_index_4 ]
            }
        }

        if res_5 == NO_RESULT_YET {
            input_index_5 := model_tree.vec_in_pid_n_or_leaf[ node_index_5 ]
            if input_index_5 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_5 + int( input_index_5 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_5 ]  {
                    node_index_5 = model_tree.vec_jump_cmp_lt[ node_index_5 ]
                } else {
                    node_index_5 = model_tree.vec_jump_cmp_ge[ node_index_5 ]
                }
            } else {
                res_5 = model_tree.vec_in_cmp_or_leaf_value[ node_index_5 ]
            }
        }

        if res_6 == NO_RESULT_YET {
            input_index_6 := model_tree.vec_in_pid_n_or_leaf[ node_index_6 ]
            if input_index_6 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_6 + int( input_index_6 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_6 ]  {
                    node_index_6 = model_tree.vec_jump_cmp_lt[ node_index_6 ]
                } else {
                    node_index_6 = model_tree.vec_jump_cmp_ge[ node_index_6 ]
                }
            } else {
                res_6 = model_tree.vec_in_cmp_or_leaf_value[ node_index_6 ]
            }
        }

        if res_7 == NO_RESULT_YET {
            input_index_7 := model_tree.vec_in_pid_n_or_leaf[ node_index_7 ]
            if input_index_7 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_7 + int( input_index_7 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_7 ]  {
                    node_index_7 = model_tree.vec_jump_cmp_lt[ node_index_7 ]
                } else {
                    node_index_7 = model_tree.vec_jump_cmp_ge[ node_index_7 ]
                }
            } else {
                res_7 = model_tree.vec_in_cmp_or_leaf_value[ node_index_7 ]
            }
        }

        if res_8 == NO_RESULT_YET {
            input_index_8 := model_tree.vec_in_pid_n_or_leaf[ node_index_8 ]
            if input_index_8 != XGB_TREE_NODE_LEAF {
                if input_slice[ stride_8 + int( input_index_8 ) ] < model_tree.vec_in_cmp_or_leaf_value[ node_index_8 ]  {
                    node_index_8 = model_tree.vec_jump_cmp_lt[ node_index_8 ]
                } else {
                    node_index_8 = model_tree.vec_jump_cmp_ge[ node_index_8 ]
                }
            } else {
                res_8 = model_tree.vec_in_cmp_or_leaf_value[ node_index_8 ]
            }
        }

    } // for
    
    return res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8         
}
