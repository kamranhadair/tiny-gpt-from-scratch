"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text: str) -> list[str]:
    """
    Build a sorted vocabulary of unique characters from the input text.

    Args:
        text (str): Raw input text.

    Returns:
        list[str]: Sorted list of unique characters.
    """
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab: list[str]) -> dict[str, int]:
    """
    Build a string-to-index mapping from a vocabulary list.

    Args:
        vocab (list[str]): List of unique characters.

    Returns:
        dict[str, int]: Mapping from character to its index.
    """
    return {char: idx for idx, char in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab: list[str]) -> dict[int, str]:
    """
    Build an index-to-string mapping from a vocabulary list.

    Args:
        vocab (list[str]): List of unique characters.

    Returns:
        dict[int, str]: Mapping from index to character.
    """
    return {idx: char for idx, char in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch: str, stoi: dict[str, int]) -> int:
    """
    Encode a single character into its token ID.

    Args:
        ch (str): Character to encode.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        int: Token ID for the character.
    """
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text: str, stoi: dict[str, int]) -> list[int]:
    """
    Encode a string into a list of token IDs.

    Args:
        text (str): Input string.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        list[int]: List of token IDs.
    """
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id: int, itos: dict[int, str]) -> str:
    """
    Decode a single token ID into its corresponding character.

    Args:
        token_id (int): Integer token ID.
        itos (dict[int, str]): Index-to-character mapping.

    Returns:
        str: The decoded character.
    """
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(token_ids: list[int], itos: dict[int, str]) -> str:
    """
    Decode a sequence of token IDs into a string.

    Args:
        token_ids (list[int]): List of integer token IDs.
        itos (dict[int, str]): Index-to-character mapping.

    Returns:
        str: The decoded string.
    """
    return "".join(decode_int(token_id, itos) for token_id in token_ids)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values: list) -> np.ndarray:
    """
    Convert a Python list into a 1D NumPy array.

    Args:
        values (list): Input list of numbers.

    Returns:
        np.ndarray: 1D NumPy array containing the input values.
    """
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr: np.ndarray) -> tuple:
    """
    Return the shape of a NumPy array.

    Args:
        arr (np.ndarray): Input NumPy array.

    Returns:
        tuple: Shape of the array.
    """
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr: np.ndarray):
    """
    Return the dtype of a NumPy array.

    Args:
        arr (np.ndarray): Input NumPy array.

    Returns:
        numpy.dtype: The data type of the array elements.
    """
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows: int, cols: int) -> np.ndarray:
    """
    Create a 2D NumPy array of zeros with shape (rows, cols).

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        np.ndarray: A float64 array of zeros.
    """
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows: int, cols: int, seed: int) -> np.ndarray:
    """
    Create a reproducible 2D array of random floats in [0, 1).

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        seed (int): Seed for the random number generator.

    Returns:
        np.ndarray: Array of shape (rows, cols) with random floats.
    """
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
import numpy as np

def index_element(arr: np.ndarray, i: int, j: int):
    """
    Return the element at row i, column j of a 2D NumPy array.

    Args:
        arr (np.ndarray): Input 2D array.
        i (int): Row index.
        j (int): Column index.

    Returns:
        Scalar value at arr[i, j].
    """
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr: np.ndarray, i: int) -> np.ndarray:
    """
    Return row i of a 2D NumPy array as a 1D array.

    Args:
        arr (np.ndarray): Input 2D array.
        i (int): Row index.

    Returns:
        np.ndarray: The i-th row as a 1D array.
    """
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr: np.ndarray, j: int) -> np.ndarray:
    """
    Return column j of a 2D NumPy array as a 1D array.

    Args:
        arr (np.ndarray): Input 2D array.
        j (int): Column index.

    Returns:
        np.ndarray: The j-th column as a 1D array.
    """
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """
    Return the sub-block of a 2D NumPy array bounded by
    rows [r0, r1) and columns [c0, c1).

    Args:
        arr (np.ndarray): Input 2D array.
        r0 (int): Starting row (inclusive).
        r1 (int): Ending row (exclusive).
        c0 (int): Starting column (inclusive).
        c1 (int): Ending column (exclusive).

    Returns:
        np.ndarray: The sliced sub-block.
    """
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the element-wise sum of two NumPy arrays.

    Args:
        a (np.ndarray): First input array.
        b (np.ndarray): Second input array.

    Returns:
        np.ndarray: Element-wise sum of a and b.
    """
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the element-wise (Hadamard) product of two NumPy arrays.

    Args:
        a (np.ndarray): First input array.
        b (np.ndarray): Second input array.

    Returns:
        np.ndarray: Element-wise product of a and b.
    """
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr: np.ndarray, scalar) -> np.ndarray:
    """
    Return a new array with a scalar added to every element.

    Args:
        arr (np.ndarray): Input array.
        scalar: A Python scalar (e.g., int or float).

    Returns:
        np.ndarray: New array with the scalar added element-wise.
    """
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Add a 1D vector to every row of a 2D matrix using broadcasting.

    Args:
        matrix (np.ndarray): 2D array of shape (R, C).
        vector (np.ndarray): 1D array of shape (C,).

    Returns:
        np.ndarray: A new array of shape (R, C).
    """
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr: np.ndarray) -> np.ndarray:
    """
    Apply the exponential function element-wise to a NumPy array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with exp applied to each element.
    """
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise natural logarithm of a NumPy array.

    Args:
        arr (np.ndarray): Input array containing positive values.

    Returns:
        np.ndarray: Array with the natural log applied to each element.
    """
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr: np.ndarray):
    """
    Return the sum of all elements in a NumPy array.

    Args:
        arr (np.ndarray): Input array of any shape.

    Returns:
        Scalar: Sum of all elements.
    """
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr: np.ndarray) -> np.ndarray:
    """
    Sum a 2D NumPy array along axis 0 (rows), returning column sums.

    Args:
        arr (np.ndarray): Input array of shape (R, C).

    Returns:
        np.ndarray: 1D array of shape (C,) containing column sums.
    """
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr: np.ndarray) -> np.ndarray:
    """
    Sum a 2D NumPy array along axis 1 (columns), returning row sums.

    Args:
        arr (np.ndarray): Input array of shape (R, C).

    Returns:
        np.ndarray: 1D array of shape (R,) containing row sums.
    """
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Return maximum values along a given axis.

    Args:
        arr (np.ndarray): Input NumPy array.
        axis (int): Axis along which to compute the maximum.

    Returns:
        np.ndarray: Array with the specified axis collapsed.
    """
    return np.max(arr, axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the matrix product of two 2D NumPy arrays.

    Args:
        a (np.ndarray): Left matrix of shape (M, K).
        b (np.ndarray): Right matrix of shape (K, N).

    Returns:
        np.ndarray: Matrix product of shape (M, N).
    """
    return a @ b

# Step 28 - transpose_matrix
import numpy as np

def transpose_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Return the transpose of a 2D NumPy array.

    Args:
        arr (np.ndarray): Input array of shape (R, C).

    Returns:
        np.ndarray: Transposed array of shape (C, R).
    """
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Sum a NumPy array along the given axis while keeping
    the reduced dimension as size 1.

    Args:
        arr (np.ndarray): Input array.
        axis (int): Axis along which to sum.

    Returns:
        np.ndarray: Summed array with the reduced axis retained.
    """
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits: np.ndarray) -> np.ndarray:
    """
    Compute the naive (unstable) softmax of a 1D logits vector.

    Args:
        logits (np.ndarray): 1D array of logits.

    Returns:
        np.ndarray: Softmax probabilities summing to 1.
    """
    exp_logits = array_exp(logits)
    return exp_logits / sum_all(exp_logits)

# Step 31 - softmax_overflow_demo
import numpy as np

def softmax_overflow_demo(large_value: float) -> dict:
    exp_value = array_exp(np.array([large_value]))
    naive_exp = float(exp_value[0])

    return {
        "naive_exp": naive_exp,
        "overflowed": bool(np.isinf(naive_exp))
    }

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits: np.ndarray) -> np.ndarray:
    """
    Compute a numerically stable softmax over a 1D logits vector.

    Args:
        logits (np.ndarray): 1D array of logits.

    Returns:
        np.ndarray: Softmax probabilities.
    """
    max_logit = max_along_axis(logits, axis=0)
    shifted = logits - max_logit
    exp_shifted = array_exp(shifted)
    return exp_shifted / sum_all(exp_shifted)

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits: np.ndarray) -> np.ndarray:
    """
    Apply a numerically stable softmax to each row of a 2D logits array.

    Args:
        logits (np.ndarray): Array of shape (N, C).

    Returns:
        np.ndarray: Softmax probabilities of shape (N, C).
    """
    row_max = max_along_axis(logits, axis=1).reshape(-1, 1)
    shifted = logits - row_max
    exp_shifted = array_exp(shifted)
    row_sums = sum_keepdims(exp_shifted, axis=1)
    return exp_shifted / row_sums

# Step 34 - read_text_file
def read_text_file(text: str) -> str:
    """
    Validate and return a text corpus.

    Args:
        text (str): Raw text blob.

    Returns:
        str: The validated text.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is an empty string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if text == "":
        raise ValueError("Input string cannot be empty.")

    return text

# Step 35 - encode_corpus_to_int_array
import numpy as np

def encode_corpus_to_int_array(text: str, stoi: dict[str, int]) -> np.ndarray:
    """
    Encode a corpus string into a 1D NumPy array of int64 token IDs.

    Args:
        text (str): Input corpus.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        np.ndarray: 1D array of token IDs with dtype int64.
    """
    return np.array(encode_string(text, stoi), dtype=np.int64)

# Step 36 - pick_split_point (not yet solved)
# TODO: implement

# Step 37 - slice_train_and_val (not yet solved)
# TODO: implement

# Step 38 - pick_block_size (not yet solved)
# TODO: implement

# Step 39 - slice_x_at_offset (not yet solved)
# TODO: implement

# Step 40 - slice_y_at_offset (not yet solved)
# TODO: implement

# Step 41 - sample_random_batch_offsets (not yet solved)
# TODO: implement

# Step 42 - stack_x_batch (not yet solved)
# TODO: implement

# Step 43 - stack_y_batch (not yet solved)
# TODO: implement

# Step 44 - get_batch (not yet solved)
# TODO: implement

# Step 45 - allocate_count_matrix (not yet solved)
# TODO: implement

# Step 46 - loop_fill_counts (not yet solved)
# TODO: implement

# Step 47 - vectorize_counts_add_at (not yet solved)
# TODO: implement

# Step 48 - add_one_smoothing (not yet solved)
# TODO: implement

# Step 49 - row_sums_of_counts (not yet solved)
# TODO: implement

# Step 50 - normalize_counts_to_probs (not yet solved)
# TODO: implement

# Step 51 - sample_next_token (not yet solved)
# TODO: implement

# Step 52 - generate_sequence (not yet solved)
# TODO: implement

# Step 53 - decode_generated_sequence (not yet solved)
# TODO: implement

# Step 54 - log_prob_of_pair (not yet solved)
# TODO: implement

# Step 55 - sum_negative_log_probs (not yet solved)
# TODO: implement

# Step 56 - average_nll (not yet solved)
# TODO: implement

# Step 57 - initialize_w_random (not yet solved)
# TODO: implement

# Step 58 - scale_w_small (not yet solved)
# TODO: implement

# Step 59 - one_hot_encode_batch (not yet solved)
# TODO: implement

# Step 60 - forward_logits_onehot (not yet solved)
# TODO: implement

# Step 61 - observe_lookup_equivalence (not yet solved)
# TODO: implement

# Step 62 - forward_logits_lookup (not yet solved)
# TODO: implement

# Step 63 - logits_to_probs_rowwise (not yet solved)
# TODO: implement

# Step 64 - gather_correct_token_probs (not yet solved)
# TODO: implement

# Step 65 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 66 - derive_dlogits_on_paper (not yet solved)
# TODO: implement

# Step 67 - compute_dlogits (not yet solved)
# TODO: implement

# Step 68 - derive_dw_on_paper (not yet solved)
# TODO: implement

# Step 69 - compute_dw_scatter_add (not yet solved)
# TODO: implement

# Step 70 - sgd_update_w (not yet solved)
# TODO: implement

# Step 71 - run_one_training_step (not yet solved)
# TODO: implement

# Step 72 - train_neural_bigram_loop (not yet solved)
# TODO: implement

# Step 73 - sample_from_neural_bigram (not yet solved)
# TODO: implement

# Step 74 - linear_forward (not yet solved)
# TODO: implement

# Step 75 - derive_dx_on_paper (not yet solved)
# TODO: implement

# Step 76 - derive_linear_dw_on_paper (not yet solved)
# TODO: implement

# Step 77 - linear_backward_dx (not yet solved)
# TODO: implement

# Step 78 - linear_backward_dw (not yet solved)
# TODO: implement

# Step 79 - bias_add_forward (not yet solved)
# TODO: implement

# Step 80 - bias_add_backward_db (not yet solved)
# TODO: implement

# Step 81 - relu_forward (not yet solved)
# TODO: implement

# Step 82 - relu_backward (not yet solved)
# TODO: implement

# Step 83 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 84 - layernorm_forward_mean (not yet solved)
# TODO: implement

# Step 85 - layernorm_forward_variance (not yet solved)
# TODO: implement

# Step 86 - layernorm_forward_normalize (not yet solved)
# TODO: implement

# Step 87 - layernorm_forward_affine (not yet solved)
# TODO: implement

# Step 88 - layernorm_backward_subtract_mean (not yet solved)
# TODO: implement

# Step 89 - layernorm_backward_divide_std (not yet solved)
# TODO: implement

# Step 90 - layernorm_backward_full (not yet solved)
# TODO: implement

# Step 91 - layernorm_backward_implementation (not yet solved)
# TODO: implement

# Step 92 - create_token_embedding (not yet solved)
# TODO: implement

# Step 93 - token_embedding_forward (not yet solved)
# TODO: implement

# Step 94 - token_embedding_backward (not yet solved)
# TODO: implement

# Step 95 - create_positional_embedding (not yet solved)
# TODO: implement

# Step 96 - slice_positional_embedding (not yet solved)
# TODO: implement

# Step 97 - add_token_and_positional_embeddings (not yet solved)
# TODO: implement

# Step 98 - embedding_sum_backward (not yet solved)
# TODO: implement

# Step 99 - create_qkv_projections (not yet solved)
# TODO: implement

# Step 100 - compute_query (not yet solved)
# TODO: implement

# Step 101 - compute_key (not yet solved)
# TODO: implement

# Step 102 - compute_value (not yet solved)
# TODO: implement

# Step 103 - compute_attention_scores (not yet solved)
# TODO: implement

# Step 104 - scale_attention_scores (not yet solved)
# TODO: implement

# Step 105 - build_causal_mask (not yet solved)
# TODO: implement

# Step 106 - apply_causal_mask (not yet solved)
# TODO: implement

# Step 107 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 108 - attention_weighted_values (not yet solved)
# TODO: implement

# Step 109 - apply_output_projection (not yet solved)
# TODO: implement

# Step 110 - output_projection_backward (not yet solved)
# TODO: implement

# Step 111 - attention_value_backward (not yet solved)
# TODO: implement

# Step 112 - masked_softmax_backward (not yet solved)
# TODO: implement

# Step 113 - scale_scores_backward (not yet solved)
# TODO: implement

# Step 114 - qk_scores_backward (not yet solved)
# TODO: implement

# Step 115 - qkv_projection_backward (not yet solved)
# TODO: implement

# Step 116 - choose_attention_head_config (not yet solved)
# TODO: implement

# Step 117 - create_multihead_qkv_projections (not yet solved)
# TODO: implement

# Step 118 - create_multihead_output_projection (not yet solved)
# TODO: implement

# Step 119 - reshape_to_heads (not yet solved)
# TODO: implement

# Step 120 - transpose_heads_to_front (not yet solved)
# TODO: implement

# Step 121 - get_multihead_n_heads (not yet solved)
# TODO: implement

# Step 122 - get_multihead_sequence_length (not yet solved)
# TODO: implement

# Step 123 - compute_d_head (not yet solved)
# TODO: implement

# Step 124 - multihead_masked_softmax_scores (not yet solved)
# TODO: implement

# Step 125 - multihead_weighted_sum (not yet solved)
# TODO: implement

# Step 126 - transpose_heads_to_back (not yet solved)
# TODO: implement

# Step 127 - get_multihead_output_sequence_length (not yet solved)
# TODO: implement

# Step 128 - merge_heads_to_d_model (not yet solved)
# TODO: implement

# Step 129 - multihead_output_projection_forward (not yet solved)
# TODO: implement

# Step 130 - multihead_reshape_transpose_backward (not yet solved)
# TODO: implement

# Step 131 - ffn_linear_one_forward (not yet solved)
# TODO: implement

# Step 132 - ffn_activation_forward (not yet solved)
# TODO: implement

# Step 133 - ffn_linear_two_forward (not yet solved)
# TODO: implement

# Step 134 - ffn_backward (not yet solved)
# TODO: implement

# Step 135 - residual_forward (not yet solved)
# TODO: implement

# Step 136 - residual_backward (not yet solved)
# TODO: implement

# Step 137 - pre_layernorm_sublayer_forward (not yet solved)
# TODO: implement

# Step 138 - transformer_block_forward (not yet solved)
# TODO: implement

# Step 139 - transformer_block_backward (not yet solved)
# TODO: implement

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

