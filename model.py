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

# Step 36 - pick_split_point
def pick_split_point(n: int, train_frac: float) -> int:
    """
    Compute the train/validation split index.

    Args:
        n (int): Total number of tokens.
        train_frac (float): Fraction of data to use for training.

    Returns:
        int: Split index.
    """
    return int(n * train_frac)

# Step 37 - slice_train_and_val
import numpy as np

def slice_train_and_val(data: np.ndarray, split_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an encoded corpus into training and validation arrays.

    Args:
        data (np.ndarray): 1D array of token IDs.
        split_idx (int): Index at which to split.

    Returns:
        tuple[np.ndarray, np.ndarray]: (train, val)
    """
    train = data[:split_idx]
    val = data[split_idx:]
    return train, val

# Step 38 - pick_block_size
def pick_block_size(default_size: int) -> int:
    """
    Return a valid block size (context length).

    Args:
        default_size (int): Desired block size.

    Returns:
        int: Block size, clamped to a minimum of 1.
    """
    return max(1, default_size)

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data: np.ndarray, i: int, block_size: int) -> np.ndarray:
    """
    Extract an input window of length block_size starting at index i.

    Args:
        data (np.ndarray): 1D array of token IDs.
        i (int): Starting index.
        block_size (int): Length of the window.

    Returns:
        np.ndarray: Input window of shape (block_size,).
    """
    return data[i : i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data: np.ndarray, i: int, block_size: int) -> np.ndarray:
    """
    Extract the target window of length block_size starting one position
    after index i.

    Args:
        data (np.ndarray): 1D array of token IDs.
        i (int): Starting index of the input window.
        block_size (int): Length of the window.

    Returns:
        np.ndarray: Target window of shape (block_size,).
    """
    return data[i + 1 : i + block_size + 1]

# Step 41 - sample_random_batch_offsets
import numpy as np

def sample_random_batch_offsets(
    data_length: int,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample random valid starting offsets for training windows.

    Args:
        data_length (int): Length of the encoded corpus.
        block_size (int): Context length.
        batch_size (int): Number of offsets to sample.
        rng (np.random.Generator): Seeded NumPy random generator.

    Returns:
        np.ndarray: 1D array of random starting offsets.
    """
    return rng.integers(0, data_length - block_size, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(
    data: np.ndarray,
    offsets: np.ndarray,
    block_size: int
) -> np.ndarray:
    """
    Stack input windows into a 2D batch matrix.

    Args:
        data (np.ndarray): 1D array of token IDs.
        offsets (np.ndarray): 1D array of starting indices.
        block_size (int): Length of each input window.

    Returns:
        np.ndarray: 2D array where each row is an input window.
    """
    return np.stack(
        [slice_x_at_offset(data, offset, block_size) for offset in offsets]
    )

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(
    data: np.ndarray,
    offsets,
    block_size: int
) -> np.ndarray:
    """
    Stack target windows into a 2D batch matrix.

    Args:
        data (np.ndarray): 1D array of token IDs.
        offsets: Iterable of starting indices.
        block_size (int): Length of each target window.

    Returns:
        np.ndarray: 2D array of shape (B, block_size).
    """
    return np.stack(
        [slice_y_at_offset(data, offset, block_size) for offset in offsets]
    )

# Step 44 - get_batch
import numpy as np

def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create one training batch of input and target windows.

    Args:
        data (np.ndarray): 1D array of token IDs.
        block_size (int): Context length.
        batch_size (int): Number of examples in the batch.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X: Input batch of shape (batch_size, block_size)
            Y: Target batch of shape (batch_size, block_size)
    """
    offsets = sample_random_batch_offsets(
        len(data), block_size, batch_size, rng
    )

    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)

    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size: int) -> np.ndarray:
    """
    Allocate a (V, V) integer matrix of zeros for bigram counts.

    Args:
        vocab_size (int): Size of the vocabulary.

    Returns:
        np.ndarray: Integer matrix of shape (vocab_size, vocab_size).
    """
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Fill a bigram count matrix by iterating over consecutive token pairs.

    Args:
        n_matrix (np.ndarray): Bigram count matrix of shape (V, V).
        data (np.ndarray): 1D array of token IDs.

    Returns:
        np.ndarray: The updated count matrix.
    """
    for t in range(len(data) - 1):
        n_matrix[data[t], data[t + 1]] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size: int, data: np.ndarray) -> np.ndarray:
    """
    Build a bigram count matrix using vectorized accumulation.

    Args:
        vocab_size (int): Size of the vocabulary.
        data (np.ndarray): 1D array of token IDs.

    Returns:
        np.ndarray: (V, V) int64 bigram count matrix.
    """
    N = allocate_count_matrix(vocab_size)

    current = data[:-1]
    next_tokens = data[1:]

    np.add.at(N, (current, next_tokens), 1)

    return N

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(N: np.ndarray) -> np.ndarray:
    return N + 1

# Step 49 - row_sums_of_counts
import numpy as np

def row_sums_of_counts(N: np.ndarray) -> np.ndarray:
    """
    Compute row sums of a bigram count matrix while keeping dimensions.

    Args:
        N (np.ndarray): Count matrix of shape (V, V).

    Returns:
        np.ndarray: Column vector of shape (V, 1) containing row sums.
    """
    return sum_keepdims(N, axis=1)

# Step 50 - normalize_counts_to_probs
import numpy as np

def normalize_counts_to_probs(n_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a count matrix into a row-stochastic probability matrix.

    Args:
        n_matrix (np.ndarray): Bigram count matrix of shape (V, V).

    Returns:
        np.ndarray: Probability matrix of shape (V, V), with each row summing to 1.
    """
    row_totals = row_sums_of_counts(n_matrix)
    return n_matrix / row_totals

# Step 51 - sample_next_token
import numpy as np

def sample_next_token(
    p_matrix: np.ndarray,
    current_id: int,
    rng: np.random.Generator
) -> int:
    """
    Sample the next token id from a categorical distribution.

    Args:
        p_matrix (np.ndarray): Row-stochastic probability matrix.
        current_id (int): Current token id.
        rng (np.random.Generator): Random generator.

    Returns:
        int: Sampled next token id.
    """
    probs = p_matrix[current_id]
    return int(rng.choice(len(probs), p=probs))

# Step 52 - generate_sequence
import numpy as np

def generate_sequence(
    p_matrix: np.ndarray,
    start_id: int,
    length: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a token sequence autoregressively from a bigram probability matrix.

    Args:
        p_matrix (np.ndarray): Row-stochastic bigram probability matrix.
        start_id (int): Initial token id.
        length (int): Length of generated sequence.
        rng (np.random.Generator): Random generator.

    Returns:
        np.ndarray: Generated token ids of shape (length,).
    """
    sequence = np.empty(length, dtype=np.int64)
    sequence[0] = start_id

    for i in range(1, length):
        sequence[i] = sample_next_token(
            p_matrix,
            sequence[i - 1],
            rng
        )

    return sequence

# Step 53 - decode_generated_sequence
def decode_generated_sequence(token_ids, itos: dict[int, str]) -> str:
    """
    Decode generated token ids back into a string.

    Args:
        token_ids: Python list or NumPy array of integer token ids.
        itos (dict[int, str]): Integer-to-character mapping.

    Returns:
        str: Decoded text string.
    """
    chars = [decode_int(token_id, itos) for token_id in token_ids]
    return "".join(chars)

# Step 54 - log_prob_of_pair
import numpy as np

def log_prob_of_pair(
    p_matrix: np.ndarray,
    current_id: int,
    next_id: int
) -> float:
    """
    Return the natural log probability of a bigram transition.

    Args:
        p_matrix (np.ndarray): Row-stochastic probability matrix.
        current_id (int): Current token id.
        next_id (int): Next token id.

    Returns:
        float: Natural log of P[current_id, next_id].
    """
    prob = index_element(p_matrix, current_id, next_id)
    return float(array_log(np.array([prob]))[0])

# Step 55 - sum_negative_log_probs
import numpy as np

def sum_negative_log_probs(
    p_matrix: np.ndarray,
    data: np.ndarray
) -> float:
    """
    Sum negative log probabilities of consecutive token transitions.

    Args:
        p_matrix (np.ndarray): Bigram probability matrix.
        data (np.ndarray): 1D token id sequence.

    Returns:
        float: Total negative log probability.
    """
    total = 0.0

    for t in range(len(data) - 1):
        total -= log_prob_of_pair(
            p_matrix,
            data[t],
            data[t + 1]
        )

    return float(total)

# Step 56 - average_nll
import numpy as np

def average_nll(
    p_matrix: np.ndarray,
    data: np.ndarray
) -> float:
    """
    Compute the average negative log likelihood per bigram.

    Args:
        p_matrix (np.ndarray): Row-stochastic bigram probability matrix.
        data (np.ndarray): 1D array of token ids.

    Returns:
        float: Mean negative log likelihood.
    """
    total_nll = sum_negative_log_probs(p_matrix, data)
    return float(total_nll / (len(data) - 1))

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(
    vocab_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Initialize the neural bigram weight matrix with samples from
    a standard normal distribution.

    Args:
        vocab_size (int): Size of the vocabulary.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Weight matrix of shape (vocab_size, vocab_size)
                    with dtype float64.
    """
    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale a weight matrix by a constant factor.

    Args:
        w_matrix (np.ndarray): Input weight matrix.
        scale (float): Scaling factor.

    Returns:
        np.ndarray: A new scaled weight matrix.
    """
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Convert a batch of token IDs into a one-hot encoded matrix.

    Args:
        token_ids (np.ndarray): 1D array of token IDs of length N.
        vocab_size (int): Size of the vocabulary.

    Returns:
        np.ndarray: One-hot matrix of shape (N, vocab_size).
    """
    one_hot = make_2d_zeros(len(token_ids), vocab_size)
    one_hot[np.arange(len(token_ids)), token_ids] = 1.0
    return one_hot

# Step 60 - forward_logits_onehot
import numpy as np

def forward_logits_onehot(
    onehot: np.ndarray,
    w_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute neural bigram logits from one-hot encoded inputs.

    Args:
        onehot (np.ndarray): One-hot input matrix of shape (N, V).
        w_matrix (np.ndarray): Weight matrix of shape (V, V).

    Returns:
        np.ndarray: Logits matrix of shape (N, V).
    """
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(
    w: np.ndarray,
    ids: np.ndarray
) -> dict:
    """
    Verify that one-hot matmul and direct indexing produce the same logits.

    Args:
        w (np.ndarray): Weight matrix of shape (V, V).
        ids (np.ndarray): 1D array of token IDs of length B.

    Returns:
        dict: {
            'onehot_result': (B, V) ndarray,
            'index_result': (B, V) ndarray
        }
    """
    vocab_size = w.shape[0]

    onehot = one_hot_encode_batch(ids, vocab_size)
    onehot_result = forward_logits_onehot(onehot, w)

    index_result = w[ids]

    return {
        "onehot_result": onehot_result,
        "index_result": index_result,
    }

# Step 62 - forward_logits_lookup
import numpy as np

def forward_logits_lookup(
    w: np.ndarray,
    ids: np.ndarray
) -> np.ndarray:
    """
    Compute logits by directly looking up rows of the weight matrix.

    Args:
        w (np.ndarray): Weight matrix of shape (V, V).
        ids (np.ndarray): 1D array of token IDs of shape (B,).

    Returns:
        np.ndarray: Logits matrix of shape (B, V).
    """
    return w[ids]

# Step 63 - logits_to_probs_rowwise
import numpy as np

def logits_to_probs_rowwise(logits: np.ndarray) -> np.ndarray:
    """
    Convert a batch of logits into row-wise probability distributions.

    Args:
        logits (np.ndarray): Logits matrix of shape (B, V).

    Returns:
        np.ndarray: Probability matrix of shape (B, V), where each row sums to 1.
    """
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
import numpy as np

def gather_correct_token_probs(
    probs: np.ndarray,
    targets: np.ndarray
) -> np.ndarray:
    """
    Gather the probability assigned to the correct target token for each example.

    Args:
        probs (np.ndarray): Probability matrix of shape (B, V).
        targets (np.ndarray): Target token IDs of shape (B,).

    Returns:
        np.ndarray: 1D array of length B containing the correct-class probabilities.
    """
    return probs[np.arange(len(targets)), targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(
    probs: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute the mean cross-entropy loss for a batch.

    Args:
        probs (np.ndarray): Probability matrix of shape (B, V).
        targets (np.ndarray): Target token IDs of shape (B,).

    Returns:
        float: Mean negative log-likelihood.
    """
    correct_probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(correct_probs)
    return float(-np.mean(log_probs))

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper() -> str:
    """
    Return a short derivation of the gradient of the mean cross-entropy
    loss with respect to the logits.

    Returns:
        str: Explanation of the derivation.
    """
    return (
        "Cross-entropy loss is L = -(1/B) * sum(log(p_target)), where "
        "p = softmax(logits). "
        "Using the derivative of softmax together with the chain rule, "
        "the gradient simplifies to the difference between the predicted "
        "probabilities and the one-hot encoded targets, averaged over the "
        "batch. Therefore:\n\n"
        "dL/dlogits = (probs - onehot(targets)) / B\n\n"
        "Final formula: dL/dlogits = (probs - onehot(targets)) / B"
    )

# Step 67 - compute_dlogits
import numpy as np

def compute_dlogits(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the mean cross-entropy loss with respect
    to the logits.

    Args:
        probs (np.ndarray): Softmax probabilities of shape (B, V).
        targets (np.ndarray): Integer target labels of shape (B,).

    Returns:
        np.ndarray: Gradient dL/dlogits of shape (B, V).
    """
    batch_size = probs.shape[0]

    dlogits = probs.copy()
    dlogits[np.arange(batch_size), targets] -= 1.0
    dlogits /= batch_size

    return dlogits

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper() -> str:
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    dW = np.zeros((vocab_size, dlogits.shape[1]), dtype=np.float64)
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w: np.ndarray, dw: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Perform one SGD update on the weight matrix.

    Args:
        w (np.ndarray): Current weight matrix.
        dw (np.ndarray): Gradient of the loss w.r.t. w.
        learning_rate (float): SGD learning rate.

    Returns:
        np.ndarray: Updated weight matrix.
    """
    return w - learning_rate * dw

# Step 71 - run_one_training_step
import numpy as np

def run_one_training_step(
    w: np.ndarray,
    ids: np.ndarray,
    targets: np.ndarray,
    learning_rate: float
) -> dict:
    """
    Perform one SGD training step for the neural bigram model.

    Returns:
        {
            "w": updated weight matrix,
            "loss": scalar loss before the update
        }
    """
    # Forward
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)

    # Loss
    loss = cross_entropy_loss(probs, targets)

    # Backward
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, w.shape[0])

    # SGD update
    new_w = sgd_update_w(w, dw, learning_rate)

    return {
        "w": new_w,
        "loss": float(loss),
    }

# Step 72 - train_neural_bigram_loop
import numpy as np

def train_neural_bigram_loop(
    w: np.ndarray,
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    learning_rate: float,
    num_steps: int,
    log_every: int,
) -> dict:
    """
    Train the neural bigram model.

    Returns:
        {
            "w": final weight matrix,
            "loss_history": list of logged losses
        }
    """
    rng = np.random.default_rng()
    loss_history = []

    for step in range(num_steps):
        X, Y = get_batch(data, block_size, batch_size, rng)

        ids = X.reshape(-1)
        targets = Y.reshape(-1)

        out = run_one_training_step(w, ids, targets, learning_rate)
        w = out["w"]

        if step % log_every == 0:
            loss_history.append(float(out["loss"]))

    return {
        "w": w,
        "loss_history": loss_history,
    }

# Step 73 - sample_from_neural_bigram
import numpy as np

def sample_from_neural_bigram(
    W: np.ndarray,
    start_id: int,
    num_tokens: int,
    itos: dict
) -> str:
    """
    Autoregressively sample from a neural bigram model.

    Args:
        W: Weight matrix of shape (V, V).
        start_id: Initial token id.
        num_tokens: Number of new tokens to generate.
        itos: Integer-to-string mapping.

    Returns:
        Decoded string including the starting token.
    """
    rng = np.random.default_rng()

    ids = [start_id]
    current_id = start_id

    for _ in range(num_tokens):
        logits = forward_logits_lookup(W, np.array([current_id]))
        probs = logits_to_probs_rowwise(logits)[0]

        next_id = rng.choice(len(probs), p=probs)

        ids.append(int(next_id))
        current_id = int(next_id)

    return decode_ids(ids, itos)

# Step 74 - linear_forward
import numpy as np

def linear_forward(x: np.ndarray, w: np.ndarray) -> dict:
    """
    Run a bias-free linear layer.

    Args:
        x (np.ndarray): Input of shape (B, D_in).
        w (np.ndarray): Weight matrix of shape (D_in, D_out).

    Returns:
        dict: {
            "y": Output activations of shape (B, D_out),
            "cache": {
                "x": x,
                "w": w
            }
        }
    """
    y = matmul(x, w)

    return {
        "y": y,
        "cache": {
            "x": x,
            "w": w,
        },
    }

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper() -> str:
    return (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper() -> str:
    return (
        "Y = X @ W\n"
        "The loss depends on W only through the matrix product, so applying "
        "the chain rule accumulates contributions from every example in the batch.\n"
        "dL/dW = X.T @ dY\n"
        "Shape check: X is (B, D_in), dY is (B, D_out), so dL/dW is (D_in, D_out)."
    )

# Step 77 - linear_backward_dx
import numpy as np

def linear_backward_dx(dy: np.ndarray, cache: dict) -> np.ndarray:
    """
    Compute the gradient of the loss with respect to the input X.

    Args:
        dy (np.ndarray): Upstream gradient of shape (B, D_out).
        cache (dict): Cache from linear_forward containing 'x' and 'w'.

    Returns:
        np.ndarray: Gradient with respect to X, shape (B, D_in).
    """
    w = cache["w"]
    return matmul(dy, transpose_matrix(w))

# Step 78 - linear_backward_dw
import numpy as np

def linear_backward_dw(dy: np.ndarray, cache: dict) -> np.ndarray:
    """
    Compute the gradient of the loss with respect to the weight matrix W.

    Args:
        dy (np.ndarray): Upstream gradient of shape (B, D_out).
        cache (dict): Cache from linear_forward containing 'x' and 'w'.

    Returns:
        np.ndarray: Gradient with respect to W of shape (D_in, D_out).
    """
    x = cache["x"]
    return matmul(transpose_matrix(x), dy)

# Step 79 - bias_add_forward
import numpy as np

def bias_add_forward(x: np.ndarray, b: np.ndarray) -> dict:
    """
    Add a bias vector to every row of a batch.

    Args:
        x (np.ndarray): Input matrix of shape (B, D).
        b (np.ndarray): Bias vector of shape (D,).

    Returns:
        dict: {
            "y": Output matrix of shape (B, D),
            "cache": {
                "b_shape": b.shape
            }
        }
    """
    y = vector_matrix_broadcast_add(x, b)

    return {
        "y": y,
        "cache": {
            "b_shape": b.shape,
        },
    }

# Step 80 - bias_add_backward_db
import numpy as np

def bias_add_backward_db(dy: np.ndarray, cache: dict) -> np.ndarray:
    """
    Compute the gradient with respect to the bias vector.

    Args:
        dy (np.ndarray): Upstream gradient of shape (B, D).
        cache (dict): Contains 'b_shape'.

    Returns:
        np.ndarray: Bias gradient of shape (D,).
    """
    db = sum_axis0(dy)
    return db.reshape(cache["b_shape"])

# Step 81 - relu_forward
import numpy as np

def relu_forward(x: np.ndarray) -> dict:
    """
    Apply ReLU activation elementwise.

    Args:
        x (np.ndarray): Input array of any shape.

    Returns:
        dict: {
            "y": ReLU(x),
            "cache": {
                "x": x
            }
        }
    """
    y = np.maximum(x, 0.0)

    return {
        "y": y,
        "cache": {
            "x": x,
        },
    }

# Step 82 - relu_backward
import numpy as np

def relu_backward(dy: np.ndarray, cache: dict) -> np.ndarray:
    """
    Backpropagate through a ReLU activation.

    Args:
        dy (np.ndarray): Upstream gradient.
        cache (dict): Cache from relu_forward containing 'x'.

    Returns:
        np.ndarray: Gradient with respect to the input x.
    """
    x = cache["x"]
    return dy * (x > 0)

# Step 83 - softmax_cross_entropy_backward
import numpy as np

def softmax_cross_entropy_backward(
    probs: np.ndarray,
    targets: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient of the mean cross-entropy loss with respect
    to the pre-softmax logits.

    Args:
        probs (np.ndarray): Softmax probabilities of shape (B, V).
        targets (np.ndarray): Integer target labels of shape (B,).

    Returns:
        np.ndarray: Gradient dlogits of shape (B, V).
    """
    batch_size = probs.shape[0]

    dlogits = probs.copy()
    dlogits[np.arange(batch_size), targets] -= 1.0
    dlogits /= batch_size

    return dlogits

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x: np.ndarray) -> np.ndarray:
    """
    Compute the per-row (last-axis) mean, keeping the reduced dimension.

    Args:
        x (np.ndarray): Input array of shape (..., D).

    Returns:
        np.ndarray: Mean over the last axis with shape (..., 1).
    """
    return sum_keepdims(x, axis=-1) / x.shape[-1]

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Compute the per-row (last-axis) population variance.

    Args:
        x (np.ndarray): Input array of shape (..., D).
        mean (np.ndarray): Mean over the last axis with shape (..., 1).

    Returns:
        np.ndarray: Variance over the last axis with shape (..., 1).
    """
    diff = x - mean
    sq_diff = diff ** 2
    return sum_keepdims(sq_diff, axis=-1) / x.shape[-1]

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(
    x: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float
) -> np.ndarray:
    """
    Normalize x using the provided per-row mean and variance.

    Args:
        x (np.ndarray): Input array of shape (..., D).
        mean (np.ndarray): Per-row mean of shape (..., 1).
        var (np.ndarray): Per-row variance of shape (..., 1).
        eps (float): Small constant for numerical stability.

    Returns:
        np.ndarray: Normalized array with the same shape as x.
    """
    return (x - mean) / np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
import numpy as np

def layernorm_forward_affine(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float
) -> dict:
    """
    Forward pass for LayerNorm with affine transform.

    Args:
        x (np.ndarray): Input of shape (B, D).
        gamma (np.ndarray): Scale parameter of shape (D,).
        beta (np.ndarray): Shift parameter of shape (D,).
        eps (float): Numerical stability constant.

    Returns:
        dict: {
            "y": Output of shape (B, D),
            "cache": {
                "x": x,
                "x_hat": x_hat,
                "mean": mean,
                "var": var,
                "gamma": gamma,
                "eps": eps,
            }
        }
    """
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)

    scaled = elementwise_multiply(x_hat, gamma)
    y = vector_matrix_broadcast_add(scaled, beta)

    return {
        "y": y,
        "cache": {
            "x": x,
            "x_hat": x_hat,
            "mean": mean,
            "var": var,
            "gamma": gamma,
            "eps": eps,
        },
    }

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy: np.ndarray, cache: dict) -> np.ndarray:
    """
    Backprop through x_tilde = x - mean(x).

    Args:
        dy (np.ndarray): Upstream gradient, shape (..., D).
        cache (dict): Forward cache containing 'x' and 'mean'.

    Returns:
        np.ndarray: Gradient with respect to x, same shape as dy.
    """
    d = dy.shape[-1]
    mean_dy = sum_keepdims(dy, axis=-1) / d
    return dy - mean_dy

# Step 89 - layernorm_backward_divide_std
import numpy as np

def layernorm_backward_divide_std(dy: np.ndarray, cache: dict) -> np.ndarray:
    """
    Backpropagate through the divide-by-standard-deviation step only.

    Args:
        dy (np.ndarray): Upstream gradient, shape (..., D).
        cache (dict): Forward cache containing 'x_hat', 'var', and 'eps'.

    Returns:
        np.ndarray: Gradient with respect to (x - mean), same shape as dy.
    """
    std = np.sqrt(cache["var"] + cache["eps"])
    return dy / std

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy: np.ndarray, cache: dict) -> dict:
    """
    Complete backward pass for LayerNorm.

    Args:
        dy: Upstream gradient of shape (..., D).
        cache: Forward cache from layernorm_forward_affine.

    Returns:
        dict with keys:
            'dx'
            'dgamma'
            'dbeta'
    """
    x = cache["x"]
    x_hat = cache["x_hat"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]

    D = x.shape[-1]

    # affine gradients
    dbeta = sum_axis0(dy)
    dgamma = sum_axis0(dy * x_hat)

    # gradient wrt normalized activations
    dxhat = dy * gamma

    inv_std = 1.0 / np.sqrt(var + eps)

    # standard LayerNorm backward
    dx = (
        inv_std / D
    ) * (
        D * dxhat
        - sum_keepdims(dxhat, axis=-1)
        - x_hat * sum_keepdims(dxhat * x_hat, axis=-1)
    )

    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta,
    }

# Step 91 - layernorm_backward_implementation
import numpy as np

def layernorm_backward_implementation(d_out: np.ndarray, cache: dict) -> dict:
    x = cache["x"]
    x_hat = cache["x_hat"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]

    D = x.shape[-1]

    # Parameter gradients
    dbeta = sum_axis0(d_out)
    dgamma = sum_axis0(d_out * x_hat)

    # Gradient through affine transform
    dxhat = d_out * gamma

    inv_std = 1.0 / np.sqrt(var + eps)

    # LayerNorm input gradient
    dx = (
        inv_std / D
    ) * (
        D * dxhat
        - sum_keepdims(dxhat, axis=-1)
        - x_hat * sum_keepdims(dxhat * x_hat, axis=-1)
    )

    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta,
    }

# Step 92 - create_token_embedding
import numpy as np

def create_token_embedding(vocab_size: int, d_model: int, scale: float = 0.02) -> np.ndarray:
    """
    Create a token embedding matrix.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        d_model (int): Embedding dimension.
        scale (float): Scaling factor for initialization.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, d_model).
    """
    return np.random.randn(vocab_size, d_model) * scale

# Step 93 - token_embedding_forward
import numpy as np

def token_embedding_forward(token_ids: np.ndarray, embedding_matrix: np.ndarray):
    """
    Look up token embeddings for a batch of token ids.

    Args:
        token_ids (np.ndarray): Integer array of shape (B, T).
        embedding_matrix (np.ndarray): Embedding matrix of shape (V, d_model).

    Returns:
        tuple:
            out: np.ndarray of shape (B, T, d_model)
            cache: dict containing 'token_ids' and 'vocab_size'
    """
    out = embedding_matrix[token_ids]

    cache = {
        "token_ids": token_ids,
        "vocab_size": embedding_matrix.shape[0],
    }

    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    """
    Backward pass for token embedding lookup.

    Args:
        d_out (np.ndarray): Upstream gradient of shape (B, T, d_model).
        cache (dict): Cache from token_embedding_forward containing
                      'token_ids' and 'vocab_size'.

    Returns:
        np.ndarray: Gradient with respect to the embedding matrix,
                    of shape (vocab_size, d_model).
    """
    token_ids = cache["token_ids"]
    vocab_size = cache["vocab_size"]
    d_model = d_out.shape[-1]

    dE = np.zeros((vocab_size, d_model), dtype=d_out.dtype)

    # Scatter-add gradients into the corresponding embedding rows.
    np.add.at(dE, token_ids, d_out)

    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size: int,
                                d_model: int,
                                scale: float = 0.02) -> np.ndarray:
    """
    Create a learned positional embedding matrix.

    Args:
        block_size (int): Maximum sequence length.
        d_model (int): Embedding dimension.
        scale (float): Scaling factor for initialization.

    Returns:
        np.ndarray: Positional embedding matrix of shape
                    (block_size, d_model).
    """
    P = make_2d_random(block_size, d_model, seed=None)
    return scale_w_small(P, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(P: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Return the positional embeddings for the current sequence length.

    Args:
        P (np.ndarray): Positional embedding matrix of shape (block_size, d_model).
        seq_len (int): Current sequence length.

    Returns:
        np.ndarray: Array of shape (seq_len, d_model).
    """
    return P[:seq_len]

# Step 97 - add_token_and_positional_embeddings
import numpy as np

def add_token_and_positional_embeddings(
    token_embeddings: np.ndarray,
    positional_embeddings: np.ndarray
) -> np.ndarray:
    """
    Add token and positional embeddings.

    Args:
        token_embeddings (np.ndarray): Shape (B, T, d_model).
        positional_embeddings (np.ndarray): Shape (T, d_model).

    Returns:
        np.ndarray: Combined embeddings of shape (B, T, d_model).
    """
    return token_embeddings + positional_embeddings

# Step 98 - embedding_sum_backward
import numpy as np

def embedding_sum_backward(d_out: np.ndarray) -> dict:
    """
    Backward pass for token + positional embedding addition.

    Args:
        d_out (np.ndarray): Upstream gradient of shape (B, T, d_model).

    Returns:
        dict: {
            "d_token_emb": Gradient for token embeddings, shape (B, T, d_model),
            "d_pos_emb": Gradient for positional embeddings, shape (T, d_model)
        }
    """
    return {
        "d_token_emb": d_out,
        "d_pos_emb": sum_axis0(d_out),
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    """
    Create the query, key, and value projection matrices.

    Args:
        d_model (int): Input embedding dimension.
        d_head (int): Attention head dimension.
        scale (float): Scaling factor for initialization.

    Returns:
        dict: {
            'Wq': (d_model, d_head) ndarray,
            'Wk': (d_model, d_head) ndarray,
            'Wv': (d_model, d_head) ndarray
        }
    """
    return {
        "Wq": scale_w_small(make_2d_random(d_model, d_head, seed=0), scale),
        "Wk": scale_w_small(make_2d_random(d_model, d_head, seed=1), scale),
        "Wv": scale_w_small(make_2d_random(d_model, d_head, seed=2), scale),
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    """
    Project input activations into the query space.

    Args:
        x (np.ndarray): Input tensor of shape (B, T, d_model).
        w_q (np.ndarray): Query projection matrix of shape (d_model, d_head).

    Returns:
        np.ndarray: Query tensor of shape (B, T, d_head).
    """
    return matmul(x, w_q)

# Step 101 - compute_key
import numpy as np

def compute_key(x: np.ndarray, w_k: np.ndarray) -> np.ndarray:
    """
    Project input activations into the key space.

    Args:
        x (np.ndarray): Input tensor of shape (B, T, d_model).
        w_k (np.ndarray): Key projection matrix of shape (d_model, d_head).

    Returns:
        np.ndarray: Key tensor of shape (B, T, d_head).
    """
    return matmul(x, w_k)

# Step 102 - compute_value
import numpy as np

def compute_value(x: np.ndarray, w_v: np.ndarray) -> np.ndarray:
    """
    Project input activations into the value space.

    Args:
        x (np.ndarray): Input tensor of shape (B, T, d_model).
        w_v (np.ndarray): Value projection matrix of shape (d_model, d_head).

    Returns:
        np.ndarray: Value tensor of shape (B, T, d_head).
    """
    return matmul(x, w_v)

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Compute raw attention scores.

    Args:
        q: Query tensor of shape (B, T, d_head).
        k: Key tensor of shape (B, T, d_head).

    Returns:
        Attention scores of shape (B, T, T).
    """
    return matmul(q, np.swapaxes(k, -1, -2))

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """
    Scale raw attention scores by sqrt(d_head).

    Args:
        scores: np.ndarray of shape (B, T, T)
        d_head: int, dimension of each attention head

    Returns:
        np.ndarray of shape (B, T, T)
    """
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """
    Build a causal attention mask.

    Args:
        seq_len: Length of the sequence (T)

    Returns:
        Boolean NumPy array of shape (T, T)
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """
    Apply a causal mask to scaled attention scores.

    Args:
        scaled_scores: np.ndarray of shape (B, T, T)
        causal_mask: Boolean np.ndarray of shape (T, T)

    Returns:
        np.ndarray of shape (B, T, T) with masked entries set to -np.inf.
    """
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """
    Apply a numerically stable softmax over the last axis.

    Args:
        masked_scores: np.ndarray of shape (B, T, T)

    Returns:
        np.ndarray of shape (B, T, T) containing attention weights.
    """
    shifted = masked_scores - np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """
    Compute the weighted sum of value vectors.

    Args:
        attn: np.ndarray of shape (B, T, T)
        v: np.ndarray of shape (B, T, d_head)

    Returns:
        np.ndarray of shape (B, T, d_head)
    """
    return np.matmul(attn, v)

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """
    Project attention output back to the model dimension.

    Args:
        attn_out: np.ndarray of shape (B, T, d_head)
        w_o: np.ndarray of shape (d_head, d_model)

    Returns:
        np.ndarray of shape (B, T, d_model)
    """
    return np.matmul(attn_out, w_o)

# Step 110 - output_projection_backward
import numpy as np

def output_projection_backward(d_proj, cache):
    """
    Backward pass for the output projection.

    Args:
        d_proj: np.ndarray of shape (B, T, d_model)
        cache: dict containing:
            - 'attn_out': np.ndarray of shape (B, T, d_head)
            - 'w_o': np.ndarray of shape (d_head, d_model)

    Returns:
        dict with:
            - 'd_attn_out': np.ndarray of shape (B, T, d_head)
            - 'dw_o': np.ndarray of shape (d_head, d_model)
    """
    attn_out = cache["attn_out"]
    w_o = cache["w_o"]

    # Gradient w.r.t. attention output
    d_attn_out = np.matmul(d_proj, w_o.T)

    # Gradient w.r.t. output projection matrix
    B, T, d_head = attn_out.shape
    d_model = d_proj.shape[-1]

    dw_o = (
        attn_out.reshape(B * T, d_head).T
        @ d_proj.reshape(B * T, d_model)
    )

    return {
        "d_attn_out": d_attn_out,
        "dw_o": dw_o,
    }

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """
    Backward pass for out = attn @ v.

    Args:
        d_attn_out: np.ndarray of shape (B, T, d_head)
        cache: dict containing:
            - 'attn': np.ndarray of shape (B, T, T)
            - 'v': np.ndarray of shape (B, T, d_head)

    Returns:
        dict with:
            - 'd_attn': np.ndarray of shape (B, T, T)
            - 'd_v': np.ndarray of shape (B, T, d_head)
    """
    attn = cache["attn"]
    v = cache["v"]

    d_attn = np.matmul(d_attn_out, v.transpose(0, 2, 1))
    d_v = np.matmul(attn.transpose(0, 2, 1), d_attn_out)

    return {
        "d_attn": d_attn,
        "d_v": d_v,
    }

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """
    Backward pass for the masked row-wise softmax.

    Args:
        d_attn: np.ndarray of shape (B, T, T)
        cache: dict containing:
            - 'attn': np.ndarray of shape (B, T, T)
            - 'causal_mask': np.ndarray of shape (T, T), dtype=bool

    Returns:
        np.ndarray of shape (B, T, T): gradient w.r.t. masked scores.
    """
    attn = cache["attn"]
    causal_mask = cache["causal_mask"]

    # Softmax Jacobian-vector product
    dot = np.sum(d_attn * attn, axis=-1, keepdims=True)
    d_scores = attn * (d_attn - dot)

    # Zero gradients for masked (future) positions
    d_scores = np.where(causal_mask, d_scores, 0.0)

    return d_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """
    Backward pass for attention score scaling.

    Args:
        d_scaled_scores: np.ndarray of shape (B, T, T)
        d_head: int, attention head dimension

    Returns:
        np.ndarray of shape (B, T, T)
    """
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """
    Backward pass for scores = Q @ K^T.

    Args:
        d_scores: np.ndarray of shape (B, T, T)
        cache: dict containing:
            - 'q': np.ndarray of shape (B, T, d_head)
            - 'k': np.ndarray of shape (B, T, d_head)

    Returns:
        dict with:
            - 'd_q': np.ndarray of shape (B, T, d_head)
            - 'd_k': np.ndarray of shape (B, T, d_head)
    """
    q = cache["q"]
    k = cache["k"]

    d_q = np.matmul(d_scores, k)
    d_k = np.matmul(d_scores.transpose(0, 2, 1), q)

    return {
        "d_q": d_q,
        "d_k": d_k,
    }

# Step 115 - qkv_projection_backward
import numpy as np

def qkv_projection_backward(d_q, d_k, d_v, cache):
    """
    Backward pass for Q/K/V linear projections.

    Args:
        d_q, d_k, d_v: np.ndarray of shape (B, T, d_head)
        cache: dict containing:
            - 'x': np.ndarray of shape (B, T, d_model)
            - 'w_q', 'w_k', 'w_v': np.ndarray of shape (d_model, d_head)

    Returns:
        dict with keys:
            - 'dx': (B, T, d_model)
            - 'dw_q': (d_model, d_head)
            - 'dw_k': (d_model, d_head)
            - 'dw_v': (d_model, d_head)
    """
    x = cache["x"]
    w_q = cache["w_q"]
    w_k = cache["w_k"]
    w_v = cache["w_v"]

    # Gradient w.r.t. input
    dx = (
        np.matmul(d_q, w_q.T) +
        np.matmul(d_k, w_k.T) +
        np.matmul(d_v, w_v.T)
    )

    # Flatten batch and sequence dimensions
    B, T, d_model = x.shape
    x_flat = x.reshape(B * T, d_model)

    dw_q = x_flat.T @ d_q.reshape(B * T, -1)
    dw_k = x_flat.T @ d_k.reshape(B * T, -1)
    dw_v = x_flat.T @ d_v.reshape(B * T, -1)

    return {
        "dx": dx,
        "dw_q": dw_q,
        "dw_k": dw_k,
        "dw_v": dw_v,
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """
    Build the attention head configuration.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.

    Returns:
        dict with keys:
            - 'n_heads'
            - 'd_head'
            - 'd_model'

    Raises:
        ValueError: If d_model is not divisible by n_heads.
    """
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    return {
        "n_heads": n_heads,
        "d_head": d_model // n_heads,
        "d_model": d_model,
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """
    Create the Q, K, and V projection matrices for multi-head attention.

    Args:
        d_model: Model dimension.
        scale: Scaling factor for small weight initialization.

    Returns:
        dict with keys 'Wq', 'Wk', 'Wv', each of shape (d_model, d_model).
    """
    return {
        "Wq": scale_w_small(make_2d_random(d_model, d_model, seed=0), scale),
        "Wk": scale_w_small(make_2d_random(d_model, d_model, seed=1), scale),
        "Wv": scale_w_small(make_2d_random(d_model, d_model, seed=2), scale),
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """
    Initialize the multi-head attention output projection matrix.

    Args:
        d_model: Model dimension.
        scale: Scaling factor for small weight initialization.

    Returns:
        Wo: np.ndarray of shape (d_model, d_model)
    """
    return scale_w_small(
        make_2d_random(d_model, d_model, seed=0),
        scale
    )

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """
    Split the last dimension into multiple attention heads.

    Args:
        x: np.ndarray of shape (B, T, d_model)
        n_heads: Number of attention heads.
        d_head: Dimension of each head.

    Returns:
        np.ndarray of shape (B, T, n_heads, d_head)
    """
    B, T, _ = x.shape
    return x.reshape(B, T, n_heads, d_head)

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x):
    """
    Move the heads axis before the sequence axis.

    Args:
        x: np.ndarray of shape (B, T, n_heads, d_head)

    Returns:
        np.ndarray of shape (B, n_heads, T, d_head)
    """
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3))

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    """
    Return the number of attention heads from the config.

    Args:
        config: Dictionary produced by choose_attention_head_config.

    Returns:
        Integer number of attention heads.
    """
    return config["n_heads"]

# Step 122 - get_multihead_sequence_length
def get_multihead_sequence_length(x):
    """
    Return the sequence length T from an activation tensor.

    Args:
        x: np.ndarray of shape (B, T, d_model)

    Returns:
        int: Sequence length T.
    """
    return get_array_shape(x)[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    """
    Compute the dimension of each attention head.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads.

    Returns:
        int: Dimension of each attention head.

    Raises:
        ValueError: If d_model is not evenly divisible by n_heads.
    """
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, causal_mask):
    """
    Convert raw multi-head attention scores into causal attention weights.

    Args:
        scores: np.ndarray of shape (B, n_heads, T, T)
        causal_mask: np.ndarray of shape (T, T)

    Returns:
        np.ndarray of shape (B, n_heads, T, T)
    """
    B, n_heads, T, _ = scores.shape

    # Move to (B*n_heads, T, T)
    flat_scores = scores.reshape(B * n_heads, T, T)

    # Apply causal mask
    masked_scores = apply_causal_mask(flat_scores, causal_mask)

    # Convert each attention row into a softmax distribution
    flat_rows = masked_scores.reshape(-1, T)
    weights = stable_softmax_2d_rowwise(flat_rows)

    # Restore shape
    return weights.reshape(B, n_heads, T, T)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """
    Compute weighted sum of values for each attention head.

    Args:
        weights: np.ndarray of shape (B, n_heads, T, T)
        v_heads: np.ndarray of shape (B, n_heads, T, d_head)

    Returns:
        np.ndarray of shape (B, n_heads, T, d_head)
    """
    return np.matmul(weights, v_heads)

# Step 126 - transpose_heads_to_back
import numpy as np

def transpose_heads_to_back(x):
    """
    Move the heads axis back after the sequence axis.

    Args:
        x: np.ndarray of shape (B, n_heads, T, d_head)

    Returns:
        np.ndarray of shape (B, T, n_heads, d_head)
    """
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3))

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """
    Return sequence length T from a multi-head output tensor.

    Args:
        x_heads_back: np.ndarray of shape (B, T, n_heads, d_head)

    Returns:
        int: Sequence length T.
    """
    return int(get_array_shape(x_heads_back)[1])

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x):
    """
    Merge attention heads and head dimensions into d_model.

    Args:
        x: np.ndarray of shape (B, T, n_heads, d_head)

    Returns:
        np.ndarray of shape (B, T, n_heads * d_head)
    """
    B, T, n_heads, d_head = x.shape
    return x.reshape(B, T, n_heads * d_head)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """
    Forward pass for multi-head attention output projection.

    Args:
        merged (np.ndarray): Merged multi-head output of shape (B, T, d_model).
        w_out (np.ndarray): Output projection matrix of shape (d_model, d_model).
        b_out (np.ndarray): Output projection bias of shape (d_model,).

    Returns:
        dict: {
            "out": Projected output of shape (B, T, d_model),
            "cache": {
                "merged": merged,
                "w_out": w_out,
            }
        }
    """
    linear_out = linear_forward(merged, w_out)
    biased_out = bias_add_forward(linear_out["y"], b_out)

    return {
        "out": biased_out["y"],
        "cache": {
            "merged": merged,
            "w_out": w_out,
        },
    }

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """
    Backward pass for merge + transpose heads step.

    Args:
        d_merged: np.ndarray of shape (B, T, d_model)
        shape_info: dict containing:
            - B
            - T
            - n_heads
            - d_head

    Returns:
        np.ndarray of shape (B, n_heads, T, d_head)
    """
    n_heads = shape_info["n_heads"]
    d_head = shape_info["d_head"]

    heads_back = reshape_to_heads(d_merged, n_heads, d_head)
    return transpose_heads_to_front(heads_back)

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """
    First linear layer of the Transformer feed-forward network.

    Lifts activations from d_model up to the wider hidden size d_ff.

    Args:
        x (np.ndarray): Input tensor of shape (B, T, d_model).
        w1 (np.ndarray): Weight matrix of shape (d_model, d_ff).
        b1 (np.ndarray): Bias vector of shape (d_ff,).

    Returns:
        dict: {
            "h1": Pre-activation tensor of shape (B, T, d_ff),
            "cache": {
                "x": x,
                "w1": w1,
            }
        }
    """
    linear_out = linear_forward(x, w1)
    biased_out = bias_add_forward(linear_out["y"], b1)

    return {
        "h1": biased_out["y"],
        "cache": {
            "x": x,
            "w1": w1,
        },
    }

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """
    Activation stage of the Transformer feed-forward network.

    Applies ReLU elementwise to the pre-activation hidden tensor.

    Args:
        h1 (np.ndarray): Pre-activation tensor of shape (B, T, d_ff).

    Returns:
        tuple:
            a1 (np.ndarray): Post-activation tensor of shape (B, T, d_ff).
            cache (dict): {"h1": h1}
    """
    relu_out = relu_forward(h1)

    a1 = relu_out["y"]
    cache = {"h1": h1}

    return a1, cache

# Step 133 - ffn_linear_two_forward
# ── Step 133  ffn_linear_two_forward ──
def ffn_linear_two_forward(a1, w2, b2):
    """
    Second linear projection of the FFN: (B, T, d_ff) -> (B, T, d_model).

    Reuses linear_forward (matmul) and bias_add_forward (bias broadcast)
    primitives, then packages what backward will need.
    """
    lin_out = linear_forward(a1, w2)
    h2 = bias_add_forward(lin_out['y'], b2)['y']

    cache = {
        'a1': a1,
        'w2': w2,
    }

    return {'h2': h2, 'cache': cache}

# Step 134 - ffn_backward
import numpy as np

def ffn_backward(d_out, cache):
    """
    Backward pass through the position-wise feed-forward sublayer.

    Forward path:
        h1 = x @ w1 + b1
        a1 = relu(h1)
        y  = a1 @ w2 + b2

    Args:
        d_out (np.ndarray): Upstream gradient w.r.t. y, shape (B, T, d_model).
        cache (dict): Contains 'x', 'w1', 'h1', 'a1', 'w2'
                      (shapes (B,T,d_model), (d_model,d_ff), (B,T,d_ff),
                       (B,T,d_ff), (d_ff,d_model) respectively).

    Returns:
        dict: {
            "dx": gradient w.r.t. x, shape (B, T, d_model),
            "dw1": gradient w.r.t. w1, shape (d_model, d_ff),
            "db1": gradient w.r.t. b1, shape (d_ff,),
            "dw2": gradient w.r.t. w2, shape (d_ff, d_model),
            "db2": gradient w.r.t. b2, shape (d_model,),
        }
    """
    x = cache["x"]
    w1 = cache["w1"]
    h1 = cache["h1"]
    a1 = cache["a1"]
    w2 = cache["w2"]

    B, T, d_model = x.shape
    d_ff = w1.shape[1]

    # Flatten leading (B, T) axes into a single batch axis for the 2D primitives.
    x_flat = x.reshape(B * T, d_model)
    h1_flat = h1.reshape(B * T, d_ff)
    a1_flat = a1.reshape(B * T, d_ff)
    d_out_flat = d_out.reshape(B * T, d_model)

    # --- Second linear layer: y = a1 @ w2 + b2 ---
    cache2 = {"x": a1_flat, "w": w2}
    dw2 = linear_backward_dw(d_out_flat, cache2)
    da1_flat = linear_backward_dx(d_out_flat, cache2)
    db2 = bias_add_backward_db(d_out_flat, {"b_shape": (d_model,)})

    # --- ReLU activation: a1 = relu(h1) ---
    dh1_flat = relu_backward(da1_flat, {"x": h1_flat})

    # --- First linear layer: h1 = x @ w1 + b1 ---
    cache1 = {"x": x_flat, "w": w1}
    dw1 = linear_backward_dw(dh1_flat, cache1)
    dx_flat = linear_backward_dx(dh1_flat, cache1)
    db1 = bias_add_backward_db(dh1_flat, {"b_shape": (d_ff,)})

    dx = dx_flat.reshape(B, T, d_model)

    return {
        "dx": dx,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """
    Add the sublayer output back to the input (residual/skip connection).

    Args:
        x (np.ndarray): Original input, shape (B, T, d_model).
        sublayer_out (np.ndarray): Output of the sublayer (attention or FFN),
                                    shape (B, T, d_model).

    Returns:
        np.ndarray: Elementwise sum of x and sublayer_out, shape (B, T, d_model).
    """
    return elementwise_add(x, sublayer_out)

# Step 136 - residual_backward
import numpy as np

def residual_backward(d_y):
    """
    Backward pass for a residual connection y = x + sublayer_out.

    The gradient flows unchanged to both branches, since d(x + s)/dx = 1
    and d(x + s)/ds = 1.

    Args:
        d_y (np.ndarray): Upstream gradient, shape (B, T, d_model).

    Returns:
        tuple: (d_x, d_sublayer_out), each a standalone copy of d_y with
               shape (B, T, d_model).
    """
    d_x = d_y.copy()
    d_sublayer_out = d_y.copy()

    return d_x, d_sublayer_out

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params, eps=1e-5):
    """
    Pre-LayerNorm sublayer wrapper used throughout a Transformer block.

    Computes:
        normed = LayerNorm(x)
        sub_out = sublayer_fn(normed, sublayer_params)
        y = x + sub_out   (residual connection)

    Args:
        x (np.ndarray): Input tensor, shape (B, T, d_model).
        ln_params (dict): {'gamma': (d_model,), 'beta': (d_model,)}.
        sublayer_fn (callable): Function taking (normed_x, sublayer_params)
                                 and returning {'y': ..., 'cache': ...}.
        sublayer_params: Parameters passed through to sublayer_fn.
        eps (float): LayerNorm numerical stability constant.

    Returns:
        dict: {
            'y': Output tensor of shape (B, T, d_model),
            'cache': {
                'x': x,
                'ln_cache': cache from layernorm_forward_affine,
                'sublayer_cache': cache from sublayer_fn,
            }
        }
    """
    ln_out = layernorm_forward_affine(x, ln_params['gamma'], ln_params['beta'], eps)
    normed = ln_out['y']

    sub_result = sublayer_fn(normed, sublayer_params)
    sub_out = sub_result['y']

    y = residual_forward(x, sub_out)

    cache = {
        'x': x,
        'ln_cache': ln_out['cache'],
        'sublayer_cache': sub_result['cache'],
    }

    return {'y': y, 'cache': cache}

# Step 138 - transformer_block_forward
def attn_sublayer_forward(x, attn_params):
    """
    Multi-head self-attention sublayer, used as the sublayer_fn passed
    into pre_layernorm_sublayer_forward.

    Args:
        x (np.ndarray): (already layer-normed) input, shape (B, T, d_model).
        attn_params (dict): {
            'Wq', 'Wk', 'Wv': (d_model, d_model) projection matrices,
            'Wo': (d_model, d_model) output projection matrix,
            'bo': (d_model,) output projection bias,
            'n_heads': int,
        }

    Returns:
        dict: {'y': (B, T, d_model), 'cache': {...}}
    """
    B, T, d_model = x.shape
    n_heads = attn_params['n_heads']
    d_head = compute_d_head(d_model, n_heads)

    q = compute_query(x, attn_params['Wq'])
    k = compute_key(x, attn_params['Wk'])
    v = compute_value(x, attn_params['Wv'])

    q_heads = transpose_heads_to_front(reshape_to_heads(q, n_heads, d_head))
    k_heads = transpose_heads_to_front(reshape_to_heads(k, n_heads, d_head))
    v_heads = transpose_heads_to_front(reshape_to_heads(v, n_heads, d_head))

    scores = compute_attention_scores(q_heads, k_heads)
    scaled = scale_attention_scores(scores, d_head)
    causal_mask = build_causal_mask(T)
    weights = multihead_masked_softmax_scores(scaled, causal_mask)
    weighted = multihead_weighted_sum(weights, v_heads)

    merged_heads = transpose_heads_to_back(weighted)
    merged = merge_heads_to_d_model(merged_heads)

    proj = multihead_output_projection_forward(merged, attn_params['Wo'], attn_params['bo'])

    cache = {
        'x': x,
        'q': q, 'k': k, 'v': v,
        'q_heads': q_heads, 'k_heads': k_heads, 'v_heads': v_heads,
        'weights': weights, 'weighted': weighted,
        'merged': merged,
        'causal_mask': causal_mask,
        'n_heads': n_heads, 'd_head': d_head,
        'Wq': attn_params['Wq'], 'Wk': attn_params['Wk'], 'Wv': attn_params['Wv'],
        'proj_cache': proj['cache'],
    }

    return {'y': proj['out'], 'cache': cache}


def ffn_sublayer_forward(x, ffn_params):
    """
    Position-wise feed-forward sublayer, used as the sublayer_fn passed
    into pre_layernorm_sublayer_forward.

    Args:
        x (np.ndarray): (already layer-normed) input, shape (B, T, d_model).
        ffn_params (dict): {'w1', 'b1', 'w2', 'b2'}.

    Returns:
        dict: {'y': (B, T, d_model), 'cache': {...}}
    """
    lin1 = ffn_linear_one_forward(x, ffn_params['w1'], ffn_params['b1'])
    a1, act_cache = ffn_activation_forward(lin1['h1'])
    lin2 = ffn_linear_two_forward(a1, ffn_params['w2'], ffn_params['b2'])

    cache = {
        'lin1_cache': lin1['cache'],
        'act_cache': act_cache,
        'lin2_cache': lin2['cache'],
    }

    return {'y': lin2['h2'], 'cache': cache}


def transformer_block_forward(x, block_params):
    """
    Run one pre-LN Transformer block: pre-LN attention sublayer with a
    residual connection, followed by a pre-LN feed-forward sublayer with
    a residual connection.

    Args:
        x (np.ndarray): Input tensor, shape (B, T, d_model).
        block_params (dict): {
            'ln1': {'gamma', 'beta'},
            'attn': attention sublayer params,
            'ln2': {'gamma', 'beta'},
            'ffn': feed-forward sublayer params,
        }

    Returns:
        dict: {
            'y': Output tensor, shape (B, T, d_model),
            'cache': {
                'attn_branch': cache from the attention pre-LN sublayer,
                'ffn_branch': cache from the FFN pre-LN sublayer,
            }
        }
    """
    attn_result = pre_layernorm_sublayer_forward(
        x, block_params['ln1'], attn_sublayer_forward, block_params['attn']
    )
    x_after_attn = attn_result['y']

    ffn_result = pre_layernorm_sublayer_forward(
        x_after_attn, block_params['ln2'], ffn_sublayer_forward, block_params['ffn']
    )
    y = ffn_result['y']

    return {
        'y': y,
        'cache': {
            'attn_branch': attn_result['cache'],
            'ffn_branch': ffn_result['cache'],
        },
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """
    Backward pass through one pre-LN Transformer block.

    Forward path:
        h1 = x + Attn(LN1(x))
        y  = h1 + FFN(LN2(h1))

    Args:
        d_y (np.ndarray): Upstream gradient w.r.t. the block output y,
                           shape (B, T, d_model).
        cache: cache produced alongside x by the forward pass (only used
               to recover x for _complete_block_cache).
        block_params (dict): {'ln1', 'attn', 'ln2', 'ffn'} as used by
                              transformer_block_forward.

    Returns:
        tuple: (d_x, grads)
            d_x: gradient w.r.t. the block input x, same shape as x.
            grads: {
                'ln1': {'gamma', 'beta'},
                'ln2': {'gamma', 'beta'},
                'attn': {'Wq', 'Wk', 'Wv', 'Wo', 'bo'},
                'ffn': {'w1', 'b1', 'w2', 'b2'},
            }
    """
    x = cache['attn_branch']['x']
    full_cache = _complete_block_cache(x, block_params)

    attn_branch = full_cache['attn_branch']
    ffn_branch = full_cache['ffn_branch']

    # --- FFN branch: y = h1 + FFN(LN2(h1)) ---
    d_ffn_out, ffn_grads = _ffn_sublayer_backward(
        d_y, ffn_branch['sublayer_cache'], block_params['ffn']
    )
    d_ln2_x, d_gamma2, d_beta2 = layernorm_backward_affine(
        d_ffn_out, ffn_branch['ln_cache']
    )
    # Residual skip: d_h1 gets contributions from the skip (d_y) and from
    # the sublayer branch (d_ln2_x).
    d_h1 = d_y + d_ln2_x

    # --- Attention branch: h1 = x + Attn(LN1(x)) ---
    d_attn_out, attn_grads = _attn_sublayer_backward(
        d_h1, attn_branch['sublayer_cache'], block_params['attn']
    )
    d_ln1_x, d_gamma1, d_beta1 = layernorm_backward_affine(
        d_attn_out, attn_branch['ln_cache']
    )
    # Residual skip: d_x gets contributions from the skip (d_h1) and from
    # the sublayer branch (d_ln1_x).
    d_x = d_h1 + d_ln1_x

    grads = {
        'ln1': {'gamma': d_gamma1, 'beta': d_beta1},
        'ln2': {'gamma': d_gamma2, 'beta': d_beta2},
        'attn': attn_grads,
        'ffn': ffn_grads,
    }

    return d_x, grads

# Step 140 - stack_transformer_blocks
def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff, scale=0.02):
    """
    Build the parameter dictionaries for a stack of Transformer blocks.

    Args:
        n_layers (int): Number of Transformer blocks to create.
        d_model (int): Model (embedding) dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): Hidden dimension of the feed-forward network.
        scale (float): Scaling factor for weight initialization.

    Returns:
        list[dict]: A list of length n_layers, where each entry is:
            {
                'ln1': {'gamma': ones(d_model), 'beta': zeros(d_model)},
                'attn': {
                    'Wq': (d_model, d_model), 'Wk': (d_model, d_model),
                    'Wv': (d_model, d_model), 'Wo': (d_model, d_model),
                    'bo': zeros(d_model),
                },
                'ln2': {'gamma': ones(d_model), 'beta': zeros(d_model)},
                'ffn': {
                    'W1': (d_model, d_ff), 'b1': zeros(d_ff),
                    'W2': (d_ff, d_model), 'b2': zeros(d_model),
                },
            }
    """
    blocks = []

    for _ in range(n_layers):
        ln1 = {
            'gamma': np.ones(d_model),
            'beta': np.zeros(d_model),
        }

        attn = {
            'Wq': scale_w_small(make_2d_random(d_model, d_model, seed=0), scale),
            'Wk': scale_w_small(make_2d_random(d_model, d_model, seed=1), scale),
            'Wv': scale_w_small(make_2d_random(d_model, d_model, seed=2), scale),
            'Wo': scale_w_small(make_2d_random(d_model, d_model, seed=3), scale),
            'bo': np.zeros(d_model),
        }

        ln2 = {
            'gamma': np.ones(d_model),
            'beta': np.zeros(d_model),
        }

        ffn = {
            'W1': scale_w_small(make_2d_random(d_model, d_ff, seed=4), scale),
            'b1': np.zeros(d_ff),
            'W2': scale_w_small(make_2d_random(d_ff, d_model, seed=5), scale),
            'b2': np.zeros(d_model),
        }

        blocks.append({
            'ln1': ln1,
            'attn': attn,
            'ln2': ln2,
            'ffn': ffn,
        })

    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """
    Run the input sequentially through every Transformer block.

    Args:
        x (np.ndarray): Input tensor, shape (B, T, d_model).
        blocks (list[dict]): List of per-block parameter dicts, each in
                              the format consumed by transformer_block_forward.

    Returns:
        tuple:
            y (np.ndarray): Output after passing through all blocks,
                             same shape as x.
            caches (list[dict]): Per-block cache dicts, in block order,
                                  one per entry in `blocks`.
    """
    y = x
    caches = []

    for block_params in blocks:
        block_out = transformer_block_forward(y, block_params)
        y = block_out['y']
        caches.append(block_out['cache'])

    return y, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    """
    Propagate an upstream gradient back through a stack of Transformer blocks.

    Args:
        d_y (np.ndarray): Upstream gradient w.r.t. the stack's output,
                           shape (B, T, d_model).
        caches (list[dict]): Per-block forward caches, in block order
                              (as produced by forward_through_all_blocks).
        blocks (list[dict]): Per-block parameter dicts, in block order,
                              aligned with `caches`.

    Returns:
        tuple:
            d_x (np.ndarray): Gradient w.r.t. the stack's input, same
                               shape as d_y.
            grads (list[dict]): Per-block parameter gradient dicts, in
                                 the same order as `blocks`.
    """
    d_current = d_y
    grads_reversed = []

    for cache, block_params in zip(reversed(caches), reversed(blocks)):
        d_current, block_grads = transformer_block_backward(d_current, cache, block_params)
        grads_reversed.append(block_grads)

    grads = list(reversed(grads_reversed))

    return d_current, grads

# Step 143 - final_layernorm_forward
def final_layernorm_forward(x, gamma, beta, eps=1e-5):
    """
    Apply LayerNorm to the activations coming out of the last Transformer
    block, normalizing each (B, T) position independently across the
    d_model channels.

    Args:
        x (np.ndarray): Input tensor, shape (B, T, d_model).
        gamma (np.ndarray): Learnable scale, shape (d_model,).
        beta (np.ndarray): Learnable shift, shape (d_model,).
        eps (float): Numerical stability constant.

    Returns:
        tuple:
            y (np.ndarray): Normalized + affine-transformed output,
                             shape (B, T, d_model).
            cache (dict): {
                'x': x,
                'mean': per-position mean, shape (B, T, 1),
                'var': per-position variance, shape (B, T, 1),
                'x_hat': normalized activations, shape (B, T, d_model),
                'gamma': gamma,
            }
    """
    ln_out = layernorm_forward_affine(x, gamma, beta, eps)

    cache = {
        'x': ln_out['cache']['x'],
        'mean': ln_out['cache']['mean'],
        'var': ln_out['cache']['var'],
        'x_hat': ln_out['cache']['x_hat'],
        'gamma': ln_out['cache']['gamma'],
    }

    return ln_out['y'], cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    """
    Language model head: project final hidden states to vocabulary logits.

    Args:
        x (np.ndarray): Final hidden states, shape (B, T, d_model).
        w_lm (np.ndarray): LM head weight matrix, shape (d_model, vocab_size).
        b_lm (np.ndarray): LM head bias vector, shape (vocab_size,).

    Returns:
        dict: {
            "logits": Output tensor of shape (B, T, vocab_size),
            "cache": {
                "x": x,
                "w_lm": w_lm,
            }
        }
    """
    lin_out = linear_forward(x, w_lm)
    biased_out = bias_add_forward(lin_out['y'], b_lm)

    cache = {
        "x": x,
        "w_lm": w_lm,
    }

    return {"logits": biased_out['y'], "cache": cache}

# Step 145 - full_model_forward
def full_model_forward(token_ids, model_params):
    """Run the full Tiny GPT forward pass end-to-end."""
    B, T = token_ids.shape

    # Token + positional embeddings
    tok_embeddings, tok_emb_cache = token_embedding_forward(token_ids, model_params['tok_emb'])
    pos_embeddings = slice_positional_embedding(model_params['pos_emb'], T)
    x = add_token_and_positional_embeddings(tok_embeddings, pos_embeddings)

    emb_cache = {
        'tok_emb_cache': tok_emb_cache,
        'seq_len': T,
    }

    # Transformer block stack
    x, block_caches = forward_through_all_blocks(x, model_params['blocks'])

    # Final LayerNorm
    x, ln_f_cache = final_layernorm_forward(
        x, model_params['ln_f']['gamma'], model_params['ln_f']['beta']
    )

    # LM head
    lm_out = lm_head_linear_forward(
        x, model_params['lm_head']['w_lm'], model_params['lm_head']['b_lm']
    )
    logits = lm_out['logits']

    caches = {
        'emb': emb_cache,
        'blocks': block_caches,
        'ln_f': ln_f_cache,
        'lm_head': lm_out['cache'],
    }

    return logits, caches

# Step 146 - full_model_backward
import numpy as np

def full_model_backward(d_logits, caches, model_params):
    """
    Propagate gradients from the LM head logits back to every parameter
    in the Tiny GPT model, mirroring model_params's nested structure.
    """
    B, T, V = d_logits.shape

    # --- LM head linear backward: logits = x @ w_lm + b_lm ---
    lm_x = caches['lm_head']['x']
    lm_w = caches['lm_head']['w_lm']
    d_model = lm_x.shape[-1]

    d_logits_flat = d_logits.reshape(B * T, V)
    lm_cache = {'x': lm_x.reshape(B * T, d_model), 'w': lm_w}

    dw_lm = linear_backward_dw(d_logits_flat, lm_cache)
    dx_lm_flat = linear_backward_dx(d_logits_flat, lm_cache)
    db_lm = bias_add_backward_db(d_logits_flat, {'b_shape': (V,)})

    d_ln_f_out = dx_lm_flat.reshape(B, T, d_model)

    # --- Final LayerNorm backward (manual, since the cache has no 'eps') ---
    ln_cache = caches['ln_f']
    x_hat = ln_cache['x_hat']
    var = ln_cache['var']
    gamma = ln_cache['gamma']
    eps = 1e-5
    D = x_hat.shape[-1]

    dbeta_f = d_ln_f_out.sum(axis=(0, 1))
    dgamma_f = (d_ln_f_out * x_hat).sum(axis=(0, 1))

    dxhat = d_ln_f_out * gamma
    inv_std = 1.0 / np.sqrt(var + eps)
    d_blocks_out = (inv_std / D) * (
        D * dxhat
        - dxhat.sum(axis=-1, keepdims=True)
        - x_hat * (dxhat * x_hat).sum(axis=-1, keepdims=True)
    )

    # --- Transformer block stack backward ---
    d_emb, block_grads = backward_through_all_blocks(
        d_blocks_out, caches['blocks'], model_params['blocks']
    )

    # --- Embedding sum backward: x = tok_emb[token_ids] + pos_emb[:T] ---
    token_ids = caches['emb']['tok_cache']['token_ids']
    seq_len = caches['emb']['seq_len']

    d_tok_emb = np.zeros_like(model_params['tok_emb'])
    np.add.at(d_tok_emb, token_ids, d_emb)

    d_pos_emb = np.zeros_like(model_params['pos_emb'])
    d_pos_emb[:seq_len] = d_emb.sum(axis=0)

    return {
        'tok_emb': d_tok_emb,
        'pos_emb': d_pos_emb,
        'blocks': block_grads,
        'ln_f': {'gamma': dgamma_f, 'beta': dbeta_f},
        'lm_head': {'w_lm': dw_lm, 'b_lm': db_lm},
    }

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    """
    Allocate zeroed first- and second-moment buffers mirroring the
    structure of a (possibly nested) parameter dictionary/list tree.

    Args:
        model_params: A nested structure of dicts / lists whose leaves are
                       NumPy arrays (e.g. model_params as used throughout
                       this project).

    Returns:
        tuple: (m, v), each a structure parallel to model_params where
               every array leaf is replaced with np.zeros_like(leaf).
    """
    def _zeros_like_tree(node):
        if isinstance(node, dict):
            return {key: _zeros_like_tree(value) for key, value in node.items()}
        elif isinstance(node, list):
            return [_zeros_like_tree(item) for item in node]
        elif isinstance(node, np.ndarray):
            return np.zeros_like(node)
        else:
            # Non-array, non-container leaf (e.g. plain int/float config
            # value) — leave it untouched, no moment buffer needed for it.
            return node

    m = _zeros_like_tree(model_params)
    v = _zeros_like_tree(model_params)

    return m, v

# Step 148 - initialize_adam_step_counter
# ── Step 148  initialize_adam_step_counter ──
def initialize_adam_step_counter() -> int:
    """
    Return the initial Adam time-step counter, before any update has
    been applied.

    Returns:
        int: 0
    """
    return 0

# Step 149 - adam_increment_step
# ── Step 149  adam_increment_step ──
def adam_increment_step(t: int) -> int:
    """
    Advance Adam's time-step counter by one.

    Args:
        t (int): Current step count.

    Returns:
        int: t + 1, the step count to use for the upcoming update's
             bias correction.
    """
    return t + 1

# Step 150 - adam_update_first_moment
# ── Step 150  adam_update_first_moment ──
import numpy as np

def adam_update_first_moment(m: np.ndarray, grad: np.ndarray, beta1: float) -> np.ndarray:
    """
    Update Adam's first-moment estimate via an exponential moving average
    of the gradient.

    Args:
        m (np.ndarray): Previous first-moment estimate, same shape as grad.
        grad (np.ndarray): Current gradient.
        beta1 (float): First-moment decay rate.

    Returns:
        np.ndarray: Updated first-moment estimate, same shape as m/grad.
    """
    return beta1 * m + (1.0 - beta1) * grad

# Step 151 - adam_update_second_moment
import numpy as np

def adam_update_second_moment(v_prev: np.ndarray, grad: np.ndarray, beta2: float) -> np.ndarray:
    """
    Update Adam's second-moment estimate via an exponential moving average
    of the squared gradient.

    Args:
        v_prev (np.ndarray): Previous second-moment estimate, same shape as grad.
        grad (np.ndarray): Current gradient.
        beta2 (float): Second-moment decay rate.

    Returns:
        np.ndarray: Updated second-moment estimate, same shape as v_prev/grad.
    """
    return beta2 * v_prev + (1.0 - beta2) * (grad ** 2)

# Step 152 - adam_bias_correction
# ── Step 152  adam_bias_correction ──
import numpy as np

def adam_bias_correction(m: np.ndarray, v: np.ndarray, beta1: float, beta2: float, t: int):
    """
    Debias Adam's first- and second-moment estimates at step t.

    Args:
        m (np.ndarray): Raw (biased) first-moment estimate.
        v (np.ndarray): Raw (biased) second-moment estimate.
        beta1 (float): First-moment decay rate.
        beta2 (float): Second-moment decay rate.
        t (int): Current step count (1-indexed).

    Returns:
        tuple: (m_hat, v_hat), the bias-corrected first- and
               second-moment estimates, same shapes as m/v.
    """
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    return m_hat, v_hat

# Step 153 - adam_parameter_update
import numpy as np

def adam_parameter_update(param: np.ndarray, m_hat: np.ndarray, v_hat: np.ndarray,
                            lr: float, eps: float) -> np.ndarray:
    """
    Apply the final Adam update to a single parameter tensor.

    Args:
        param (np.ndarray): Current parameter values.
        m_hat (np.ndarray): Bias-corrected first-moment estimate.
        v_hat (np.ndarray): Bias-corrected second-moment estimate.
        lr (float): Learning rate.
        eps (float): Small constant for numerical stability.

    Returns:
        np.ndarray: Updated parameter array, same shape as param. A fresh
                    array is returned; param is not modified in place.
    """
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 154 - wire_full_training_loop
# ── Step 154  wire_full_training_loop ──
import numpy as np

def _adam_update_tree(params_node, grads_node, m_node, v_node, t, lr, beta1, beta2, eps):
    """
    Recursively walk parallel params/grads/m/v trees (dicts, lists, and
    ndarray leaves) and apply one Adam update at every leaf.

    Returns:
        tuple: (new_params_node, new_m_node, new_v_node)
    """
    if isinstance(params_node, dict):
        new_params = {}
        new_m = {}
        new_v = {}
        for key in params_node:
            new_params[key], new_m[key], new_v[key] = _adam_update_tree(
                params_node[key], grads_node[key], m_node[key], v_node[key],
                t, lr, beta1, beta2, eps
            )
        return new_params, new_m, new_v

    elif isinstance(params_node, list):
        new_params = []
        new_m = []
        new_v = []
        for p_item, g_item, m_item, v_item in zip(params_node, grads_node, m_node, v_node):
            up_p, up_m, up_v = _adam_update_tree(
                p_item, g_item, m_item, v_item, t, lr, beta1, beta2, eps
            )
            new_params.append(up_p)
            new_m.append(up_m)
            new_v.append(up_v)
        return new_params, new_m, new_v

    elif isinstance(params_node, np.ndarray):
        new_m = adam_update_first_moment(m_node, grads_node, beta1)
        new_v = adam_update_second_moment(v_node, grads_node, beta2)
        m_hat, v_hat = adam_bias_correction(new_m, new_v, beta1, beta2, t)
        new_params = adam_parameter_update(params_node, m_hat, v_hat, lr, eps)
        return new_params, new_m, new_v

    else:
        # Non-array, non-container leaf: nothing to update.
        return params_node, m_node, v_node


def _bridge_emb_cache_key(caches):
    """
    full_model_forward may store the token-embedding cache under
    'tok_emb_cache' while full_model_backward expects 'tok_cache'.
    Bridge the two names without touching either earlier step.
    """
    emb_cache = caches.get('emb', {})
    if 'tok_cache' not in emb_cache and 'tok_emb_cache' in emb_cache:
        emb_cache['tok_cache'] = emb_cache['tok_emb_cache']
    return caches


def wire_full_training_loop(params, train_ids, val_ids, block_size, batch_size,
                             n_steps, lr, betas, eps):
    """
    Drive the full GPT training loop for n_steps iterations using Adam.

    Returns:
        tuple:
            params (dict): Updated parameter tree after n_steps of training.
            history (list[dict]): [{'step': step, 'train_loss': loss}, ...]
    """
    beta1, beta2 = betas
    rng = np.random.default_rng()

    m, v = initialize_adam_moments(params)
    t = initialize_adam_step_counter()

    history = []

    for step in range(n_steps):
        X, Y = get_batch(train_ids, block_size, batch_size, rng)

        logits, caches = full_model_forward(X, params)
        caches = _bridge_emb_cache_key(caches)

        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = Y.reshape(-1)

        probs_flat = logits_to_probs_rowwise(logits_flat)
        loss = cross_entropy_loss(probs_flat, targets_flat)

        dlogits_flat = compute_dlogits(probs_flat, targets_flat)
        dlogits = dlogits_flat.reshape(B, T, V)

        grads = full_model_backward(dlogits, caches, params)

        t = adam_increment_step(t)
        params, m, v = _adam_update_tree(params, grads, m, v, t, lr, beta1, beta2, eps)

        history.append({'step': step, 'train_loss': float(loss)})

    return params, history

# Step 155 - logging_and_validation_loss
# ── Step 155  logging_and_validation_loss ──
import numpy as np

def logging_and_validation_loss(params, val_ids, block_size, batch_size, n_eval_batches):
    """
    Estimate held-out cross-entropy loss by averaging over n_eval_batches
    batches drawn from val_ids.

    Args:
        params (dict): Nested model parameter tree.
        val_ids (np.ndarray): 1D array of validation token ids.
        block_size (int): Context length.
        batch_size (int): Number of examples per batch.
        n_eval_batches (int): Number of batches to average the loss over.

    Returns:
        float: Mean per-batch cross-entropy loss.
    """
    rng = np.random.default_rng(0)

    losses = []

    for _ in range(n_eval_batches):
        X, Y = get_batch(val_ids, block_size, batch_size, rng)

        logits, _ = full_model_forward(X, params)

        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = Y.reshape(-1)

        probs_flat = logits_to_probs_rowwise(logits_flat)
        loss = cross_entropy_loss(probs_flat, targets_flat)

        losses.append(float(loss))

    return float(np.mean(losses))

# Step 156 - encode_prompt
# ── Step 156  encode_prompt ──
import numpy as np

def encode_prompt(prompt: str, stoi: dict[str, int]) -> np.ndarray:
    """
    Encode a raw prompt string into a batch tensor of token ids.

    Args:
        prompt (str): Raw input prompt.
        stoi (dict[str, int]): Character-to-index mapping.

    Returns:
        np.ndarray: Integer array of shape (1, T), where T = len(prompt).
    """
    ids = encode_string(prompt, stoi)
    return np.array([ids], dtype=np.int64)

# Step 157 - crop_context_to_block_size
import numpy as np

def crop_context_to_block_size(context: np.ndarray, block_size: int) -> np.ndarray:
    """
    Keep only the most recent block_size tokens of a running context.

    Args:
        context (np.ndarray): Integer array of shape (1, T).
        block_size (int): Maximum context length supported by the model's
                           positional embeddings.

    Returns:
        np.ndarray: Array of shape (1, min(T, block_size)), containing the
                    tail of context if T > block_size, otherwise context
                    unchanged.
    """
    T = context.shape[1]

    if T <= block_size:
        return context

    return context[:, T - block_size:]

# Step 158 - forward_to_get_logits
def forward_to_get_logits(model_params, context):
    """
    Thin inference-mode wrapper around the full model forward pass.

    Runs the complete Tiny GPT forward pipeline and returns only the
    logits, discarding the intermediate caches since backprop isn't
    needed at generation time.

    Args:
        model_params (dict): Nested model parameter tree, as consumed by
                              full_model_forward.
        context (np.ndarray): Integer token id array of shape (1, T).

    Returns:
        np.ndarray: Logits of shape (1, T, vocab_size).
    """
    logits, _ = full_model_forward(context, model_params)
    return logits

# Step 159 - take_last_position_logits
import numpy as np

def take_last_position_logits(logits: np.ndarray) -> np.ndarray:
    """
    Extract the logits at the final time step of a sequence.

    Args:
        logits (np.ndarray): Logits of shape (1, T, vocab_size).

    Returns:
        np.ndarray: Logits at the last position, shape (1, vocab_size).
    """
    return logits[:, -1, :]

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

