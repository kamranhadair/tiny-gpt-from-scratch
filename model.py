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

