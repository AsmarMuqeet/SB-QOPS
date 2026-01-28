import multiprocessing
import time

import zarr
from parallelbar import progress_starmap
import numpy as np
from psfam.pauli_organizer import eigenvalue
from tqdm.auto import tqdm

from psfam_update.pauli_organizer import PauliOrganizer


# Define the eigenvalue function (same as before)
# def eigenvalue(i, j):
#     ib = list(bin(i)[2:])
#     jb = list(bin(j)[2:])
#     a = len(ib)
#     b = len(jb)
#     m = max(a, b)
#     ib = tobin(i, m)
#     jb = tobin(j, m)
#     s = 0
#     for k in range(m):
#         if (ib[k] == '1') & (jb[k] == '1'):
#             s = s + 1
#     s = (s + 1) % 2
#     if s == 0:
#         s = -1
#     return s
#
#
# # Convert to binary representation
# def tobin(x, s):
#     return [(x >> k) & 1 for k in range(s - 1, -1, -1)]
#
#
# # Function to compute a portion of the eigenvalues matrix in parallel
# def compute_chunk(start, end, N, eigenvalues):
#     for i in range(start, end):
#         for j in range(N):
#             eigenvalues[i, j] = eigenvalue(i, j)
#
#
# # Function to parallelize the eigenvalue computation
# def precompute_eigenvalues(N):
#     # Create an empty Zarr array to store the eigenvalues
#     eigenvalues = zarr.create(store="D:/matrix/eigenvalues20.zarr", shape=(N, N), chunks=(1000, 1000), dtype="int8",
#                               compressor=zarr.Blosc(cname="lz4", clevel=5, shuffle=2))
#
#     # Number of parallel workers (you can adjust this based on your CPU cores)
#     num_workers = multiprocessing.cpu_count()
#
#     # Calculate the range of rows each worker will process
#     chunk_size = N // num_workers
#     ranges = [(i * chunk_size, (i + 1) * chunk_size) if i < num_workers - 1 else (i * chunk_size, N) for i in
#               range(num_workers)]
#
#     progress_starmap(compute_chunk,[(start, end, N, eigenvalues) for start, end in ranges],n_cpu=num_workers)
#
#     # # Use a multiprocessing Pool to compute eigenvalues in parallel
#     # with multiprocessing.Pool(processes=num_workers) as pool:
#     #     # Each worker will compute a chunk of the matrix
#     #     pool.starmap(compute_chunk, [(start, end, N, eigenvalues) for start, end in ranges])


def vectorized_eigenvalues(N):
    m = N.bit_length()

    # Binary matrix where row i is the bit representation of i
    # Shape: (N, m), values: 0 or 1
    bin_array = ((np.arange(N)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.int8)

    # Count of positions where both i and j have a 1 in the same bit
    shared_ones = bin_array @ bin_array.T  # shape (N, N)

    # Apply the (s + 1) % 2 rule and map 0 -> -1, 1 -> 1
    parity = (shared_ones + 1) % 2
    result = 2 * parity - 1  # maps 0 → -1, 1 → 1

    return result.astype(np.int8)

def get_coefficients(qubits,paulidict):
    PO = PauliOrganizer(qubits,build=False)
    for k in paulidict.keys():
        PO.input_pauli_decomp(k, paulidict[k])

    c = []
    for state_index in tqdm(range(0, PO.N)):
        s = 0
        for string_index in range(0, PO.N):
            v = eigenvalue(string_index, state_index)
            effect = PO.pauli_decomposition[string_index][0] * v
            s = s + effect
        c = c + [s]
    return c

def get_coefficients_v(qubits,paulidict):
    PO = PauliOrganizer(qubits,build=False)
    for k in paulidict.keys():
        PO.input_pauli_decomp(k, paulidict[k])

    eigenvalue = vectorized_eigenvalues(PO.N)

    c = []
    for state_index in tqdm(range(0, PO.N)):
        s = 0
        for string_index in range(0, PO.N):
            v = int(eigenvalue[string_index, state_index])
            effect = PO.pauli_decomposition[string_index][0] * v
            s = s + effect
        c = c + [s]
    return c


import torch
import zarr
import os

def compute_binary_block_torch(start, stop, m, device='cuda'):
    """
    Returns a binary matrix of shape (stop - start, m) on the specified device,
    where each row is the binary representation of an integer.
    """
    block = torch.arange(start, stop, device=device).unsqueeze(1)
    shifts = torch.arange(m - 1, -1, -1, device=device)
    return ((block >> shifts) & 1).to(torch.float16)

def compute_eigenvalues_torch_to_zarr(N, zarr_path=r"D:\matrix\eigenvalues20.zarr", chunk_size=2**10, device='cuda'):
    """
    Computes the NxN eigenvalue matrix in chunks using PyTorch (on GPU) and writes it directly to a Zarr array on disk.
    """
    m = N.bit_length()

    # Create Zarr array for output
    store = zarr.DirectoryStore(zarr_path)
    z = zarr.open(store, mode='w', shape=(N, N), chunks=(chunk_size, chunk_size), dtype='int8',
                  compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=2))
    #z = np.zeros(shape=(N,N),dtype=np.int8)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        bin_i = compute_binary_block_torch(i_start, i_end, m, device=device)

        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            bin_j = compute_binary_block_torch(j_start, j_end, m, device=device)

            # Dot product to count shared 1s
            shared_ones = torch.matmul(bin_i, bin_j.T)

            # Apply eigenvalue rule: (s + 1) % 2 → 0 → -1, 1 → 1
            parity = (shared_ones + 1) % 2
            chunk_result = 2 * parity - 1  # shape: (i_chunk, j_chunk)

            # Move to CPU before saving to Zarr
            z[i_start:i_end, j_start:j_end] = chunk_result.cpu().numpy().astype('int8')

    # print(f"✅ Zarr array saved to: {os.path.abspath(zarr_path)}")
    return z


def get_coefficients_zarr(qubits,paulidict):
    PO = PauliOrganizer(qubits,build=False)
    for k in paulidict.keys():
        PO.input_pauli_decomp(k, paulidict[k])

    eigenvalue = zarr.open(r"D:\matrix\eigenvalues20.zarr", mode='r')

    c = []
    for state_index in tqdm(range(0, PO.N)):
        s = 0
        arr = eigenvalue[:, state_index]
        for string_index in range(0, PO.N):
            v = int(arr[string_index])
            effect = PO.pauli_decomposition[string_index][0] * v
            s = s + effect
        c = c + [s]
    return c




if __name__ == '__main__':
    # Example: Precompute eigenvalues for N = 20
    # N = 2**20
    # precompute_eigenvalues(N)

    N = 20
    paulidict = {"IZIZI".zfill(N).replace("0","I"):0.5,"IZZZI".zfill(N).replace("0","I"):0.5,
                 "ZZIZI".zfill(N).replace("0","I"):0.5,"ZZZZZ".zfill(N).replace("0","I"):0.5}
    #print(get_coefficients(N, paulidict))
    # start = time.time()
    # print(get_coefficients_v(N,paulidict))
    # end = time.time()
    # print(end - start)
    start = time.time()
    compute_eigenvalues_torch_to_zarr(2**N)
    print(get_coefficients_zarr(N, paulidict))
    end = time.time()
    print(end - start)