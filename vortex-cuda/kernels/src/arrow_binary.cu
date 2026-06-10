// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <limits.h>
#include <stdint.h>

namespace {

constexpr uint32_t MAX_INLINED_SIZE = 12;

struct BinaryView {
    uint32_t size;
    uint8_t inline_data[MAX_INLINED_SIZE];
};

struct BinaryViewRef {
    uint32_t size;
    uint8_t prefix[4];
    uint32_t buffer_index;
    uint32_t offset;
};

// Read one validity bit from a little-endian Arrow/Vortex bitmap.
__device__ bool get_bit(const uint8_t *const validity, uint64_t bit_idx) {
    return (validity[bit_idx / 8] >> (bit_idx % 8)) & 1;
}

// Return whether a row is valid, treating a missing validity bitmap as all-valid.
__device__ bool is_valid(const uint8_t *const validity, uint32_t has_validity, uint64_t idx) {
    return has_validity == 0 || get_bit(validity, idx);
}

// Repack a sliced validity bitmap so Arrow can read it from bit offset zero.
__device__ void repack_validity_device(const uint8_t *const input,
                                       uint8_t *const output,
                                       uint64_t len,
                                       uint64_t input_offset,
                                       uint64_t output_bytes) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t start = start_elem(worker, output_bytes);
    const uint64_t stop = stop_elem(worker, output_bytes);

    for (uint64_t byte_idx = start; byte_idx < stop; byte_idx++) {
        const uint64_t first_bit = byte_idx * 8;
        const uint64_t input_bit = input_offset + first_bit;
        const uint64_t input_byte = input_bit / 8;
        const uint32_t bit_offset = static_cast<uint32_t>(input_bit % 8);
        const uint32_t bits = static_cast<uint32_t>(min(static_cast<uint64_t>(8), len - first_bit));

        uint16_t shifted = static_cast<uint16_t>(input[input_byte]) >> bit_offset;
        if (bit_offset + bits > 8) {
            shifted |= static_cast<uint16_t>(input[input_byte + 1]) << (8 - bit_offset);
        }

        uint8_t byte = static_cast<uint8_t>(shifted);
        if (bits < 8) {
            byte &= static_cast<uint8_t>((1u << bits) - 1u);
        }
        output[byte_idx] = byte;
    }
}

// Initialize scan input from BinaryView sizes. Null rows contribute zero bytes so the gather kernel
// never needs to read their view payload.
__device__ void init_scan_device(const BinaryView *const views,
                                 const uint8_t *const validity,
                                 const uint64_t *const data_buffer_lens,
                                 int32_t *const scan,
                                 uint32_t *const status,
                                 uint32_t has_validity,
                                 uint64_t data_buffer_count,
                                 uint64_t len,
                                 uint64_t scan_len) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t start = start_elem(worker, scan_len);
    const uint64_t stop = stop_elem(worker, scan_len);

    for (uint64_t idx = start; idx < stop; idx++) {
        if (idx >= len || !is_valid(validity, has_validity, idx)) {
            scan[idx] = 0;
            continue;
        }

        const BinaryView view = views[idx];
        const uint32_t size = view.size;
        if (size > static_cast<uint32_t>(INT32_MAX)) {
            scan[idx] = 0;
            atomicMax(status, 2u);
            continue;
        }

        if (size > MAX_INLINED_SIZE) {
            const BinaryViewRef *const view_ref = reinterpret_cast<const BinaryViewRef *>(&view);
            const uint64_t buffer_index = static_cast<uint64_t>(view_ref->buffer_index);
            const uint64_t offset = static_cast<uint64_t>(view_ref->offset);
            const uint64_t end = offset + static_cast<uint64_t>(size);
            if (buffer_index >= data_buffer_count || end < offset || end > data_buffer_lens[buffer_index]) {
                scan[idx] = 0;
                atomicMax(status, 1u);
                continue;
            }
        }

        scan[idx] = static_cast<int32_t>(size);
    }
}

// Validate scanned Arrow Binary offsets after CUB exclusive-sum, catching cumulative i32 overflow.
__device__ void validate_offsets_device(const BinaryView *const views,
                                        const uint8_t *const validity,
                                        const int32_t *const offsets,
                                        uint32_t *const status,
                                        uint32_t has_validity,
                                        uint64_t len) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t start = start_elem(worker, len);
    const uint64_t stop = stop_elem(worker, len);

    for (uint64_t idx = start; idx < stop; idx++) {
        const int32_t offset = offsets[idx];
        const int32_t next_offset = offsets[idx + 1];
        if (offset < 0 || next_offset < 0) {
            atomicMax(status, 2u);
            continue;
        }

        const uint32_t size = is_valid(validity, has_validity, idx) ? views[idx].size : 0;
        const int64_t expected = static_cast<int64_t>(offset) + static_cast<int64_t>(size);
        if (expected > static_cast<int64_t>(INT32_MAX) || expected != static_cast<int64_t>(next_offset)) {
            atomicMax(status, 2u);
        }
    }
}

__device__ uint64_t upper_bound_offsets(const int32_t *const offsets, uint64_t len, uint64_t value) {
    uint64_t first = 0;
    while (len > 0) {
        const uint64_t half = len / 2;
        const uint64_t mid = first + half;
        if (static_cast<uint64_t>(offsets[mid]) <= value) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return first;
}

__device__ const uint8_t *input_ptr(const BinaryView &view, const uint64_t *const data_buffer_ptrs) {
    if (view.size <= MAX_INLINED_SIZE) {
        return view.inline_data;
    }

    const BinaryViewRef *const view_ref = reinterpret_cast<const BinaryViewRef *>(&view);
    return reinterpret_cast<const uint8_t *>(data_buffer_ptrs[view_ref->buffer_index]) + view_ref->offset;
}

// Copy BinaryView payload bytes into one contiguous Arrow Binary values buffer.
__device__ void gather_device(const BinaryView *const views,
                              const uint64_t *const data_buffer_ptrs,
                              const int32_t *const offsets,
                              uint8_t *const output,
                              uint64_t len,
                              uint64_t total_bytes) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t start = start_elem(worker, total_bytes);
    const uint64_t stop = stop_elem(worker, total_bytes);
    if (start == stop) {
        return;
    }

    uint64_t row = upper_bound_offsets(offsets, len + 1, start) - 1;
    uint64_t row_start = static_cast<uint64_t>(offsets[row]);
    uint64_t row_end = static_cast<uint64_t>(offsets[row + 1]);
    BinaryView view = views[row];
    const uint8_t *input = input_ptr(view, data_buffer_ptrs);

    for (uint64_t byte_idx = start; byte_idx < stop; byte_idx++) {
        while (byte_idx >= row_end) {
            row++;
            row_start = static_cast<uint64_t>(offsets[row]);
            row_end = static_cast<uint64_t>(offsets[row + 1]);
            view = views[row];
            input = input_ptr(view, data_buffer_ptrs);
        }
        output[byte_idx] = input[byte_idx - row_start];
    }
}

} // namespace

// Copy a possibly sliced validity bitmap into Arrow's offset-zero bitmap layout.
extern "C" __global__ void arrow_binary_repack_validity(const uint8_t *const input,
                                                        uint8_t *const output,
                                                        uint64_t len,
                                                        uint64_t input_offset,
                                                        uint64_t output_bytes) {
    repack_validity_device(input, output, len, input_offset, output_bytes);
}

// Fill the CUB scan input with per-row binary lengths plus a final zero sentinel.
extern "C" __global__ void arrow_binary_init_scan(const BinaryView *const views,
                                                  const uint8_t *const validity,
                                                  const uint64_t *const data_buffer_lens,
                                                  int32_t *const scan,
                                                  uint32_t *const status,
                                                  uint32_t has_validity,
                                                  uint64_t data_buffer_count,
                                                  uint64_t len,
                                                  uint64_t scan_len) {
    init_scan_device(views,
                     validity,
                     data_buffer_lens,
                     scan,
                     status,
                     has_validity,
                     data_buffer_count,
                     len,
                     scan_len);
}

// Check that the scanned offsets are exactly the Arrow Binary offsets this input requires.
extern "C" __global__ void arrow_binary_validate_offsets(const BinaryView *const views,
                                                         const uint8_t *const validity,
                                                         const int32_t *const offsets,
                                                         uint32_t *const status,
                                                         uint32_t has_validity,
                                                         uint64_t len) {
    validate_offsets_device(views, validity, offsets, status, has_validity, len);
}

// Gather inline and referenced BinaryView payloads into Arrow Binary's contiguous values buffer.
extern "C" __global__ void arrow_binary_gather(const BinaryView *const views,
                                               const uint64_t *const data_buffer_ptrs,
                                               const int32_t *const offsets,
                                               uint8_t *const output,
                                               uint64_t len,
                                               uint64_t total_bytes) {
    gather_device(views, data_buffer_ptrs, offsets, output, len, total_bytes);
}
