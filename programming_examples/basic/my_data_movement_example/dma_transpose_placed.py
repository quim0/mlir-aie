import argparse
import numpy as np
import sys

import aie.dialects.index as index_dialect
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

COLUMNS = 8
REPETITIONS = 64
DATA_SIZE = 15 * (2**10) # 20KiB

def my_passthrough():
    # 10KB
    data_ty = np.ndarray[(DATA_SIZE,), np.dtype[np.int8]]
    host_data_ty = np.ndarray[(REPETITIONS*COLUMNS*DATA_SIZE,), np.dtype[np.int8]]

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            tiles = []
            ofs_comp_to_mem = []
            ofs_comp_to_mem_c2 = []
            ofs = []
            ofs_c2 = []
            for column_idx in range(COLUMNS):
                ComputeTile = tile(column_idx, 2)
                tiles.append(ComputeTile)
                MemTile = tile(column_idx, 1)
                ShimTile = tile(column_idx, 0)

                of_compute_to_memtile = object_fifo(f"out_to_memtime_col{column_idx}", ComputeTile, MemTile, 2, data_ty)
                of_compute_to_memtile_channel2 = object_fifo(f"out_to_memtime_col{column_idx}c2", ComputeTile, MemTile, 2, data_ty)

                of_memtile_to_shim = object_fifo(f"out_to_shim_col{column_idx}", MemTile, ShimTile, 2, data_ty)
                of_memtile_to_shim_channel2 = object_fifo(f"out_to_shim_col{column_idx}c2", MemTile, ShimTile, 2, data_ty)

                object_fifo_link(of_compute_to_memtile, of_memtile_to_shim)
                object_fifo_link(of_compute_to_memtile_channel2, of_memtile_to_shim_channel2)

                ofs.append(of_memtile_to_shim)
                ofs_c2.append(of_memtile_to_shim_channel2)
                ofs_comp_to_mem.append(of_compute_to_memtile)
                ofs_comp_to_mem_c2.append(of_compute_to_memtile_channel2)

            for column_idx in range(COLUMNS):
                @core(tiles[column_idx])
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        # Just acquire and release the two objects
                        mem = ofs_comp_to_mem[column_idx].acquire(ObjectFifoPort.Produce, 1)
                        mem2 = ofs_comp_to_mem_c2[column_idx].acquire(ObjectFifoPort.Produce, 1)
                        ofs_comp_to_mem[column_idx].release(ObjectFifoPort.Produce, 1)
                        ofs_comp_to_mem_c2[column_idx].release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(host_data_ty)
            def sequence(A):
                """
                Data is organized like this.
                (A buffer):
                COLUMN0_ObjectFifo0_repetition0
                COLUMN1_ObjectFifo0_repetition0
                COLUMN2_ObjectFifo0_repetition0
                COLUMN3_ObjectFifo0_repetition0
                COLUMN0_ObjectFifo0_repetition1
                COLUMN1_ObjectFifo0_repetition1
                COLUMN2_ObjectFifo0_repetition1
                COLUMN3_ObjectFifo0_repetition1
                ...
                COLUMN0_ObjectFifo1_repetition0
                COLUMN1_ObjectFifo1_repetition0
                COLUMN2_ObjectFifo1_repetition0
                COLUMN3_ObjectFifo1_repetition0
                COLUMN0_ObjectFifo1_repetition1
                COLUMN1_ObjectFifo1_repetition1
                COLUMN2_ObjectFifo1_repetition1
                COLUMN3_ObjectFifo1_repetition1
                ...
                """

                # Send data from first objectFifo (REPETITIONS*DATA_SIZE_COLUMN)
                for column_idx in range(COLUMNS):
                    npu_dma_memcpy_nd(
                    metadata=ofs[column_idx],
                    bd_id=3,
                    mem=A,
                    # repeat REPETITIONS times
                    sizes=[REPETITIONS, 1, 1, DATA_SIZE],
                    strides=[DATA_SIZE*COLUMNS, 0, 0, 1],
                    offsets=[0, 0, 0, column_idx*DATA_SIZE],
                    issue_token=True
                    )

                # Send data from second objectFifo
                for column_idx in range(COLUMNS):
                    npu_dma_memcpy_nd(
                    metadata=ofs_c2[column_idx],
                    bd_id=4,
                    mem=A,
                    # repeat REPETITIONS times
                    sizes=[REPETITIONS, 1, 1, DATA_SIZE],
                    strides=[DATA_SIZE*COLUMNS, 0, 0, 1],
                    # ObjectFifo1 starts to write at the end of the ObjectFifo0 space
                    offsets=[0, 0, 0, column_idx*DATA_SIZE + REPETITIONS*COLUMNS*DATA_SIZE],
                    issue_token=True
                    )

                # Wait only once at the end of the program
                for column_idx in range(COLUMNS):
                    dma_wait(ofs[column_idx], ofs_c2[column_idx])
                    
    print(ctx.module)

if __name__ == "__main__":
    my_passthrough()
