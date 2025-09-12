##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

CHESS ?= false

all: build/final.xclbin

devicename ?= npu2#$(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_scalar_add

aie_py_src=${targetname}.py
use_placed?=1

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif


build/kernel.o: ${srcdir}/kernel.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F} -j 24
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif
else ifeq ($(devicename),npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -c $< -o ${@F} -j 24
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
endif
else
	@echo "Device type not supported"
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/final.xclbin: build/aie.mlir build/kernel.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%) -j 24
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
	--no-xchesscc --no-xbridge \
	--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

endif
		
${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
