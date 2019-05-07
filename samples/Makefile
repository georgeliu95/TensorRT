SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
CUDA_TRIPLE=x86_64-linux
DLSW_TRIPLE=x86_64-linux
ifeq ($(TARGET), aarch64)
CUDA_TRIPLE=aarch64-linux
DLSW_TRIPLE=aarch64-linux-gnu
endif
ifeq ($(TARGET), qnx)
CUDA_TRIPLE=aarch64-qnx
DLSW_TRIPLE=aarch64-unknown-nto-qnx
endif
ifeq ($(TARGET), android32)
DLSW_TRIPLE=arm-linux-androideabi
CUDA_TRIPLE=$(DLSW_TRIPLE)
endif
ifeq ($(TARGET), android64)
DLSW_TRIPLE=aarch64-linux-androideabi
CUDA_TRIPLE=$(DLSW_TRIPLE)
endif
export TARGET
export VERBOSE
export LIBDIR
export DLSW_TRIPLE
export CUDA_TRIPLE
samples=sampleCharRNN sampleFasterRCNN sampleGoogleNet sampleINT8 sampleINT8API sampleMLP sampleMNIST sampleMNISTAPI sampleMovieLens sampleNMT sampleOnnxMNIST samplePlugin sampleSSD sampleUffMNIST sampleUffSSD trtexec

# sampleMovieLensMPS should only be compiled for Linux targets.
# sample uses Linux specific shared memory and IPC libraries.
ifeq ($(TARGET),x86_64)
	samples += sampleMovieLensMPS
endif

.PHONY: all clean help
all:
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample) &&) :

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)" &&) :
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."
	$(AT)echo "\tCUDA_INSTALL_DIR - Directory where cuda installs to."
