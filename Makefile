CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CORE_OBJS = ds4.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression rust-test

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-metadata-dump ds4-sampling-dump ds4-logits-dump ds4-restore-dump ds4-decode-policy-dump ds4-kv-policy-dump ds4-kvc-file-dump ds4-kv-trailer-dump ds4-session-payload-dump ds4-first-kernel-oracle-dump ds4-layer0-attn-hc-pre-oracle-dump ds4-layer0-qkv-rope-oracle-dump ds4-layer0-attn-output-oracle-dump ds4-layer0-ffn-output-oracle-dump ds4-layer0-output-head-oracle-dump ds4-two-layer-output-head-oracle-dump ds4-layer2-compressor-state-oracle-dump ds4-layer2-attn-output-oracle-dump ds4-layer2-ffn-output-oracle-dump ds4-layer3-ffn-output-oracle-dump ds4-layer4-ffn-output-oracle-dump ds4-all-layer-final-hc-oracle-dump ds4-full-output-head-oracle-dump ds4-short-continuation-output-head-oracle-dump ds4-ratio-boundary-output-head-oracle-dump ds4-graph-checkpoint-dump

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent, and dump helpers"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent, and dump helpers"
	@echo "  make rust-test    Run Rust workspace tests"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-metadata-dump: ds4_metadata_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_metadata_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-sampling-dump: ds4_sampling_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_sampling_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-logits-dump: ds4_logits_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_logits_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-restore-dump: ds4_restore_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_restore_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-decode-policy-dump: ds4_decode_policy_dump.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_decode_policy_dump.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-kv-policy-dump: ds4_kv_policy_dump.o ds4_kvstore.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_kv_policy_dump.o ds4_kvstore.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-kvc-file-dump: ds4_kvc_file_dump.o ds4_kvstore.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_kvc_file_dump.o ds4_kvstore.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-kv-trailer-dump: ds4_kv_trailer_dump.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_kv_trailer_dump.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-session-payload-dump: ds4_session_payload_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_session_payload_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-first-kernel-oracle-dump: ds4_first_kernel_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_first_kernel_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer0-attn-hc-pre-oracle-dump: ds4_layer0_attn_hc_pre_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer0_attn_hc_pre_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer0-qkv-rope-oracle-dump: ds4_layer0_qkv_rope_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer0_qkv_rope_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer0-attn-output-oracle-dump: ds4_layer0_attn_output_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer0_attn_output_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer0-ffn-output-oracle-dump: ds4_layer0_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer0_ffn_output_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer0-output-head-oracle-dump: ds4_layer0_output_head_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer0_output_head_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-two-layer-output-head-oracle-dump: ds4_two_layer_output_head_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_two_layer_output_head_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer2-compressor-state-oracle-dump: ds4_layer2_compressor_state_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer2_compressor_state_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer2-attn-output-oracle-dump: ds4_layer2_attn_output_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer2_attn_output_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer2-ffn-output-oracle-dump: ds4_layer2_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer2_ffn_output_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer3-ffn-output-oracle-dump: ds4_layer3_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer3_ffn_output_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-layer4-ffn-output-oracle-dump: ds4_layer4_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_layer4_ffn_output_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-all-layer-final-hc-oracle-dump: ds4_all_layer_final_hc_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_all_layer_final_hc_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-full-output-head-oracle-dump: ds4_full_output_head_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_full_output_head_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-short-continuation-output-head-oracle-dump: ds4_short_continuation_output_head_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_short_continuation_output_head_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-ratio-boundary-output-head-oracle-dump: ds4_ratio_boundary_output_head_oracle_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_ratio_boundary_output_head_oracle_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-graph-checkpoint-dump: ds4_graph_checkpoint_dump.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_graph_checkpoint_dump.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_metadata_dump_cpu.o ds4_sampling_dump_cpu.o ds4_logits_dump_cpu.o ds4_restore_dump_cpu.o ds4_decode_policy_dump_cpu.o ds4_kv_policy_dump_cpu.o ds4_kvc_file_dump_cpu.o ds4_kv_trailer_dump_cpu.o ds4_session_payload_dump_cpu.o ds4_first_kernel_oracle_dump_cpu.o ds4_layer0_attn_hc_pre_oracle_dump_cpu.o ds4_layer0_qkv_rope_oracle_dump_cpu.o ds4_layer0_attn_output_oracle_dump_cpu.o ds4_layer0_ffn_output_oracle_dump_cpu.o ds4_layer0_output_head_oracle_dump_cpu.o ds4_two_layer_output_head_oracle_dump_cpu.o ds4_layer2_compressor_state_oracle_dump_cpu.o ds4_layer2_attn_output_oracle_dump_cpu.o ds4_layer2_ffn_output_oracle_dump_cpu.o ds4_layer3_ffn_output_oracle_dump_cpu.o ds4_layer4_ffn_output_oracle_dump_cpu.o ds4_all_layer_final_hc_oracle_dump_cpu.o ds4_full_output_head_oracle_dump_cpu.o ds4_short_continuation_output_head_oracle_dump_cpu.o ds4_ratio_boundary_output_head_oracle_dump_cpu.o ds4_graph_checkpoint_dump_cpu.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-metadata-dump ds4_metadata_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-sampling-dump ds4_sampling_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-logits-dump ds4_logits_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-restore-dump ds4_restore_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-decode-policy-dump ds4_decode_policy_dump_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-kv-policy-dump ds4_kv_policy_dump_cpu.o ds4_kvstore.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-kvc-file-dump ds4_kvc_file_dump_cpu.o ds4_kvstore.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-kv-trailer-dump ds4_kv_trailer_dump_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-session-payload-dump ds4_session_payload_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-first-kernel-oracle-dump ds4_first_kernel_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-attn-hc-pre-oracle-dump ds4_layer0_attn_hc_pre_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-qkv-rope-oracle-dump ds4_layer0_qkv_rope_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-attn-output-oracle-dump ds4_layer0_attn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-ffn-output-oracle-dump ds4_layer0_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-output-head-oracle-dump ds4_layer0_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-two-layer-output-head-oracle-dump ds4_two_layer_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer2-compressor-state-oracle-dump ds4_layer2_compressor_state_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer2-attn-output-oracle-dump ds4_layer2_attn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer2-ffn-output-oracle-dump ds4_layer2_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer3-ffn-output-oracle-dump ds4_layer3_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer4-ffn-output-oracle-dump ds4_layer4_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-all-layer-final-hc-oracle-dump ds4_all_layer_final_hc_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-full-output-head-oracle-dump ds4_full_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-short-continuation-output-head-oracle-dump ds4_short_continuation_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-ratio-boundary-output-head-oracle-dump ds4_ratio_boundary_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-graph-checkpoint-dump ds4_graph_checkpoint_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent, and dump helpers"
	@echo "  make rust-test           Run Rust workspace tests"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-metadata-dump ds4-sampling-dump ds4-logits-dump ds4-restore-dump ds4-decode-policy-dump ds4-kv-policy-dump ds4-kvc-file-dump ds4-kv-trailer-dump ds4-session-payload-dump ds4-first-kernel-oracle-dump ds4-layer0-attn-hc-pre-oracle-dump ds4-layer0-qkv-rope-oracle-dump ds4-layer0-attn-output-oracle-dump ds4-layer0-ffn-output-oracle-dump ds4-layer0-output-head-oracle-dump ds4-two-layer-output-head-oracle-dump ds4-layer2-compressor-state-oracle-dump ds4-layer2-attn-output-oracle-dump ds4-layer2-ffn-output-oracle-dump ds4-layer3-ffn-output-oracle-dump ds4-layer4-ffn-output-oracle-dump ds4-all-layer-final-hc-oracle-dump ds4-full-output-head-oracle-dump ds4-short-continuation-output-head-oracle-dump ds4-ratio-boundary-output-head-oracle-dump ds4-graph-checkpoint-dump CUDA_ARCH=

cuda-generic:
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-metadata-dump ds4-sampling-dump ds4-logits-dump ds4-restore-dump ds4-decode-policy-dump ds4-kv-policy-dump ds4-kvc-file-dump ds4-kv-trailer-dump ds4-session-payload-dump ds4-first-kernel-oracle-dump ds4-layer0-attn-hc-pre-oracle-dump ds4-layer0-qkv-rope-oracle-dump ds4-layer0-attn-output-oracle-dump ds4-layer0-ffn-output-oracle-dump ds4-layer0-output-head-oracle-dump ds4-two-layer-output-head-oracle-dump ds4-layer2-compressor-state-oracle-dump ds4-layer2-attn-output-oracle-dump ds4-layer2-ffn-output-oracle-dump ds4-layer3-ffn-output-oracle-dump ds4-layer4-ffn-output-oracle-dump ds4-all-layer-final-hc-oracle-dump ds4-full-output-head-oracle-dump ds4-short-continuation-output-head-oracle-dump ds4-ratio-boundary-output-head-oracle-dump ds4-graph-checkpoint-dump CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-metadata-dump ds4-sampling-dump ds4-logits-dump ds4-restore-dump ds4-decode-policy-dump ds4-kv-policy-dump ds4-kvc-file-dump ds4-kv-trailer-dump ds4-session-payload-dump ds4-first-kernel-oracle-dump ds4-layer0-attn-hc-pre-oracle-dump ds4-layer0-qkv-rope-oracle-dump ds4-layer0-attn-output-oracle-dump ds4-layer0-ffn-output-oracle-dump ds4-layer0-output-head-oracle-dump ds4-two-layer-output-head-oracle-dump ds4-layer2-compressor-state-oracle-dump ds4-layer2-attn-output-oracle-dump ds4-layer2-ffn-output-oracle-dump ds4-layer3-ffn-output-oracle-dump ds4-layer4-ffn-output-oracle-dump ds4-all-layer-final-hc-oracle-dump ds4-full-output-head-oracle-dump ds4-short-continuation-output-head-oracle-dump ds4-ratio-boundary-output-head-oracle-dump ds4-graph-checkpoint-dump CUDA_ARCH="$(CUDA_ARCH)"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: ds4_agent.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-metadata-dump: ds4_metadata_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-sampling-dump: ds4_sampling_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-logits-dump: ds4_logits_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-restore-dump: ds4_restore_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-decode-policy-dump: ds4_decode_policy_dump.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-kv-policy-dump: ds4_kv_policy_dump.o ds4_kvstore.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-kvc-file-dump: ds4_kvc_file_dump.o ds4_kvstore.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-kv-trailer-dump: ds4_kv_trailer_dump.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-session-payload-dump: ds4_session_payload_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-first-kernel-oracle-dump: ds4_first_kernel_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer0-attn-hc-pre-oracle-dump: ds4_layer0_attn_hc_pre_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer0-qkv-rope-oracle-dump: ds4_layer0_qkv_rope_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer0-attn-output-oracle-dump: ds4_layer0_attn_output_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer0-ffn-output-oracle-dump: ds4_layer0_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer0-output-head-oracle-dump: ds4_layer0_output_head_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-two-layer-output-head-oracle-dump: ds4_two_layer_output_head_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer2-compressor-state-oracle-dump: ds4_layer2_compressor_state_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer2-attn-output-oracle-dump: ds4_layer2_attn_output_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer2-ffn-output-oracle-dump: ds4_layer2_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer3-ffn-output-oracle-dump: ds4_layer3_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-layer4-ffn-output-oracle-dump: ds4_layer4_ffn_output_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-all-layer-final-hc-oracle-dump: ds4_all_layer_final_hc_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-full-output-head-oracle-dump: ds4_full_output_head_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-short-continuation-output-head-oracle-dump: ds4_short_continuation_output_head_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-ratio-boundary-output-head-oracle-dump: ds4_ratio_boundary_output_head_oracle_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-graph-checkpoint-dump: ds4_graph_checkpoint_dump.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_metadata_dump_cpu.o ds4_sampling_dump_cpu.o ds4_logits_dump_cpu.o ds4_restore_dump_cpu.o ds4_decode_policy_dump_cpu.o ds4_kv_policy_dump_cpu.o ds4_kvc_file_dump_cpu.o ds4_kv_trailer_dump_cpu.o ds4_session_payload_dump_cpu.o ds4_first_kernel_oracle_dump_cpu.o ds4_layer0_attn_hc_pre_oracle_dump_cpu.o ds4_layer0_qkv_rope_oracle_dump_cpu.o ds4_layer0_attn_output_oracle_dump_cpu.o ds4_layer0_ffn_output_oracle_dump_cpu.o ds4_layer0_output_head_oracle_dump_cpu.o ds4_two_layer_output_head_oracle_dump_cpu.o ds4_layer2_compressor_state_oracle_dump_cpu.o ds4_layer2_attn_output_oracle_dump_cpu.o ds4_layer2_ffn_output_oracle_dump_cpu.o ds4_layer3_ffn_output_oracle_dump_cpu.o ds4_layer4_ffn_output_oracle_dump_cpu.o ds4_all_layer_final_hc_oracle_dump_cpu.o ds4_full_output_head_oracle_dump_cpu.o ds4_short_continuation_output_head_oracle_dump_cpu.o ds4_ratio_boundary_output_head_oracle_dump_cpu.o ds4_graph_checkpoint_dump_cpu.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-metadata-dump ds4_metadata_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-sampling-dump ds4_sampling_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-logits-dump ds4_logits_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-restore-dump ds4_restore_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-decode-policy-dump ds4_decode_policy_dump_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-kv-policy-dump ds4_kv_policy_dump_cpu.o ds4_kvstore.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-kvc-file-dump ds4_kvc_file_dump_cpu.o ds4_kvstore.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-kv-trailer-dump ds4_kv_trailer_dump_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-session-payload-dump ds4_session_payload_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-first-kernel-oracle-dump ds4_first_kernel_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-attn-hc-pre-oracle-dump ds4_layer0_attn_hc_pre_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-qkv-rope-oracle-dump ds4_layer0_qkv_rope_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-attn-output-oracle-dump ds4_layer0_attn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-ffn-output-oracle-dump ds4_layer0_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer0-output-head-oracle-dump ds4_layer0_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-two-layer-output-head-oracle-dump ds4_two_layer_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer2-compressor-state-oracle-dump ds4_layer2_compressor_state_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer2-attn-output-oracle-dump ds4_layer2_attn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer2-ffn-output-oracle-dump ds4_layer2_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-layer4-ffn-output-oracle-dump ds4_layer4_ffn_output_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-all-layer-final-hc-oracle-dump ds4_all_layer_final_hc_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-full-output-head-oracle-dump ds4_full_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-short-continuation-output-head-oracle-dump ds4_short_continuation_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-ratio-boundary-output-head-oracle-dump ds4_ratio_boundary_output_head_oracle_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-graph-checkpoint-dump ds4_graph_checkpoint_dump_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_kvstore.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_metadata_dump.o: ds4_metadata_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_metadata_dump.c

ds4_sampling_dump.o: ds4_sampling_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_sampling_dump.c

ds4_logits_dump.o: ds4_logits_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_logits_dump.c

ds4_restore_dump.o: ds4_restore_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_restore_dump.c

ds4_decode_policy_dump.o: ds4_decode_policy_dump.c ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ ds4_decode_policy_dump.c

ds4_kv_policy_dump.o: ds4_kv_policy_dump.c ds4.h ds4_kvstore.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kv_policy_dump.c

ds4_kvc_file_dump.o: ds4_kvc_file_dump.c ds4.h ds4_kvstore.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvc_file_dump.c

ds4_kv_trailer_dump.o: ds4_kv_trailer_dump.c ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ ds4_kv_trailer_dump.c

ds4_session_payload_dump.o: ds4_session_payload_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_session_payload_dump.c

ds4_first_kernel_oracle_dump.o: ds4_first_kernel_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_first_kernel_oracle_dump.c

ds4_layer0_attn_hc_pre_oracle_dump.o: ds4_layer0_attn_hc_pre_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer0_attn_hc_pre_oracle_dump.c

ds4_layer0_qkv_rope_oracle_dump.o: ds4_layer0_qkv_rope_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer0_qkv_rope_oracle_dump.c

ds4_layer0_attn_output_oracle_dump.o: ds4_layer0_attn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer0_attn_output_oracle_dump.c

ds4_layer0_ffn_output_oracle_dump.o: ds4_layer0_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer0_ffn_output_oracle_dump.c

ds4_layer0_output_head_oracle_dump.o: ds4_layer0_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer0_output_head_oracle_dump.c

ds4_two_layer_output_head_oracle_dump.o: ds4_two_layer_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_two_layer_output_head_oracle_dump.c

ds4_layer2_compressor_state_oracle_dump.o: ds4_layer2_compressor_state_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer2_compressor_state_oracle_dump.c

ds4_layer2_attn_output_oracle_dump.o: ds4_layer2_attn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer2_attn_output_oracle_dump.c

ds4_layer2_ffn_output_oracle_dump.o: ds4_layer2_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer2_ffn_output_oracle_dump.c

ds4_layer3_ffn_output_oracle_dump.o: ds4_layer3_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer3_ffn_output_oracle_dump.c

ds4_layer4_ffn_output_oracle_dump.o: ds4_layer4_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_layer4_ffn_output_oracle_dump.c

ds4_all_layer_final_hc_oracle_dump.o: ds4_all_layer_final_hc_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_all_layer_final_hc_oracle_dump.c

ds4_full_output_head_oracle_dump.o: ds4_full_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_full_output_head_oracle_dump.c

ds4_short_continuation_output_head_oracle_dump.o: ds4_short_continuation_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_short_continuation_output_head_oracle_dump.c

ds4_ratio_boundary_output_head_oracle_dump.o: ds4_ratio_boundary_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ratio_boundary_output_head_oracle_dump.c

ds4_graph_checkpoint_dump.o: ds4_graph_checkpoint_dump.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_graph_checkpoint_dump.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_kvstore.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metadata_dump_cpu.o: ds4_metadata_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_metadata_dump.c

ds4_sampling_dump_cpu.o: ds4_sampling_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_sampling_dump.c

ds4_logits_dump_cpu.o: ds4_logits_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_logits_dump.c

ds4_restore_dump_cpu.o: ds4_restore_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_restore_dump.c

ds4_decode_policy_dump_cpu.o: ds4_decode_policy_dump.c ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -Wno-unused-function -c -o $@ ds4_decode_policy_dump.c

ds4_kv_policy_dump_cpu.o: ds4_kv_policy_dump.c ds4.h ds4_kvstore.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_kv_policy_dump.c

ds4_kvc_file_dump_cpu.o: ds4_kvc_file_dump.c ds4.h ds4_kvstore.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_kvc_file_dump.c

ds4_kv_trailer_dump_cpu.o: ds4_kv_trailer_dump.c ds4_server.c ds4.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -Wno-unused-function -c -o $@ ds4_kv_trailer_dump.c

ds4_session_payload_dump_cpu.o: ds4_session_payload_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_session_payload_dump.c

ds4_first_kernel_oracle_dump_cpu.o: ds4_first_kernel_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_first_kernel_oracle_dump.c

ds4_layer0_attn_hc_pre_oracle_dump_cpu.o: ds4_layer0_attn_hc_pre_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer0_attn_hc_pre_oracle_dump.c

ds4_layer0_qkv_rope_oracle_dump_cpu.o: ds4_layer0_qkv_rope_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer0_qkv_rope_oracle_dump.c

ds4_layer0_attn_output_oracle_dump_cpu.o: ds4_layer0_attn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer0_attn_output_oracle_dump.c

ds4_layer0_ffn_output_oracle_dump_cpu.o: ds4_layer0_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer0_ffn_output_oracle_dump.c

ds4_layer0_output_head_oracle_dump_cpu.o: ds4_layer0_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer0_output_head_oracle_dump.c

ds4_two_layer_output_head_oracle_dump_cpu.o: ds4_two_layer_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_two_layer_output_head_oracle_dump.c

ds4_layer2_compressor_state_oracle_dump_cpu.o: ds4_layer2_compressor_state_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer2_compressor_state_oracle_dump.c

ds4_layer2_attn_output_oracle_dump_cpu.o: ds4_layer2_attn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer2_attn_output_oracle_dump.c

ds4_layer2_ffn_output_oracle_dump_cpu.o: ds4_layer2_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer2_ffn_output_oracle_dump.c

ds4_layer3_ffn_output_oracle_dump_cpu.o: ds4_layer3_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer3_ffn_output_oracle_dump.c

ds4_layer4_ffn_output_oracle_dump_cpu.o: ds4_layer4_ffn_output_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_layer4_ffn_output_oracle_dump.c

ds4_all_layer_final_hc_oracle_dump_cpu.o: ds4_all_layer_final_hc_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_all_layer_final_hc_oracle_dump.c

ds4_full_output_head_oracle_dump_cpu.o: ds4_full_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_full_output_head_oracle_dump.c

ds4_short_continuation_output_head_oracle_dump_cpu.o: ds4_short_continuation_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_short_continuation_output_head_oracle_dump.c

ds4_ratio_boundary_output_head_oracle_dump_cpu.o: ds4_ratio_boundary_output_head_oracle_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_ratio_boundary_output_head_oracle_dump.c

ds4_graph_checkpoint_dump_cpu.o: ds4_graph_checkpoint_dump.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_graph_checkpoint_dump.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test
	./ds4_test

rust-test:
	cargo test --workspace

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-metadata-dump ds4-sampling-dump ds4-logits-dump ds4-restore-dump ds4-decode-policy-dump ds4-kv-policy-dump ds4-kvc-file-dump ds4-kv-trailer-dump ds4-session-payload-dump ds4-first-kernel-oracle-dump ds4-layer0-attn-hc-pre-oracle-dump ds4-layer0-qkv-rope-oracle-dump ds4-layer0-attn-output-oracle-dump ds4-layer0-ffn-output-oracle-dump ds4-layer0-output-head-oracle-dump ds4-two-layer-output-head-oracle-dump ds4-layer2-compressor-state-oracle-dump ds4-layer2-attn-output-oracle-dump ds4-layer2-ffn-output-oracle-dump ds4-layer3-ffn-output-oracle-dump ds4-layer4-ffn-output-oracle-dump ds4-all-layer-final-hc-oracle-dump ds4-full-output-head-oracle-dump ds4-short-continuation-output-head-oracle-dump ds4-ratio-boundary-output-head-oracle-dump ds4-graph-checkpoint-dump ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
