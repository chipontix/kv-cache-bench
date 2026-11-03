BUILD ?= build
LABEL ?= local

.PHONY: help build test bench bench-compare profile mutation mutation-boundary clean
help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

build: ## configure and build (Release)
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD) -j

test: build ## CPU tests, GPU correctness tests, length boundary tests (exit 77 = GPU part skipped: no Metal device)
	$(BUILD)/cpu_tests
	$(BUILD)/gpu_tests || [ $$? -eq 77 ]
	$(BUILD)/boundary_tests || [ $$? -eq 77 ]

bench: build ## full sweep on this machine, writes results/$(LABEL)/
	./scripts/bench.sh $(LABEL)

bench-compare: ## our kernel vs llama.cpp and MLX (both built/installed into build-cmp/), writes results/$(LABEL)/compare.csv
	./scripts/compare.sh $(LABEL)

profile: build ## xctrace profile of the paged kernel, f16 vs i8: per-kernel GPU time and available GPU counters, writes results/int8-profile/
	./scripts/profile.sh

mutation: build ## inject known bugs into the kernel and check that gpu_tests catches each one
	./scripts/mutation.sh

mutation-boundary: build ## inject bugs at the length boundaries and check that boundary_tests catches each one
	./scripts/mutation.sh boundary

clean: ## remove the build trees
	rm -rf $(BUILD) build-cmp
