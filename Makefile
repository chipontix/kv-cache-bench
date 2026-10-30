BUILD ?= build
LABEL ?= local

.PHONY: help build test bench mutation clean
help: ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

build: ## configure and build (Release)
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD) -j

test: build ## CPU tests, then GPU correctness tests (exit 77 = skipped: no Metal device)
	$(BUILD)/cpu_tests
	$(BUILD)/gpu_tests || [ $$? -eq 77 ]

bench: build ## full sweep on this machine, writes results/$(LABEL)/
	./scripts/bench.sh $(LABEL)

mutation: build ## inject known bugs into the kernel and check that gpu_tests catches each one
	./scripts/mutation.sh

clean: ## remove the build tree
	rm -rf $(BUILD)
