/*
 * TinyEXR - CPU feature detection + SIMD dispatch table.
 *
 * Phase 1: scalar-only reporting. Runtime CPUID detection and the dispatch
 * vtable are filled in by Phase 7.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

uint32_t exr_simd_capabilities(void) { return EXR_SIMD_NONE; }

const char *exr_simd_info(void) { return "scalar"; }
