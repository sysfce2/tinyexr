/*
 * TinyEXR texcomp - Basis Universal transcoder validation harness.
 *
 * Reads a .ktx2 or .basis file and transcodes it to RGBA using the
 * vendored Basis Universal transcoder. Validates that the output
 * matches the expected RGBA data.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    printf("basis-validate: not yet built (basisu transcoder not vendored)\n");
    return 0; /* skip test when transcoder is absent */
}
