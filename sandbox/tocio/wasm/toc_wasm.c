/*
 * tocio - WebAssembly shim. Flat C exports for a JS host (e.g. the tinyexr web
 * viewer): parse a config, build a processor, apply it to a float buffer on the
 * CPU, or emit a GLSL shader + LUT texture metadata to render on the GPU.
 *
 * FileTransform LUTs require a JS-injected reader (no filesystem in WASM); the
 * smoke configs here use matrix/builtin transforms only.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h" /* toc_malloc/toc_free for the JS-facing string dup */

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define TOC_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define TOC_EXPORT
#endif

TOC_EXPORT void *tocw_parse(const char *yaml, int len) {
    toc_config *cfg = NULL;
    if (!TOC_OK(toc_config_parse(yaml, (size_t)len, NULL, &cfg))) return NULL;
    return cfg;
}
TOC_EXPORT void tocw_free_config(void *cfg) {
    toc_config_free((toc_config *)cfg);
}

TOC_EXPORT void *tocw_processor(void *cfg, const char *src, const char *dst) {
    toc_op_list *ops = NULL;
    if (!TOC_OK(toc_processor_from_colorspaces((toc_config *)cfg, src, dst, NULL,
                                               &ops)))
        return NULL;
    return ops;
}
TOC_EXPORT void *tocw_processor_view(void *cfg, const char *src,
                                     const char *display, const char *view) {
    toc_op_list *ops = NULL;
    if (!TOC_OK(toc_processor_from_display_view((toc_config *)cfg, src, display,
                                                view, NULL, &ops)))
        return NULL;
    return ops;
}
TOC_EXPORT void tocw_free_ops(void *ops) {
    toc_op_list_free((toc_op_list *)ops);
}

/* Apply in place to interleaved RGBA float (channels 3 or 4). Returns 0 on ok. */
TOC_EXPORT int tocw_apply(void *ops, float *rgba, int npix, int channels) {
    return TOC_OK(toc_apply((toc_op_list *)ops, rgba, (size_t)npix, channels))
               ? 0
               : -1;
}

/* Emit a GLSL shader. target: 0=ES3.0, 1=GLSL330, 2=Vulkan450. Returns a
 * malloc'd NUL-terminated string the caller frees with tocw_free_str, or NULL. */
TOC_EXPORT char *tocw_emit_glsl(void *ops, int target) {
    toc_shader sh;
    char *out;
    size_t n;
    if (!TOC_OK(toc_emit_glsl((toc_op_list *)ops, (toc_glsl_target)target, NULL,
                              &sh)))
        return NULL;
    if (!sh.source) return NULL;
    /* duplicate into a plain malloc buffer so JS frees it simply */
    {
        const char *s = sh.source;
        n = 0;
        while (s[n]) ++n;
        out = (char *)toc_malloc(toc_default_allocator(), n + 1);
        if (out) {
            size_t i;
            for (i = 0; i <= n; ++i) out[i] = s[i];
        }
    }
    toc_shader_free(&sh);
    return out;
}

/* Emit fused C source. Returns malloc'd string or NULL. */
TOC_EXPORT char *tocw_emit_c(void *ops) {
    char *src = NULL;
    size_t len = 0;
    if (!TOC_OK(toc_emit_c((toc_op_list *)ops, NULL, NULL, &src, &len)))
        return NULL;
    return src;
}

TOC_EXPORT void tocw_free_str(char *p) {
    toc_free(toc_default_allocator(), p);
}

TOC_EXPORT int tocw_num_colorspaces(void *cfg) {
    return toc_config_num_colorspaces((toc_config *)cfg);
}
TOC_EXPORT const char *tocw_colorspace_name(void *cfg, int i) {
    return toc_config_colorspace_name((toc_config *)cfg, i);
}
