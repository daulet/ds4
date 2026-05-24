#include "ds4.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s --model FILE --prompt FILE [--limit-tokens N] [--backend cuda|metal] [--output FILE]\n", prog);
}

static ds4_backend parse_backend(const char *s) {
    if (strcmp(s, "metal") == 0) return DS4_BACKEND_METAL;
    if (strcmp(s, "cuda") == 0) return DS4_BACKEND_CUDA;
    if (strcmp(s, "cpu") == 0) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-prefill-whole-short-oracle-dump: unknown backend: %s\n", s);
    exit(2);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *prompt = NULL;
    const char *output = NULL;
    int limit_tokens = 0;
    ds4_backend backend = DS4_BACKEND_METAL;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--limit-tokens") == 0 && i + 1 < argc) {
            char *end = NULL;
            long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end || v <= 0 || v > 1000000L) {
                usage(argv[0]);
                return 2;
            }
            limit_tokens = (int)v;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = parse_backend(argv[++i]);
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!model || !prompt) {
        usage(argv[0]);
        return 2;
    }

    FILE *fp = stdout;
    if (output) {
        fp = fopen(output, "wb");
        if (!fp) {
            fprintf(stderr, "%s: %s: %s\n", argv[0], output, strerror(errno));
            return 1;
        }
    }

    int rc = ds4_dump_prefill_whole_short_oracle_json(model, prompt, limit_tokens, backend, fp);
    if (output && fclose(fp) != 0 && rc == 0) {
        fprintf(stderr, "%s: %s: %s\n", argv[0], output, strerror(errno));
        rc = 1;
    }
    return rc;
}
