#include "ds4.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ds4_graph_checkpoint_options checkpoint;
    const char *output_path;
} dump_options;

static void usage(FILE *fp) {
    fputs("usage: ds4-graph-checkpoint-dump -m MODEL -o OUTPUT.json --model-sha256 HEX --short-prompt FILE --long-prompt FILE [--mtp FILE] [--ctx N] [--backend cuda|metal]\n", fp);
}

static void die(const char *msg) {
    fprintf(stderr, "ds4-graph-checkpoint-dump: %s\n", msg);
    exit(1);
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    die("unknown backend");
    return DS4_BACKEND_CPU;
}

static bool is_sha256_hex(const char *s) {
    if (!s || strlen(s) != 64) return false;
    for (const char *p = s; *p; p++) {
        if (!(*p >= '0' && *p <= '9') && !(*p >= 'a' && *p <= 'f')) return false;
    }
    return true;
}

static int parse_positive_int(const char *s) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end || v <= 0 || v > 1000000L) die("invalid positive integer");
    return (int)v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *flag) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-graph-checkpoint-dump: missing value for %s\n", flag);
        exit(2);
    }
    return argv[++*i];
}

static dump_options parse_args(int argc, char **argv) {
    dump_options opt = {
        .checkpoint = {
            .backend = DS4_BACKEND_METAL,
            .ctx_size = 32768,
        },
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            opt.checkpoint.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) {
            opt.output_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--model-sha256")) {
            opt.checkpoint.model_sha256 = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--short-prompt")) {
            opt.checkpoint.short_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--long-prompt")) {
            opt.checkpoint.long_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            opt.checkpoint.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx")) {
            opt.checkpoint.ctx_size = parse_positive_int(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--backend")) {
            opt.checkpoint.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.checkpoint.model_path ||
        !opt.output_path ||
        !opt.checkpoint.model_sha256 ||
        !opt.checkpoint.short_prompt_path ||
        !opt.checkpoint.long_prompt_path)
    {
        usage(stderr);
        exit(2);
    }
    if (!is_sha256_hex(opt.checkpoint.model_sha256)) die("model sha256 must be 64 lowercase hex characters");
    return opt;
}

int main(int argc, char **argv) {
    dump_options opt = parse_args(argc, argv);
    FILE *fp = fopen(opt.output_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-graph-checkpoint-dump: failed to open %s: %s\n",
                opt.output_path,
                strerror(errno));
        return 1;
    }
    const int rc = ds4_dump_graph_checkpoint_oracle_json(&opt.checkpoint, fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-graph-checkpoint-dump: failed to close %s: %s\n",
                opt.output_path,
                strerror(errno));
        return 1;
    }
    return rc;
}
