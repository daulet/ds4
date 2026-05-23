#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model_path;
    const char *output_path;
    int token;
} dump_options;

static void usage(FILE *fp) {
    fputs("usage: ds4-layer0-attn-hc-pre-oracle-dump -m MODEL -o OUTPUT.json [--token N]\n", fp);
}

static void die(const char *msg) {
    fprintf(stderr, "ds4-layer0-attn-hc-pre-oracle-dump: %s\n", msg);
    exit(1);
}

static const char *need_arg(int *i, int argc, char **argv, const char *flag) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-layer0-attn-hc-pre-oracle-dump: missing value for %s\n", flag);
        exit(2);
    }
    return argv[++*i];
}

static int parse_nonnegative_int(const char *s) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end || v < 0 || v > 1000000L) die("invalid token id");
    return (int)v;
}

static dump_options parse_args(int argc, char **argv) {
    dump_options opt = { .token = 0 };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            opt.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) {
            opt.output_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--token")) {
            opt.token = parse_nonnegative_int(need_arg(&i, argc, argv, arg));
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model_path || !opt.output_path) {
        usage(stderr);
        exit(2);
    }
    return opt;
}

int main(int argc, char **argv) {
    dump_options opt = parse_args(argc, argv);
    FILE *fp = fopen(opt.output_path, "wb");
    if (!fp) {
        fprintf(stderr,
                "ds4-layer0-attn-hc-pre-oracle-dump: failed to open %s: %s\n",
                opt.output_path,
                strerror(errno));
        return 1;
    }
    const int rc = ds4_dump_layer0_attn_hc_pre_oracle_json(opt.model_path, opt.token, fp);
    if (fclose(fp) != 0) {
        fprintf(stderr,
                "ds4-layer0-attn-hc-pre-oracle-dump: failed to close %s: %s\n",
                opt.output_path,
                strerror(errno));
        return 1;
    }
    return rc;
}
