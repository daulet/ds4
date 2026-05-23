#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model_path;
    const char *output_path;
    int         token;
} opts;

static void usage(FILE *fp) {
    fputs("usage: ds4-layer2-compressor-state-oracle-dump -m MODEL -o OUTPUT.json [--token N]\n", fp);
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < 0 || v > 2147483647L) return 0;
    *out = (int)v;
    return 1;
}

static opts parse_opts(int argc, char **argv) {
    opts opt = {0};
    opt.token = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if ((!strcmp(arg, "-m") || !strcmp(arg, "--model")) && i + 1 < argc) {
            opt.model_path = argv[++i];
        } else if ((!strcmp(arg, "-o") || !strcmp(arg, "--output")) && i + 1 < argc) {
            opt.output_path = argv[++i];
        } else if (!strcmp(arg, "--token") && i + 1 < argc) {
            if (!parse_int(argv[++i], &opt.token)) {
                usage(stderr);
                exit(2);
            }
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
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
    opts opt = parse_opts(argc, argv);
    FILE *fp = fopen(opt.output_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-layer2-compressor-state-oracle-dump: open '%s': %s\n", opt.output_path, strerror(errno));
        return 1;
    }
    const int rc = ds4_dump_layer2_compressor_state_oracle_json(opt.model_path, opt.token, fp);
    if (fclose(fp) != 0 && rc == 0) {
        fprintf(stderr, "ds4-layer2-compressor-state-oracle-dump: close '%s': %s\n", opt.output_path, strerror(errno));
        return 1;
    }
    return rc;
}
