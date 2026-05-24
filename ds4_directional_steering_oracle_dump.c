#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s --model FILE --dir-steering-file FILE "
            "--dir-steering-attn F --dir-steering-ffn F [--token ID] [--output FILE]\n",
            prog);
}

static int parse_float_arg(const char *arg, const char *value, float *out) {
    char *end = NULL;
    errno = 0;
    float v = strtof(value, &end);
    if (errno || !end || *end != '\0') {
        fprintf(stderr, "%s: invalid float: %s\n", arg, value);
        return 0;
    }
    *out = v;
    return 1;
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *steering_file = NULL;
    const char *output = NULL;
    float steering_attn = 0.0f;
    float steering_ffn = 0.0f;
    int token = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
        } else if (strcmp(argv[i], "--dir-steering-file") == 0 && i + 1 < argc) {
            steering_file = argv[++i];
        } else if (strcmp(argv[i], "--dir-steering-attn") == 0 && i + 1 < argc) {
            const char *arg = argv[i];
            const char *value = argv[++i];
            if (!parse_float_arg(arg, value, &steering_attn)) return 2;
        } else if (strcmp(argv[i], "--dir-steering-ffn") == 0 && i + 1 < argc) {
            const char *arg = argv[i];
            const char *value = argv[++i];
            if (!parse_float_arg(arg, value, &steering_ffn)) return 2;
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--token") == 0 && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            long v = strtol(argv[++i], &end, 10);
            if (errno || !end || *end != '\0' || v < 0 || v > 2147483647L) {
                usage(argv[0]);
                return 2;
            }
            token = (int)v;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!model || !steering_file || (steering_attn == 0.0f && steering_ffn == 0.0f)) {
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

    int rc = ds4_dump_directional_steering_decode_oracle_json(
            model, token, steering_file, steering_attn, steering_ffn, fp);
    if (output && fclose(fp) != 0 && rc == 0) {
        fprintf(stderr, "%s: %s: %s\n", argv[0], output, strerror(errno));
        rc = 1;
    }
    return rc;
}
