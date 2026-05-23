#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s --model FILE [--token ID] [--output FILE]\n", prog);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *output = NULL;
    int token = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
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

    if (!model) {
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

    int rc = ds4_dump_all_layer_final_hc_oracle_json(model, token, fp);
    if (output && fclose(fp) != 0 && rc == 0) {
        fprintf(stderr, "%s: %s: %s\n", argv[0], output, strerror(errno));
        rc = 1;
    }
    return rc;
}
