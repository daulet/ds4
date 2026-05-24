#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s --model FILE [--output FILE]\n", prog);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *output = NULL;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model = argv[++i];
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output = argv[++i];
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

    int rc = ds4_dump_ratio_boundary_output_head_oracle_json(model, fp);
    if (output && fclose(fp) != 0 && rc == 0) {
        fprintf(stderr, "%s: %s: %s\n", argv[0], output, strerror(errno));
        rc = 1;
    }
    return rc;
}
