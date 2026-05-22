#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-sampling-dump [OUTPUT]\n"
        "\n"
        "Emit deterministic JSON from the current C fixed-logits sampler and logprob math.\n"
        "\n"
        "Arguments:\n"
        "  OUTPUT   Optional output file. Defaults to stdout.\n"
        "  -h, --help\n"
        "           Show this help\n");
}

int main(int argc, char **argv) {
    const char *output_path = NULL;
    if (argc > 2) {
        usage(stderr);
        return 2;
    }
    if (argc == 2) {
        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
            usage(stdout);
            return 0;
        }
        output_path = argv[1];
    }

    FILE *fp = stdout;
    if (output_path) {
        fp = fopen(output_path, "wb");
        if (!fp) {
            fprintf(stderr, "ds4-sampling-dump: failed to open %s: %s\n",
                    output_path, strerror(errno));
            return 1;
        }
    }

    int rc = ds4_dump_sampling_oracle_json(fp);
    if (output_path && fclose(fp) != 0) {
        fprintf(stderr, "ds4-sampling-dump: failed to close %s: %s\n",
                output_path, strerror(errno));
        return 1;
    }
    return rc;
}
