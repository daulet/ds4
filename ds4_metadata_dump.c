#include "ds4.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-metadata-dump [-m FILE] [--mtp FILE] [-o FILE]\n"
        "\n"
        "Emit deterministic JSON from the current C GGUF metadata loader.\n"
        "\n"
        "Options:\n"
        "  -m, --model FILE   GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE         Optional MTP support GGUF to bind and dump\n"
        "  --directory-only   Parse GGUF metadata/tensor directory without DS4 validation\n"
        "  --validate-config-only\n"
        "                     Run DS4 metadata validation without tensor binding\n"
        "  --validate-layout-only\n"
        "                     Run DS4 tensor binding/layout validation from directories only\n"
        "  -o, --output FILE  Write JSON to FILE instead of stdout\n"
        "  -h, --help         Show this help\n");
}

int main(int argc, char **argv) {
    const char *model_path = "ds4flash.gguf";
    const char *mtp_path = NULL;
    const char *output_path = NULL;
    unsigned flags = 0;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            model_path = argv[i];
        } else if (!strcmp(arg, "--mtp")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            mtp_path = argv[i];
        } else if (!strcmp(arg, "--directory-only")) {
            flags |= DS4_METADATA_DUMP_DIRECTORY_ONLY;
        } else if (!strcmp(arg, "--validate-config-only")) {
            flags |= DS4_METADATA_DUMP_VALIDATE_CONFIG_ONLY;
        } else if (!strcmp(arg, "--validate-layout-only")) {
            flags |= DS4_METADATA_DUMP_VALIDATE_LAYOUT_ONLY;
        } else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            output_path = argv[i];
        } else {
            fprintf(stderr, "ds4-metadata-dump: unknown argument: %s\n", arg);
            usage(stderr);
            return 2;
        }
    }

    FILE *fp = stdout;
    if (output_path) {
        fp = fopen(output_path, "wb");
        if (!fp) {
            fprintf(stderr, "ds4-metadata-dump: failed to open %s: %s\n",
                    output_path, strerror(errno));
            return 1;
        }
    }

    int rc = ds4_dump_metadata_json_ex(model_path, mtp_path, fp, flags);
    if (output_path && fclose(fp) != 0) {
        fprintf(stderr, "ds4-metadata-dump: failed to close %s: %s\n",
                output_path, strerror(errno));
        return 1;
    }
    return rc;
}
