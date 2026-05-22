#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "ds4_server.c"

#include <errno.h>
#include <locale.h>

typedef struct {
    const char *id;
    const char *dsml;
} trailer_memory_entry;

typedef struct {
    const char *name;
    const char *text;
    const trailer_memory_entry *entries;
    int entry_count;
    bool disabled;
    const char * const *wanted_ids;
    int wanted_count;
} trailer_case;

static void trailer_json_string(FILE *fp, const char *s) {
    buf b = {0};
    json_escape(&b, s ? s : "");
    fputs(b.ptr ? b.ptr : "\"\"", fp);
    buf_free(&b);
}

static void trailer_json_string_n(FILE *fp, const char *s, size_t len) {
    buf b = {0};
    json_escape_n(&b, s ? s : "", s ? len : 0);
    fputs(b.ptr ? b.ptr : "\"\"", fp);
    buf_free(&b);
}

static void trailer_hex_file(FILE *fp, const void *ptr, size_t len) {
    static const char hex[] = "0123456789abcdef";
    const uint8_t *p = ptr;
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        fputc(hex[p[i] >> 4], fp);
        fputc(hex[p[i] & 15], fp);
    }
    fputc('"', fp);
}

static void trailer_put32(uint8_t *p, uint32_t v) {
    ds4_kvstore_le_put32(p, v);
}

static bool capture_tool_map(server *s, const char *text,
                             uint8_t **bytes_out, size_t *len_out,
                             uint64_t *estimated_out,
                             uint64_t *written_out,
                             bool *write_ok_out) {
    *bytes_out = NULL;
    *len_out = 0;
    *estimated_out = 0;
    *written_out = 0;
    *write_ok_out = false;
    if (!kv_tool_map_serialized_size(s, text, estimated_out)) return false;
    FILE *fp = tmpfile();
    if (!fp) return false;
    bool ok = kv_tool_map_write(s, fp, text, written_out) && fflush(fp) == 0;
    *write_ok_out = ok;
    if (fseeko(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return false;
    }
    off_t end = ftello(fp);
    if (end < 0 || fseeko(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return false;
    }
    uint8_t *bytes = xmalloc((size_t)end + 1);
    if (fread(bytes, 1, (size_t)end, fp) != (size_t)end) {
        free(bytes);
        fclose(fp);
        return false;
    }
    fclose(fp);
    *bytes_out = bytes;
    *len_out = (size_t)end;
    return true;
}

static int load_count_from_bytes(const uint8_t *bytes, size_t len,
                                 const char * const *wanted_ids,
                                 int wanted_count) {
    server dst = {0};
    pthread_mutex_init(&dst.tool_mu, NULL);
    stop_list wanted = {0};
    const stop_list *wanted_ptr = NULL;
    for (int i = 0; i < wanted_count; i++) {
        id_list_push_unique(&wanted, wanted_ids[i]);
    }
    if (wanted.len > 0) wanted_ptr = &wanted;
    FILE *fp = tmpfile();
    if (!fp) die("tmpfile failed");
    if (fwrite(bytes, 1, len, fp) != len || fflush(fp) != 0 ||
        fseeko(fp, 0, SEEK_SET) != 0)
    {
        die("failed to prepare trailer load fixture");
    }
    int loaded = kv_tool_map_load_from_pos(&dst, fp, wanted_ptr);
    fclose(fp);
    id_list_free(&wanted);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&dst.tool_mu);
    return loaded;
}

static const char *decode_error_name(const uint8_t *bytes, size_t len,
                                     int max_entries,
                                     uint32_t *count_out,
                                     int *decoded_out) {
    *count_out = 0;
    *decoded_out = 0;
    if (len == 0) return NULL;
    if (len < KV_TOOL_MAP_HEADER) return "short-header";
    if (bytes[0] != KV_TOOL_MAP_MAGIC0 || bytes[1] != KV_TOOL_MAP_MAGIC1 ||
        bytes[2] != KV_TOOL_MAP_MAGIC2 || bytes[3] != KV_TOOL_MAP_VERSION)
        return "bad-header";
    uint32_t count = le_get32(bytes + 4);
    *count_out = count;
    if ((uint64_t)count > (uint64_t)max_entries * 4u) return "count-limit";
    size_t pos = KV_TOOL_MAP_HEADER;
    for (uint32_t i = 0; i < count; i++) {
        if (len - pos < 8) return "short-entry-header";
        uint32_t id_len = le_get32(bytes + pos);
        uint32_t dsml_len = le_get32(bytes + pos + 4);
        pos += 8;
        if (id_len == 0 || id_len > 256) return "bad-id-len";
        if (dsml_len == 0 || dsml_len > DS4_TOOL_MEMORY_MAX_BYTES) return "bad-dsml-len";
        if (len - pos < id_len) return "truncated-id";
        pos += id_len;
        if (len - pos < dsml_len) return "truncated-dsml";
        pos += dsml_len;
        (*decoded_out)++;
    }
    return NULL;
}

static void write_decoded_entries(FILE *fp, const uint8_t *bytes, size_t len) {
    if (len == 0 || len < KV_TOOL_MAP_HEADER) {
        fputs("[]", fp);
        return;
    }
    uint32_t count = le_get32(bytes + 4);
    size_t pos = KV_TOOL_MAP_HEADER;
    fputc('[', fp);
    bool first = true;
    for (uint32_t i = 0; i < count; i++) {
        if (len - pos < 8) break;
        uint32_t id_len = le_get32(bytes + pos);
        uint32_t dsml_len = le_get32(bytes + pos + 4);
        pos += 8;
        if (id_len == 0 || id_len > 256 || dsml_len == 0 ||
            dsml_len > DS4_TOOL_MEMORY_MAX_BYTES || len - pos < id_len)
            break;
        const uint8_t *id = bytes + pos;
        pos += id_len;
        if (len - pos < dsml_len) break;
        const uint8_t *dsml = bytes + pos;
        pos += dsml_len;
        if (!first) fputc(',', fp);
        first = false;
        fputs("{\"id\":", fp);
        trailer_json_string_n(fp, (const char *)id, id_len);
        fputs(",\"dsml_hex\":", fp);
        trailer_hex_file(fp, dsml, dsml_len);
        fputc('}', fp);
    }
    fputc(']', fp);
}

static void write_decode_result(FILE *fp, const uint8_t *bytes, size_t len) {
    uint32_t count = 0;
    int decoded = 0;
    const char *error = decode_error_name(bytes, len, DS4_TOOL_MEMORY_DEFAULT_MAX_IDS,
                                          &count, &decoded);
    fprintf(fp, "{\"ok\":%s,\"count\":%u,\"decoded_entries\":%d",
            error ? "false" : "true", count, decoded);
    if (error) {
        fputs(",\"error\":", fp);
        trailer_json_string(fp, error);
    }
    fputs(",\"entries\":", fp);
    write_decoded_entries(fp, bytes, len);
    fputc('}', fp);
}

static void write_memory_entries(FILE *fp, const trailer_memory_entry *entries, int count) {
    fputc('[', fp);
    for (int i = 0; i < count; i++) {
        if (i) fputc(',', fp);
        fputs("{\"id\":", fp);
        trailer_json_string(fp, entries[i].id);
        fputs(",\"dsml_hex\":", fp);
        trailer_hex_file(fp, entries[i].dsml, strlen(entries[i].dsml));
        fputc('}', fp);
    }
    fputc(']', fp);
}

static void write_wanted_ids(FILE *fp, const char * const *ids, int count) {
    fputc('[', fp);
    for (int i = 0; i < count; i++) {
        if (i) fputc(',', fp);
        trailer_json_string(fp, ids[i]);
    }
    fputc(']', fp);
}

static void write_case(FILE *fp, const trailer_case *tc, bool first) {
    if (!first) fputs(",\n", fp);
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.disable_exact_dsml_tool_replay = tc->disabled;
    for (int i = 0; i < tc->entry_count; i++) {
        tool_memory_put(&s, tc->entries[i].id, tc->entries[i].dsml);
    }
    uint8_t *bytes = NULL;
    size_t len = 0;
    uint64_t estimated = 0;
    uint64_t written = 0;
    bool write_ok = false;
    bool captured = capture_tool_map(&s, tc->text, &bytes, &len,
                                     &estimated, &written, &write_ok);
    if (!captured) die("failed to capture tool map");
    int loaded_all = load_count_from_bytes(bytes, len, NULL, 0);
    int loaded_wanted = load_count_from_bytes(bytes, len,
                                             tc->wanted_ids, tc->wanted_count);

    fputs("    {\"name\":", fp);
    trailer_json_string(fp, tc->name);
    fprintf(fp, ",\"disabled\":%s,\"text_hex\":", tc->disabled ? "true" : "false");
    trailer_hex_file(fp, tc->text, strlen(tc->text));
    fputs(",\"memory_entries\":", fp);
    write_memory_entries(fp, tc->entries, tc->entry_count);
    fputs(",\"wanted_ids\":", fp);
    write_wanted_ids(fp, tc->wanted_ids, tc->wanted_count);
    fprintf(fp,
            ",\"serialized_size\":%llu,\"write_ok\":%s,"
            "\"written_bytes\":%llu,\"trailer_hex\":",
            (unsigned long long)estimated,
            write_ok ? "true" : "false",
            (unsigned long long)written);
    trailer_hex_file(fp, bytes, len);
    fprintf(fp, ",\"trailer_bytes\":%llu,\"load_all_count\":%d,"
            "\"load_wanted_count\":%d,\"decoded\":",
            (unsigned long long)len, loaded_all, loaded_wanted);
    write_decode_result(fp, bytes, len);
    fputc('}', fp);

    free(bytes);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static const char DSML_BASH[] =
    "\n\n<tool_calls>\n"
    "<invoke name=\"bash\">\n"
    "<parameter name=\"command\" string=\"true\">pwd</parameter>\n"
    "</invoke>\n"
    "</tool_calls>";
static const char DSML_EDIT[] =
    "\n\n<tool_calls>\n"
    "<invoke name=\"edit\">\n"
    "<parameter name=\"patch\" string=\"true\">alpha beta</parameter>\n"
    "</invoke>\n"
    "</tool_calls>";
static const char DSML_UTF8[] =
    "\n\n<tool_calls>\n"
    "<invoke name=\"note\">\n"
    "<parameter name=\"text\" string=\"true\">caf\xc3\xa9 \\xe2\\x82\\xac</parameter>\n"
    "</invoke>\n"
    "</tool_calls>";

static void write_tool_map_cases(FILE *fp) {
    const trailer_memory_entry none_entries[] = {
        {"call_unused", DSML_BASH},
    };
    const trailer_memory_entry one_entries[] = {
        {"call_keep", DSML_BASH},
    };
    const trailer_memory_entry filter_entries[] = {
        {"call_keep", DSML_BASH},
        {"call_drop", DSML_EDIT},
    };
    const trailer_memory_entry same_block_entries[] = {
        {"call_a", DSML_BASH},
        {"call_b", DSML_BASH},
    };
    const trailer_memory_entry utf8_entries[] = {
        {"call_utf8_long_identifier_012345678901234567890123456789", DSML_UTF8},
    };
    const char *wanted_keep[] = {"call_keep"};
    const char *wanted_b[] = {"call_b"};
    const trailer_case cases[] = {
        {"empty_text", "", none_entries, 1, false, NULL, 0},
        {"single_block", DSML_BASH, one_entries, 1, false, NULL, 0},
        {"filters_by_text", DSML_BASH, filter_entries, 2, false, wanted_keep, 1},
        {"duplicate_block_once", "\n\n<tool_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\" string=\"true\">pwd</parameter>\n</invoke>\n</tool_calls> suffix \n\n<tool_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\" string=\"true\">pwd</parameter>\n</invoke>\n</tool_calls>", one_entries, 1, false, NULL, 0},
        {"multiple_ids_same_block", DSML_BASH, same_block_entries, 2, false, wanted_b, 1},
        {"utf8_and_long_id", DSML_UTF8, utf8_entries, 1, false, NULL, 0},
        {"disabled_replay", DSML_BASH, one_entries, 1, true, NULL, 0},
    };
    fputs("  \"tool_map_cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        write_case(fp, &cases[i], i == 0);
    }
    fputs("\n  ],\n", fp);
}

static void write_extension_flag_cases(FILE *fp) {
    struct flag_case {
        const char *name;
        uint8_t ext_flags;
        uint64_t trailer_bytes;
    } cases[] = {
        {"token_text", 0, 0},
        {"tool_map_payload", KV_EXT_TOOL_MAP, 16},
        {"responses_visible_no_payload", KV_EXT_RESPONSES_VISIBLE, 0},
        {"thinking_visible_no_payload", KV_EXT_THINKING_VISIBLE, 0},
        {"responses_priority", KV_EXT_TOOL_MAP | KV_EXT_RESPONSES_VISIBLE | KV_EXT_THINKING_VISIBLE, 0},
    };
    fputs("  \"extension_flag_cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (i) fputs(",\n", fp);
        fputs("    {\"name\":", fp);
        trailer_json_string(fp, cases[i].name);
        fprintf(fp, ",\"ext_flags\":%u,\"tool_map_present\":%s,"
                "\"trailer_bytes\":%llu,\"key_kind\":",
                cases[i].ext_flags,
                (cases[i].ext_flags & KV_EXT_TOOL_MAP) ? "true" : "false",
                (unsigned long long)cases[i].trailer_bytes);
        trailer_json_string(fp, ds4_kvstore_key_kind(cases[i].ext_flags));
        fputc('}', fp);
    }
    fputs("\n  ],\n", fp);
}

static uint8_t *make_entry_bytes(const char *id, const char *dsml, size_t *len_out) {
    size_t id_len = strlen(id);
    size_t dsml_len = strlen(dsml);
    size_t len = 8 + id_len + dsml_len;
    uint8_t *bytes = xmalloc(len);
    trailer_put32(bytes, (uint32_t)id_len);
    trailer_put32(bytes + 4, (uint32_t)dsml_len);
    memcpy(bytes + 8, id, id_len);
    memcpy(bytes + 8 + id_len, dsml, dsml_len);
    *len_out = len;
    return bytes;
}

static void write_malformed_case(FILE *fp, const char *name,
                                 const char *expected_error,
                                 const uint8_t *bytes, size_t len,
                                 bool first) {
    if (!first) fputs(",\n", fp);
    int loaded = load_count_from_bytes(bytes, len, NULL, 0);
    fputs("    {\"name\":", fp);
    trailer_json_string(fp, name);
    fputs(",\"expected_error\":", fp);
    trailer_json_string(fp, expected_error);
    fputs(",\"trailer_hex\":", fp);
    trailer_hex_file(fp, bytes, len);
    fprintf(fp, ",\"load_count\":%d,\"decoded\":", loaded);
    write_decode_result(fp, bytes, len);
    fputc('}', fp);
}

static void write_malformed_cases(FILE *fp) {
    uint8_t short_header[] = {'K', 'T'};
    uint8_t bad_magic[] = {'X', 'T', 'M', 1, 0, 0, 0, 0};
    uint8_t bad_version[] = {'K', 'T', 'M', 2, 0, 0, 0, 0};
    uint8_t count_limit[] = {'K', 'T', 'M', 1, 0, 0, 0, 0};
    trailer_put32(count_limit + 4, DS4_TOOL_MEMORY_DEFAULT_MAX_IDS * 4u + 1u);
    uint8_t zero_id[] = {'K', 'T', 'M', 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 'x'};
    uint8_t long_id[] = {'K', 'T', 'M', 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 'x'};
    uint8_t zero_dsml[] = {'K', 'T', 'M', 1, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 'c', 'a', 'l', 'l'};
    uint8_t truncated_id[] = {'K', 'T', 'M', 1, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 'c', 'a'};
    uint8_t truncated_dsml[] = {'K', 'T', 'M', 1, 1, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 'c', 'a', 'l', 'l', 'x', 'y'};
    size_t first_len = 0;
    uint8_t *first = make_entry_bytes("call_a", DSML_BASH, &first_len);
    size_t partial_len = 8 + first_len + 8 + 2;
    uint8_t *partial_second = xmalloc(partial_len);
    memcpy(partial_second, "KTM\1", 4);
    trailer_put32(partial_second + 4, 2);
    memcpy(partial_second + 8, first, first_len);
    trailer_put32(partial_second + 8 + first_len, 4);
    trailer_put32(partial_second + 8 + first_len + 4, 5);
    memcpy(partial_second + 8 + first_len + 8, "ca", 2);

    fputs("  \"malformed_cases\": [\n", fp);
    write_malformed_case(fp, "short_header", "short-header", short_header, sizeof(short_header), true);
    write_malformed_case(fp, "bad_magic", "bad-header", bad_magic, sizeof(bad_magic), false);
    write_malformed_case(fp, "bad_version", "bad-header", bad_version, sizeof(bad_version), false);
    write_malformed_case(fp, "count_limit", "count-limit", count_limit, sizeof(count_limit), false);
    write_malformed_case(fp, "zero_id_len", "bad-id-len", zero_id, sizeof(zero_id), false);
    write_malformed_case(fp, "id_len_too_long", "bad-id-len", long_id, sizeof(long_id), false);
    write_malformed_case(fp, "zero_dsml_len", "bad-dsml-len", zero_dsml, sizeof(zero_dsml), false);
    write_malformed_case(fp, "truncated_id", "truncated-id", truncated_id, sizeof(truncated_id), false);
    write_malformed_case(fp, "truncated_dsml", "truncated-dsml", truncated_dsml, sizeof(truncated_dsml), false);
    write_malformed_case(fp, "partial_second_entry", "truncated-id", partial_second, partial_len, false);
    fputs("\n  ]\n", fp);
    free(first);
    free(partial_second);
}

static int kv_trailer_dump_json(FILE *fp) {
    fputs("{\n", fp);
    fputs("  \"schema\": \"ds4.kv_trailer_oracle.v1\",\n", fp);
    fputs("  \"source\": \"current-c-server-kv-trailer-no-model\",\n", fp);
    fputs("  \"model\": \"no model is loaded for this oracle\",\n", fp);
    fprintf(fp,
            "  \"constants\": {\"tool_map_magic_hex\":\"4b544d\","
            "\"tool_map_version\":%u,\"tool_map_header\":%u,"
            "\"max_id_len\":256,\"ext_flags\":{\"tool_map\":%u,"
            "\"responses_visible\":%u,\"thinking_visible\":%u}},\n",
            KV_TOOL_MAP_VERSION,
            KV_TOOL_MAP_HEADER,
            KV_EXT_TOOL_MAP,
            KV_EXT_RESPONSES_VISIBLE,
            KV_EXT_THINKING_VISIBLE);
    write_tool_map_cases(fp);
    write_extension_flag_cases(fp);
    write_malformed_cases(fp);
    fputs("}\n", fp);
    return ferror(fp) ? 1 : 0;
}

static void trailer_usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-kv-trailer-dump [OUTPUT]\n"
            "\n"
            "Emit deterministic no-model JSON for server-owned KVC trailer payloads.\n"
            "\n"
            "Arguments:\n"
            "  OUTPUT   Optional output file. Defaults to stdout.\n"
            "  -h, --help\n"
            "           Show this help\n");
}

int main(int argc, char **argv) {
    setlocale(LC_NUMERIC, "C");
    const char *output_path = NULL;
    if (argc > 2) {
        trailer_usage(stderr);
        return 2;
    }
    if (argc == 2) {
        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
            trailer_usage(stdout);
            return 0;
        }
        output_path = argv[1];
    }
    FILE *fp = stdout;
    if (output_path) {
        fp = fopen(output_path, "wb");
        if (!fp) {
            fprintf(stderr, "ds4-kv-trailer-dump: failed to open %s: %s\n",
                    output_path, strerror(errno));
            return 1;
        }
    }
    int rc = kv_trailer_dump_json(fp);
    if (output_path && fclose(fp) != 0) {
        fprintf(stderr, "ds4-kv-trailer-dump: failed to close %s: %s\n",
                output_path, strerror(errno));
        return 1;
    }
    return rc;
}
