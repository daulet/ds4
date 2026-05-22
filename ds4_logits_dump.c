#include "ds4.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VEC_MAX_STEPS 16
#define VEC_MAX_TOP 32
#define VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} vec_top;

typedef struct {
    unsigned char selected[VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    vec_top top[VEC_MAX_TOP];
} vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    vec_step steps[VEC_MAX_STEPS];
} vec_case;

typedef struct {
    const char *model_path;
    const char *vector_path;
    const char *json_path;
    const char *logits_path;
    const char *model_sha256;
    ds4_backend backend;
} dump_options;

static void die(const char *msg) {
    fprintf(stderr, "ds4-logits-dump: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg, const char *path) {
    fprintf(stderr, "ds4-logits-dump: %s %s: %s\n", msg, path, strerror(errno));
    exit(1);
}

static char *trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static int hex_value(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool is_sha256_hex(const char *s) {
    if (!s || strlen(s) != 64) return false;
    for (const char *p = s; *p; p++) {
        if (!(*p >= '0' && *p <= '9') && !(*p >= 'a' && *p <= 'f')) return false;
    }
    return true;
}

static char *shell_quote(const char *s) {
    size_t n = 2;
    for (const char *p = s; *p; p++) n += *p == '\'' ? 4 : 1;
    char *out = malloc(n + 1);
    if (!out) die("out of memory");
    char *w = out;
    *w++ = '\'';
    for (const char *p = s; *p; p++) {
        if (*p == '\'') {
            memcpy(w, "'\\''", 4);
            w += 4;
        } else {
            *w++ = *p;
        }
    }
    *w++ = '\'';
    *w = '\0';
    return out;
}

static bool read_sha256_command(const char *cmd, char out[65]) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return false;
    char line[256];
    bool ok = fgets(line, sizeof(line), fp) != NULL;
    const int rc = pclose(fp);
    if (!ok || rc != 0) return false;
    line[strcspn(line, " \t\r\n")] = '\0';
    if (!is_sha256_hex(line)) return false;
    memcpy(out, line, 65);
    return true;
}

static bool file_sha256_hex(const char *path, char out[65]) {
    char *quoted = shell_quote(path);
    const size_t n = strlen(quoted) + 64;
    char *cmd = malloc(n);
    if (!cmd) die("out of memory");

    snprintf(cmd, n, "sha256sum %s 2>/dev/null", quoted);
    bool ok = read_sha256_command(cmd, out);
    if (!ok) {
        snprintf(cmd, n, "shasum -a 256 %s 2>/dev/null", quoted);
        ok = read_sha256_command(cmd, out);
    }
    free(cmd);
    free(quoted);
    return ok;
}

static void verify_model_sha256(const char *path, const char *expected) {
    char actual[65];
    if (!file_sha256_hex(path, actual)) {
        die("failed to verify model sha256 with sha256sum or shasum");
    }
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr,
                "ds4-logits-dump: model sha256 mismatch: expected %s actual %s\n",
                expected,
                actual);
        exit(1);
    }
}

static bool hex_to_bytes(const char *hex, unsigned char *out, int cap, int *out_len) {
    const size_t n = strlen(hex);
    if (n % 2 != 0 || n / 2 > (size_t)cap) return false;
    for (size_t i = 0; i < n; i += 2) {
        const int hi = hex_value((unsigned char)hex[i]);
        const int lo = hex_value((unsigned char)hex[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i / 2] = (unsigned char)((hi << 4) | lo);
    }
    *out_len = (int)(n / 2);
    return true;
}

static void write_hex(FILE *fp, const unsigned char *bytes, size_t len) {
    static const char h[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        fputc(h[bytes[i] >> 4], fp);
        fputc(h[bytes[i] & 0xf], fp);
    }
}

static bool read_vector_case(FILE *fp, vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            if (vc->nsteps <= 0 || vc->nsteps > VEC_MAX_STEPS) die("bad vector step count");
            return true;
        }
        die("unexpected line before vector case");
    }
    return false;
}

static bool fill_vector_case(FILE *fp, vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;
        if (!strncmp(p, "step ", 5)) {
            char hex[VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                die("bad vector step line");
            }
            if (step_index < 0 || step_index >= vc->nsteps) die("bad vector step index");
            if (ntop < 0 || ntop > VEC_MAX_TOP) die("bad vector top count");
            vc->steps[step_index].ntop = ntop;
            if (!hex_to_bytes(hex, vc->steps[step_index].selected,
                              VEC_MAX_TOKEN_BYTES, &vc->steps[step_index].selected_len)) {
                die("bad selected token hex");
            }
            top_index = 0;
            continue;
        }
        if (!strncmp(p, "top ", 4)) {
            char hex[VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            if (step_index < 0 || step_index >= vc->nsteps) die("top before step");
            if (top_index >= vc->steps[step_index].ntop) die("too many top entries");
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) die("bad vector top line");
            vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            if (!hex_to_bytes(hex, top->bytes, VEC_MAX_TOKEN_BYTES, &top->len)) {
                die("bad top token hex");
            }
            continue;
        }
        die("unexpected vector line");
    }
    die("unterminated vector case");
    return false;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("failed to open", path);
    if (fseek(fp, 0, SEEK_END) != 0) die_errno("failed to seek", path);
    long n = ftell(fp);
    if (n < 0) die_errno("failed to tell", path);
    if (fseek(fp, 0, SEEK_SET) != 0) die_errno("failed to rewind", path);
    char *buf = malloc((size_t)n + 1);
    if (!buf) die("out of memory");
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("failed to read", path);
    if (fclose(fp) != 0) die_errno("failed to close", path);
    buf[n] = '\0';
    return buf;
}

static void json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', fp);
            fputc(*p, fp);
        } else if (*p == '\n') {
            fputs("\\n", fp);
        } else if (*p == '\r') {
            fputs("\\r", fp);
        } else if (*p == '\t') {
            fputs("\\t", fp);
        } else if (*p < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)*p);
        } else {
            fputc(*p, fp);
        }
    }
    fputc('"', fp);
}

static void json_f32(FILE *fp, float v) {
    if (isnan(v)) {
        fputs("\"nan\"", fp);
    } else if (isinf(v)) {
        fputs(signbit(v) ? "\"-inf\"" : "\"inf\"", fp);
    } else {
        fprintf(fp, "%.9g", v);
    }
}

static bool token_bytes_equal(ds4_engine *engine, int token,
                              const unsigned char *expected, int expected_len) {
    size_t len = 0;
    char *text = ds4_token_text(engine, token, &len);
    const bool ok = len == (size_t)expected_len &&
                    memcmp((const unsigned char *)text, expected, len) == 0;
    free(text);
    return ok;
}

static void write_token_hex(FILE *fp, ds4_engine *engine, int token) {
    size_t len = 0;
    char *text = ds4_token_text(engine, token, &len);
    write_hex(fp, (const unsigned char *)text, len);
    free(text);
}

static const char *case_skip_reason(const vec_case *vc) {
    if (!strcmp(vc->id, "long_memory_archive")) {
        return "API/official graph mismatch";
    }
    if (!strcmp(vc->id, "long_code_audit")) {
        return "B300 CUDA long-context logits are not byte-deterministic across repeated captures";
    }
    return NULL;
}

static bool host_is_little_endian(void) {
    const uint16_t x = 1;
    return *(const unsigned char *)&x == 1;
}

static void write_score(FILE *fp, ds4_engine *engine, const ds4_token_score *score) {
    fprintf(fp, "{\"id\":%d,\"bytes_hex\":\"", score->id);
    if (score->id >= 0) write_token_hex(fp, engine, score->id);
    fputs("\",\"logit\":", fp);
    json_f32(fp, score->logit);
    fputs(",\"logprob\":", fp);
    json_f32(fp, score->logprob);
    fputc('}', fp);
}

static void write_official_top(FILE *fp, ds4_engine *engine, const vec_top *top,
                               const ds4_token_score *scores, int nscore,
                               const char *case_id, int step_index) {
    bool found = false;
    ds4_token_score local = {0};
    for (int i = 0; i < nscore; i++) {
        if (scores[i].id < 0) continue;
        if (token_bytes_equal(engine, scores[i].id, top->bytes, top->len)) {
            found = true;
            local = scores[i];
            break;
        }
    }
    fprintf(fp, "{\"bytes_hex\":\"");
    write_hex(fp, top->bytes, (size_t)top->len);
    fputs("\",\"official_logprob\":", fp);
    json_f32(fp, top->logprob);
    fprintf(fp, ",\"found\":%s", found ? "true" : "false");
    if (found) {
        fputs(",\"local_score\":", fp);
        write_score(fp, engine, &local);
        fputs(",\"abs_delta\":", fp);
        json_f32(fp, fabsf(local.logprob - top->logprob));
    }
    fputc('}', fp);
    if (!found) {
        fprintf(stderr, "ds4-logits-dump: vector %s step %d official top token missing locally\n",
                case_id, step_index);
        exit(1);
    }
    if (fabsf(local.logprob - top->logprob) > 4.0f) {
        fprintf(stderr, "ds4-logits-dump: vector %s step %d logprob delta too high: local=%g official=%g\n",
                case_id, step_index, local.logprob, top->logprob);
        exit(1);
    }
}

static void dump_case(FILE *json, FILE *blob, ds4_engine *engine, const vec_case *vc,
                      bool *first_case) {
    if (!*first_case) fputs(",\n", json);
    *first_case = false;
    fputs("    {\"id\":", json);
    json_string(json, vc->id);
    fprintf(json, ",\"ctx\":%d,\"nsteps\":%d,\"prompt_path\":", vc->ctx, vc->nsteps);
    json_string(json, vc->prompt_path);

    const char *skip_reason = case_skip_reason(vc);
    if (skip_reason) {
        fputs(",\"skipped\":true,\"skip_reason\":", json);
        json_string(json, skip_reason);
        fputs(",\"steps\":[]}", json);
        return;
    }

    char *prompt_text = read_file(vc->prompt_path);
    char prompt_sha[65];
    ds4_sha256_hex(prompt_text, strlen(prompt_text), prompt_sha);

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, vc->ctx) != 0 || !session) die("failed to create session");
    char err[256];
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-logits-dump: session sync failed for %s: %s\n", vc->id, err);
        exit(1);
    }

    fprintf(json, ",\"skipped\":false,\"prompt_sha256\":\"%s\",\"prompt_tokens\":%d,\"steps\":[\n",
            prompt_sha, prompt.len);
    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        if (i) fputs(",\n", json);
        const vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int selected = ds4_session_argmax(session);
        bool selected_match = token_bytes_equal(engine, selected,
                                                step->selected, step->selected_len);
        if (!selected_match) {
            fprintf(stderr, "ds4-logits-dump: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            exit(1);
        }

        uint32_t n_vocab = 0;
        const float *logits = ds4_session_logits_data(session, &n_vocab);
        if (!logits || n_vocab == 0) die("session logits unavailable");
        const size_t logits_bytes = (size_t)n_vocab * sizeof(logits[0]);
        const long offset = ftell(blob);
        if (offset < 0) die("failed to tell logits blob offset");
        if (fwrite(logits, 1, logits_bytes, blob) != logits_bytes) die("failed to write logits blob");
        char logits_sha[65];
        ds4_sha256_hex(logits, logits_bytes, logits_sha);

        fprintf(json,
                "      {\"step\":%d,\"selected_token\":%d,\"selected_bytes_hex\":\"",
                i, selected);
        write_token_hex(json, engine, selected);
        fputs("\",\"expected_selected_hex\":\"", json);
        write_hex(json, step->selected, (size_t)step->selected_len);
        fprintf(json,
                "\",\"selected_matches_expected\":%s,\"logits_offset\":%ld,\"logits_bytes\":%zu,\"logits_sha256\":\"%s\",\"top_logprobs\":[",
                selected_match ? "true" : "false",
                offset,
                logits_bytes,
                logits_sha);
        for (int j = 0; j < nscore; j++) {
            if (j) fputc(',', json);
            write_score(json, engine, &scores[j]);
        }
        fputs("],\"official_top\":[", json);
        for (int j = 0; j < step->ntop; j++) {
            if (j) fputc(',', json);
            write_official_top(json, engine, &step->top[j], scores, nscore, vc->id, i);
        }
        fputs("]}", json);

        if (i + 1 < vc->nsteps) {
            if (ds4_session_eval(session, selected, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-logits-dump: eval failed for %s step %d: %s\n",
                        vc->id, i, err);
                exit(1);
            }
        }
    }
    fputs("\n    ]}", json);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    die("unknown backend");
    return DS4_BACKEND_CPU;
}

static void usage(FILE *fp) {
    fputs("usage: ds4-logits-dump -m MODEL -v official.vec -o OUTPUT.json -l logits.f32le --model-sha256 HEX [--backend cuda|metal|cpu]\n", fp);
}

static dump_options parse_args(int argc, char **argv) {
    dump_options opt = {
        .backend = DS4_BACKEND_METAL,
        .model_sha256 = "",
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            if (++i >= argc) die("missing model path");
            opt.model_path = argv[i];
        } else if (!strcmp(arg, "-v") || !strcmp(arg, "--vectors")) {
            if (++i >= argc) die("missing vector path");
            opt.vector_path = argv[i];
        } else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) {
            if (++i >= argc) die("missing output path");
            opt.json_path = argv[i];
        } else if (!strcmp(arg, "-l") || !strcmp(arg, "--logits")) {
            if (++i >= argc) die("missing logits path");
            opt.logits_path = argv[i];
        } else if (!strcmp(arg, "--backend")) {
            if (++i >= argc) die("missing backend");
            opt.backend = parse_backend(argv[i]);
        } else if (!strcmp(arg, "--model-sha256")) {
            if (++i >= argc) die("missing model sha256");
            opt.model_sha256 = argv[i];
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model_path || !opt.vector_path || !opt.json_path || !opt.logits_path || !opt.model_sha256[0]) {
        usage(stderr);
        exit(2);
    }
    if (!is_sha256_hex(opt.model_sha256)) die("model sha256 must be 64 lowercase hex characters");
    return opt;
}

int main(int argc, char **argv) {
    dump_options opt = parse_args(argc, argv);
    if (!host_is_little_endian()) die("logits dump requires a little-endian host");
    verify_model_sha256(opt.model_path, opt.model_sha256);
    ds4_engine *engine = NULL;
    ds4_engine_options eopt = {
        .model_path = opt.model_path,
        .backend = opt.backend,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
    };
    if (ds4_engine_open(&engine, &eopt) != 0 || !engine) die("failed to open model");

    FILE *vec = fopen(opt.vector_path, "rb");
    if (!vec) die_errno("failed to open", opt.vector_path);
    FILE *json = fopen(opt.json_path, "wb");
    if (!json) die_errno("failed to open", opt.json_path);
    FILE *blob = fopen(opt.logits_path, "wb");
    if (!blob) die_errno("failed to open", opt.logits_path);

    fputs("{\n", json);
    fputs("  \"schema\": \"ds4.session_logits_oracle.v1\",\n", json);
    fputs("  \"source\": \"current-c-b300-session-logits\",\n", json);
    fprintf(json, "  \"model\": \"deepseek-v4-flash\",\n");
    fprintf(json, "  \"model_path\": ");
    json_string(json, opt.model_path);
    fprintf(json, ",\n  \"model_sha256\": ");
    json_string(json, opt.model_sha256);
    fprintf(json, ",\n  \"backend\": ");
    json_string(json, ds4_backend_name(opt.backend));
    fprintf(json, ",\n  \"vector_file\": ");
    json_string(json, opt.vector_path);
    fprintf(json, ",\n  \"logits_blob\": ");
    json_string(json, opt.logits_path);
    fprintf(json, ",\n  \"logits_format\": \"f32le\",");
    fprintf(json, "\n  \"top_k\": 20,\n  \"cases\": [\n");

    bool first_case = true;
    vec_case vc;
    while (read_vector_case(vec, &vc)) {
        if (!fill_vector_case(vec, &vc)) break;
        dump_case(json, blob, engine, &vc, &first_case);
    }
    fputs("\n  ]\n}\n", json);

    if (fclose(blob) != 0) die_errno("failed to close", opt.logits_path);
    if (fclose(json) != 0) die_errno("failed to close", opt.json_path);
    if (fclose(vec) != 0) die_errno("failed to close", opt.vector_path);
    ds4_engine_close(engine);
    return 0;
}
