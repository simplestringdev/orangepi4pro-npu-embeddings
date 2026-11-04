/*
 * npu_embed_server: serve o encoder do MiniLM (NBG da Allwinner) na NPU VIPLite por um socket Unix.
 *
 * Uso: npu_embed_server <modelo.nb> <caminho-do-socket>
 *
 * Protocolo (little-endian), uma conexão por lote:
 *   servidor -> cliente ao conectar: 8 x uint32  { magic 'NPUE', bytes_in0, bytes_in1, bytes_out,
 *                                                 data_format_out, fixed_point_pos_out (int32),
 *                                                 data_format_in0, fixed_point_pos_in0 (int32) }
 *     data_format: 0=FP32 1=FP16 3=INT8 2=UINT8 5=INT16 (VIP_BUFFER_FORMAT_*); fixed_point_pos só vale para INT16/INT8
 *     in0 = embeddings [128,384]; in1 = máscara aditiva FP16 [128] (0 ou -1e4)
 *   cliente -> servidor: N x ( in0[bytes_in0] + in1[bytes_in1] )
 *   servidor -> cliente: para cada requisição, out[bytes_out]
 * O cliente fecha a conexão ao terminar. As conexões são atendidas uma a uma (a NPU é uma só).
 *
 * Sem dependência além de libVIPhal/libNBGlinker (já instaladas no Orange Pi 4 Pro).
 * Compilar: ver ops/npu/build.sh.
 */
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "vip_lite.h"

#define MAGIC 0x4555504eu /* 'NPUE' */
#define MAX_TENSORS 8

typedef struct {
    vip_buffer buf;
    uint32_t bytes;
    vip_enum fmt;
    int32_t fp;
    void *ptr;
} tensor_t;

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

static int read_all(int fd, void *buf, size_t n) {
    uint8_t *p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r == 0) return 0;
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += r; n -= (size_t)r;
    }
    return 1;
}

static int write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += w; n -= (size_t)w;
    }
    return 1;
}

static uint32_t elem_size(vip_enum fmt) {
    switch (fmt) {
    case VIP_BUFFER_FORMAT_FP32: return 4;
    case VIP_BUFFER_FORMAT_FP16:
    case VIP_BUFFER_FORMAT_INT16: return 2;
    case VIP_BUFFER_FORMAT_INT8:
    case VIP_BUFFER_FORMAT_UINT8: return 1;
    default: return 0;
    }
}

#define CHECK(call, what)                                                                  \
    do {                                                                                   \
        vip_status_e _s = (call);                                                          \
        if (_s != VIP_SUCCESS) { fprintf(stderr, "npu_embed: %s falhou (status %d)\n", what, (int)_s); return 1; } \
    } while (0)

static int make_tensor(vip_network net, int is_out, uint32_t idx, tensor_t *t) {
    vip_buffer_create_params_t p;
    uint32_t nd = 0, sizes[6] = {0};
    vip_enum fmt = 0, qf = 0;
    int32_t fp = 0;
    memset(&p, 0, sizeof(p));
    vip_status_e (*q)(vip_network, vip_uint32_t, vip_enum, void *) = is_out ? vip_query_output : vip_query_input;
    CHECK(q(net, idx, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &nd), "query dims");
    CHECK(q(net, idx, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, sizes), "query sizes");
    CHECK(q(net, idx, VIP_BUFFER_PROP_DATA_FORMAT, &fmt), "query formato");
    CHECK(q(net, idx, VIP_BUFFER_PROP_QUANT_FORMAT, &qf), "query quant");
    if (qf == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) CHECK(q(net, idx, VIP_BUFFER_PROP_FIXED_POINT_POS, &fp), "query fp");
    if (nd == 0 || nd > 6 || elem_size(fmt) == 0) { fprintf(stderr, "npu_embed: tensor %s%u incompativel\n", is_out ? "out" : "in", idx); return 1; }
    uint32_t count = 1;
    p.num_of_dims = nd;
    for (uint32_t i = 0; i < nd; i++) { p.sizes[i] = sizes[i]; count *= sizes[i]; }
    p.data_format = fmt;
    p.quant_format = qf;
    if (qf == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) p.quant_data.dfp.fixed_point_pos = fp;
    p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    CHECK(vip_create_buffer(&p, sizeof(p), &t->buf), "create buffer");
    t->bytes = count * elem_size(fmt);
    t->fmt = fmt;
    t->fp = fp;
    t->ptr = vip_map_buffer(t->buf);
    if (!t->ptr) { fprintf(stderr, "npu_embed: map buffer falhou\n"); return 1; }
    return 0;
}

static int load_file(const char *path, void **data, uint32_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("npu_embed: abrir modelo"); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    *data = malloc((size_t)n);
    if (!*data || fread(*data, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "npu_embed: leitura do modelo falhou\n"); fclose(f); return 1; }
    fclose(f);
    *size = (uint32_t)n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "uso: %s modelo.nb /caminho/socket\n", argv[0]); return 2; }
    /* sigaction SEM SA_RESTART: o accept() precisa devolver EINTR para o laço ver g_stop e sair limpo */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    void *nb = NULL; uint32_t nb_size = 0;
    if (load_file(argv[1], &nb, &nb_size)) return 1;

    CHECK(vip_init(), "vip_init");
    vip_network net = NULL;
    CHECK(vip_create_network(nb, nb_size, VIP_CREATE_NETWORK_FROM_MEMORY, &net), "create network");
    uint32_t n_in = 0, n_out = 0;
    CHECK(vip_query_network(net, VIP_NETWORK_PROP_INPUT_COUNT, &n_in), "query inputs");
    CHECK(vip_query_network(net, VIP_NETWORK_PROP_OUTPUT_COUNT, &n_out), "query outputs");
    if (n_in != 2 || n_out != 1) { fprintf(stderr, "npu_embed: esperava 2 entradas e 1 saida, achei %u/%u\n", n_in, n_out); return 1; }
    tensor_t in[2], out;
    memset(in, 0, sizeof(in)); memset(&out, 0, sizeof(out));
    for (uint32_t i = 0; i < 2; i++) if (make_tensor(net, 0, i, &in[i])) return 1;
    if (make_tensor(net, 1, 0, &out)) return 1;
    CHECK(vip_prepare_network(net), "prepare network"); /* o driver exige prepare ANTES de set_input/set_output */
    for (uint32_t i = 0; i < 2; i++) CHECK(vip_set_input(net, i, in[i].buf), "set input");
    CHECK(vip_set_output(net, 0, out.buf), "set output");

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(argv[2]) >= sizeof(addr.sun_path)) { fprintf(stderr, "npu_embed: caminho do socket longo demais\n"); return 1; }
    strcpy(addr.sun_path, argv[2]);
    unlink(argv[2]);
    mode_t old = umask(0177); /* socket 0600: so o dono fala com a NPU */
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0 || listen(srv, 8) < 0) { perror("npu_embed: bind/listen"); return 1; }
    umask(old);
    fprintf(stderr, "npu_embed: pronto. in0=%u in1=%u out=%u (fmt %d, fp %d) socket=%s\n", in[0].bytes, in[1].bytes, out.bytes, (int)out.fmt, out.fp, argv[2]);

    uint32_t hdr[8] = {MAGIC, in[0].bytes, in[1].bytes, out.bytes, (uint32_t)out.fmt, (uint32_t)out.fp,
                       (uint32_t)in[0].fmt, (uint32_t)in[0].fp};
    while (!g_stop) {
        int c = accept(srv, NULL, NULL);
        if (c < 0) { if (errno == EINTR) continue; break; }
        if (write_all(c, hdr, sizeof(hdr)) < 0) { close(c); continue; }
        for (;;) {
            int r0 = read_all(c, in[0].ptr, in[0].bytes);
            if (r0 <= 0) break;
            if (read_all(c, in[1].ptr, in[1].bytes) <= 0) break;
            vip_flush_buffer(in[0].buf, VIP_BUFFER_OPER_TYPE_FLUSH);
            vip_flush_buffer(in[1].buf, VIP_BUFFER_OPER_TYPE_FLUSH);
            if (vip_run_network(net) != VIP_SUCCESS) { fprintf(stderr, "npu_embed: run falhou\n"); break; }
            vip_flush_buffer(out.buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
            if (write_all(c, out.ptr, out.bytes) < 0) break;
        }
        close(c);
    }
    close(srv); unlink(argv[2]);
    vip_unmap_buffer(out.buf); vip_unmap_buffer(in[0].buf); vip_unmap_buffer(in[1].buf);
    vip_destroy_buffer(out.buf); vip_destroy_buffer(in[0].buf); vip_destroy_buffer(in[1].buf);
    vip_finish_network(net); vip_destroy_network(net); vip_destroy();
    return 0;
}
