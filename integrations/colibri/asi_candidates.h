/* ASI experiment: immutable candidate mask for one process/request.
 * File: layers experts candidates_per_layer, then ordered IDs per layer.
 * Scores keep their full-model softmax denominator; only eligibility changes.
 */
static unsigned char *asi_candidates = NULL;
static void asi_load_candidates(int layers, int experts, int topk) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;
    const char *path = getenv("ASI_CANDIDATES");
    if (!path || !*path) return;
    FILE *f = fopen(path, "r");
    int l, e, n;
    if (!f || fscanf(f, "%d %d %d", &l, &e, &n) != 3 ||
        l != layers || e != experts || n < topk || n > experts) {
        fprintf(stderr, "ASI invalid candidate header\n"); exit(2);
    }
    asi_candidates = calloc((size_t)layers * experts, 1);
    if (!asi_candidates) { fprintf(stderr, "ASI allocation failed\n"); exit(2); }
    for (int layer = 0; layer < layers; layer++) {
        for (int j = 0; j < n; j++) {
            int id;
            if (fscanf(f, "%d", &id) != 1 || id < 0 || id >= experts ||
                asi_candidates[layer * experts + id]) {
                fprintf(stderr, "ASI invalid/duplicate candidate\n"); exit(2);
            }
            asi_candidates[layer * experts + id] = 1;
        }
    }
    char extra;
    if (fscanf(f, " %c", &extra) == 1) { fprintf(stderr, "ASI trailing candidate data\n"); exit(2); }
    fclose(f);
    fprintf(stderr, "[ASI] fixed candidates=%d/%d per layer, active=%d; full softmax preserved\n", n, experts, topk);
}
