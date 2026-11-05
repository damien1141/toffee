# XSpark v2 — runtime-faithful draft training for ik_llama.cpp

This is a ground-up rewrite of the `xspark` pipeline. It trains draft models
whose PyTorch forward is a **bit-exact mirror of the ik_llama.cpp graph**, and
exports GGUFs that match the loader contract exactly.

## Why the rewrite: the 10% acceptance mystery

The v1 exporter wrote the draft as a **DSV4-backbone** `dflash` GGUF, but the
trainer's actual backbone was a plain `nn.TransformerEncoder`. Only its
LayerNorms were exported — **every attention / MoE / hyper-connection tensor
(`attn_q_a`, `attn_kv`, `ffn_*_exps`, `hc_*`, …) was written as a dummy
zero tensor**. At runtime the DSV4 blocks computed zeros, the residual stream
passed through nearly untouched, and the effective model was:

```
logits[i] = const_vector + W2 @ W1[prev_token]
```

i.e. **a pure bigram model**. The 10% acceptance rate was your Markov head
working alone — the trained parallel backbone never executed at all.

v2 fixes this by training the exact architecture the runtime runs.

## What's in the box

```
xspark/
  model.py    runtime-faithful draft model (dense dflash graph):
              fc + enc.output_norm context fusion, N dense layers
              (q/k per-head norms, NORM rope, SwiGLU), output_norm,
              frozen shared embed/lm_head, and per-variant heads:
                dspark  : markov_w1/w2 + conf_head
                dflash2 : two-tap dynamic convs (4 sites/layer) +
                          selector A/B/H codebooks
  harvest.py  streams the target layer-by-layer (12GB friendly) and stores
              ctx features [S, L*H] + the POST-FINAL-NORM hidden [S, H].
              Teacher logits are reconstructed exactly at train time via the
              frozen lm_head (DeepSpec's trick) — replaces the old lossy
              top-K approximation. Also fixes v1's missing final norm.
  train.py    anchor-block training: random anchors per sequence, block =
              [seed, mask x (B-1)], context K/V from fused features,
              bidirectional block attention, exact full-vocab TV distillation,
              CE + confidence (dspark) or lattice CE (dflash2)
  export.py   writes the dense-dflash GGUF (dspark) or the dflash2 GGUF
              (conv + selector tensors) with the exact tensor inventory,
              shapes, and metadata keys ik_llama.cpp expects
  evalx.py    offline acceptance evaluation that replays the runtime's
              drafting semantics (autoregressive chain / lattice walk) —
              iterate without building C++
  cli.py      prepare | harvest | train | export | eval | run-all
```

## The two variants

Both export under `general.architecture = "dflash"` (ik_llama.cpp
auto-detects the variant from the tensor signature + metadata).

### dspark — parallel backbone + Markov head

- Every block row is a standard next-token row: row `j`'s logits predict
  `x_{p+j+1}` (row 0 is the **seed row** — the last verified token at its true
  position — and its prediction **is** used).
- The head adds a rank-256 bigram correction `W2 @ W1[prev]`, teacher-forced
  during training, autoregressive (own greedy picks) at inference — exactly
  `build_dspark_logits` in ik_llama.cpp.
- Loss: `0.1 * CE + 0.9 * TV + 1.0 * BCE(conf, 1 - TV)` with position decay
  `exp(-k/4)` (the DeepSpec recipe).
- Run with `--spec-type dspark:n_max=8` (max drafts = `block_size`).

### dflash2 — DFlash + path selector + dynamic convolutions

Implements Inco's [DFlash2](https://inco.ai/blog/dflash2/):

- **Two-tap dynamic depthwise convolutions** before/after every attention and
  FFN sublayer: `out[t,c] = Σ_k (dyn[t,s,k,c//gs] + base[s,k,c]) * x[t-k,c]`.
  The whole draft batch is one conv block, so row 1's tap reads the **seed
  row = the last verified token** — the blog's Figure 4 exactly.
- **Candidate selector**: the runtime keeps each row's top-K candidates and
  scores adjacent pairs
  `S_t(a,b) = U_t(b) + ⟨B(b), A(a) ⊙ H(h_t)⟩`
  (`selector_predecessor` = A, `selector_successor` = B, `selector_hidden` =
  H on the output-normed hidden). A greedy walk (or temperature sampling with
  rejection sampling) picks the path.
- **Alignment** (this is the subtle part, verified against both ik_llama.cpp
  and mainline llama.cpp): mask rows predict their **own position** `x_{p+j}`;
  the walk reads lattice columns `1..B-1` with the **anchor token** as the
  pos-1 predecessor; the seed row's own logits are unused (they exist only to
  feed the conv tap and the anchor lookup).
- The selector is trained with a **lattice CE**: for every predecessor state
  (all K candidate columns), cross-entropy toward the true token — this
  directly optimizes the walk *including recovery from wrong states*.
- Run with `--spec-type dflash:n_max=7` (max drafts = `block_size - 1`).

## Usage

```bash
# 1) data
python -m toffee xspark prepare

# 2) harvest teacher features (streams the target layer-by-layer)
python -m toffee xspark harvest --model ./Sage-A1.5 \
    --output-dir xspark_data/harvest --samples 2000 --seq-len 2048 \
    --target-layers auto          # 5 evenly spread interior layers

# 3a) train a dspark draft
python -m toffee xspark train --variant dspark --target-model ./Sage-A1.5 \
    --data-dir xspark_data/harvest --output-dir xspark_out/dspark \
    --epochs 3 --anchors 16 --block-size 8

# 3b) ...or a dflash2 draft (recommended focus)
python -m toffee xspark train --variant dflash2 --target-model ./Sage-A1.5 \
    --data-dir xspark_data/harvest --output-dir xspark_out/dflash2 \
    --epochs 3 --anchors 16 --block-size 8

# 4) offline acceptance eval (replays runtime semantics — no C++ needed)
python -m toffee xspark eval --head xspark_out/dflash2/dflash2_head.pt \
    --target-model ./Sage-A1.5 --data-dir xspark_data/harvest --anchors 512

# 5) export to GGUF (tokenizer copied from the target GGUF)
python -m toffee xspark export --head xspark_out/dflash2/dflash2_head.pt \
    --target-model ./Sage-A1.5 --target-gguf ./Sage-Q4KM.gguf \
    --dtype f16

# 6) serve with ik_llama.cpp
./llama-server -m ./Sage-Q4KM.gguf -md xspark_out/dflash2/DFlash2-Draft.gguf \
    --spec-type dflash:n_max=7 -c $((2048+64)) --temp 1.0
# dspark variant: --spec-type dspark:n_max=8
```

Notes:
- The draft GGUF **omits** `token_embd`/`output` by default — ik_llama.cpp
  aliases them from the target at spec-init (the preferred IO-sharing mode;
  dflash2 requires it). Pass `--self-contained` to embed frozen copies.
- `--anchors` is the main VRAM knob (blocks sampled per sequence per step).
  16 fits ~12GB comfortably at hidden 2048; raise it on bigger GPUs.
- The eval reports `accept_len` (tokens committed per verify round, incl. the
  bonus token), per-position `accept@k`, `recall@k` and `oracle_len` (the
  recall ceiling — the selection headroom from the DFlash2 blog).

## The GGUF contract (verified against the ik_llama.cpp loader)

Metadata (arch `dflash`):

| key | value |
|---|---|
| `dflash.block_size` | draft block rows (seed + B-1 masks) |
| `tokenizer.ggml.mask_token_id` | resolved from the HF tokenizer / target GGUF |
| `dflash.target_layers` | **1-based** layer ids (auto: 5 evenly spread interior layers) |
| `dflash.target_hidden_size` | target hidden (cross-check) |
| `attention.causal` | `false` (blocks are bidirectional) |
| `attention.layer_norm_rms_epsilon`, `rope.freq_base`, head counts, `embedding_length`, `block_count`, `feed_forward_length` | standard |
| dflash2 only: `dflash.conv_kernel_size`, `dflash.conv_group_size`, `dflash.selector_rank`, `dflash.selector_top_k` | conv/selector geometry |

Tensors (numpy shape → ggml `ne` is the reverse):

| tensor | numpy shape | notes |
|---|---|---|
| `fc.weight` | `(n_embd, L*n_embd)` | target-feature fusion |
| `enc.output_norm.weight` | `(n_embd,)` | after fc |
| `output_norm.weight` | `(n_embd,)` | before lm_head & selector H |
| `blk.N.attn_{q,k,v}.weight` | `(heads*hd, n_embd)` / `(kv*hd, n_embd)` | dense attention |
| `blk.N.attn_{q,k}_norm.weight` | `(hd,)` | per-head RMSNorms (mandatory) |
| `blk.N.attn_output.weight` | `(n_embd, heads*hd)` | |
| `blk.N.ffn_{gate,up}.weight` | `(n_ff, n_embd)` | SwiGLU |
| `blk.N.ffn_down.weight` | `(n_embd, n_ff)` | |
| `markov_w1.weight` | `(vocab, rank)` | dspark |
| `markov_w2.weight` | `(vocab, rank)` | dspark |
| `conf_proj.{weight,bias}` | `(1, n_embd+rank)` / `(1,)` | dspark (dead at runtime, carried for the loader) |
| `selector_predecessor.weight` | `(vocab, rank)` | dflash2 (A) |
| `selector_successor.weight` | `(vocab, rank)` | dflash2 (B) |
| `selector_hidden.weight` | `(rank, n_embd)` | dflash2 (H) |
| `blk.N.attn_conv_base` | `(2, K, n_embd)` | dflash2 — **no `.weight` suffix!** |
| `blk.N.attn_conv_proj.weight` | `(2*K*n_embd/gs, n_embd)` | dflash2 |

## Training-geometry cheat sheet

```
context features:  positions 0 .. p-1      (kv strictly before the block)
seed row:          x_p at its TRUE position p   (features lag tokens by one)
mask rows:         positions p+1 .. p+B-1
mask (full):       every row sees all ctx rows (pos < p) + all block rows

dspark  rows:  row j  ->  predicts x_{p+j+1}   (row 0 = seed, its output used)
dflash2 rows:  row j  ->  predicts x_{p+j}     (row j >= 1; row 0's logits unused)

dflash2 lattice:  column j uses row j's top-K candidates;
                  column 1's predecessor = the anchor token;
                  column j>1's predecessors = row j-1's candidates;
                  walk: prev = 0; for pos in 1..B-1: pick argmax/sampled
                        successor at scores[pos][:, prev]
```

## Validation

`tests/xspark_smoke_test.py` — CPU-verifiable guarantees:

1. `dflash2_conv` matches a literal transcription of `build_dflash2_conv`
   (both conv sides), value-for-value.
2. The selector lattice matches a literal transcription of the C++ loop
   (`build_dflash.cpp` L642-700): scores, unary, anchor replication, column-0
   zeros.
3. Forward/backward for both variants: gradients reach fc / attention / FFN /
   markov / selector / conv params; the frozen shared embed + lm_head are
   never touched.
4. Exported GGUFs round-trip: every tensor name + shape + KV key verified
   against the loader contract (29 tensors dspark, 36 dflash2).

`tests/xspark_learn_test.py` — end-to-end learning on a synthetic bigram
world (V=256, perfect teacher): dspark reaches **accept_len 6.7/8** (89%
first-token accuracy vs the 90% world ceiling); dflash2 reaches
**accept_len 5.0/7** with 78% recall@0 and ~96% selection accuracy — with
*random* context features, i.e. the conv/selector chain alone carries the
signal. Real runs also get the target's layer features as context.

## References

- DFlash2 blog: https://inco.ai/blog/dflash2/
- DSpark/dFlash training reference: https://github.com/deepseek-ai/DeepSpec
- Runtime: https://github.com/ikawrakow/ik_llama.cpp
  (`src/graphs/build_dflash.cpp`, `common/speculative-dflash-impl.h`)
