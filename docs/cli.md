# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate hybrid MoE decode |
| `ft bench decode` | Decode with `offload` and with `hybrid` on this machine, report the faster and file it for `--moe-strategy auto` |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, a `.gguf` file, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |
| `--hf-overrides` | — | JSON object applied to the checkpoint's config as vLLM's `--hf-overrides`: a nested config section updates key by key, any other value is replaced whole. A YaRN `rope_parameters` override serves `original_max_position_embeddings * factor` positions |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--api-key` | `$FREETOKEN_API_KEY`, else off | Require the key on every route except `/health`, as `Authorization: Bearer <key>` or `x-api-key: <key>`; other requests get a 401 |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--preset` | — | `single`: the settings for one request at a time on a GPU smaller than the model, filled in only where the command line says nothing. `single-measured`: the same, and the first start times what the settings cannot choose. `single-fast`: `single` with a captured prefill forward for continuations of up to 64 tokens, with `single`'s replies. `single-bucket`: `single-fast` with captured forwards for chunks of exactly 128, 256, 512 and 1,024 tokens, with `single`'s replies. `multi`: `single` for several requests at a time, 8 unless `--max-running-requests` names another number. See [Presets](#presets) |
| `--calibrate-again` | off | With `--preset single-measured`: time on this start also when a choice is stored for this checkpoint, flags and machine: the candidates the stored calibration had no time for, or all of them once none is left. See [Presets](#presets) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens; with offloaded experts see [performance/prefill.md](performance/prefill.md) |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--cuda-graph-bs` | 1, 2, 4 and the multiples of 8 up to `--cuda-graph-max-bs` | The batch sizes captured as CUDA graphs, e.g. `1 2 3 4 5 6 7 8 12 16`. A decode batch replays the graph of the next size up with its other rows padded, so a size of its own saves the padded rows; a batch above the largest size runs without a graph. Each size takes GPU memory and startup time. Not with `--cuda-graph-max-bs` |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--embed-weights` | gpu | Where the input token embedding tables live. `host` keeps them in pinned host RAM and each lookup reads its rows over PCIe, so their VRAM goes to the expert cache and KV. A table tied to the LM head stays on the GPU, and `host` is refused when there is no other table; Gemma-4 E2B / E4B's per-layer table is the one that moves, the way those models fit an 8 GB GPU |
| `--online-quant` | — | Convert the Linear layers and input embedding tables that the checkpoint stores in bf16 at load. `df11` compresses them losslessly to about 11 bits per weight and decodes them bit-for-bit on use: about 30% less VRAM for those weights, at the cost of a decode per use. Layers the checkpoint quantizes, the LM head, a table tied to it, vision encoders and layers whose weight the model reads directly are left as they are, and so are tables under TP > 1; refused when no layer qualifies |
| `--online-quant-modules` | all | Limit `--online-quant` to these modules: comma-separated checkpoint module names or glob patterns, e.g. `model.layers.*.self_attn,model.embed_tokens` (`model.language_model.layers.*` where the checkpoint nests the text model); a name also covers the modules under it, and one that selects no convertible layer is refused. A fused layer (`qkv_proj`) must be covered in all of its checkpoint parts |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |
| `--kv-cache-dtype` | auto | How K and V are stored. `auto` keeps the compute dtype. `fp8_per_token_head` (e4m3) and `int8_per_token_head` store 8 bits per element plus one fp32 scale per token and head (two per MLA latent row, for ckv and kpe), about half the KV memory. The `triton`, `qsa_sparse`, `m3_sparse` and `dsa` backends read them; `auto` picks one of these. They need `--dtype bfloat16` and no tensor parallelism. An `mtp` draft stores its KV in the same format |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does, [performance/prefill.md](performance/prefill.md) for prefill speed, and [performance/decode.md](performance/decode.md) for decode speed.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid where [`ft bench decode`](#ft-bench-decode) measured it faster for the same flags; fused on unified-memory GPUs (GB10), see [models.md](models.md#unified-memory-gpus-gb10--dgx-spark). `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name. Per-tensor FP8 linears with a static activation scale decode on a split-K kernel (`triton-w8a8`) when no decode step of the server has more than 8 rows; it adds the same products as `torch._scaled_mm` in another order, so outputs can differ in rounding. The kernel is chosen per call, so a prefill chunk of 8 tokens or fewer (such as the uncached tail after a prefix-cache hit) takes it too. `linear.fp8_tensor=torch` keeps `torch._scaled_mm` |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-mem` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / memory such as `12GiB` / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--moe-cache-scope` | unified | `unified`: one LRU pool of slots per expert quant (quant type and bank layout), so all MoE layers share one pool when they share a quant format; `per-layer`: each layer has its own pool. `per-layer` does not combine with `--moe-strategy cpu` or `--moe-cpu-layers` |
| `--moe-cache-split` | equal | How the pools share the slots: `equal` (in proportion to their layer counts), or `profile:<path>`: every pool keeps `top_k` slots and the rest go to the most used experts in the profile's per-layer expert counts. With `offload`, a pool smaller than the expert count lowers `--max-running-requests` to its slots / top_k |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`/`hybrid`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for hosts that cap CUDA pinned memory (Windows/WSL): it pins layers from the middle out until CUDA refuses one and OS-locks the rest (`FREETOKEN_PIN_BUDGET_GB` and FTW checkpoints lock from a size estimate instead), see [models.md](models.md#hosts-that-cap-pinned-memory-wsl); every value needs an expert format the CPU executor serves (bf16, fp8_block, nvfp4, mxfp4, GGUF on x86 CPUs, except gpt-oss GGUF files whose per-expert biases decode on the GPU only) |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-locked-fetch` | auto | With `hybrid` and locked layers: share of a locked layer's misses fetched per step into its GPU slots through pinned staging. Auto takes p / (2p + c) from the `ft bench bw` row of the layer's kind (at most 1 per step without a row); 0 = locked layers decode on CPU only |
| `--moe-locked-max-fetch` | 8 | Most experts a locked layer fetches per step; sizes the staging (that many rows of the widest expert, per bank) |
| `--moe-collect-stats` | off | Log decode MoE stats with each decode log line: hit rate, and per layer-step the misses, experts fetched and experts computed on CPU, for pinned, locked and CPU-only layers |
| `--prefill-graph-tokens` | 0 | Prefill: with speculative decoding, a chunk of at most this many tokens on a cached prefix runs as captured verify steps instead of an eager forward; 0 = always eager |
| `--prefill-graph-chunks` | none | Prefill: with `--prefill-graph-tokens`, token counts of the prefill forwards to capture, e.g. `1-64,128-384:64`; a chunk within the limit then runs as one or two captured forwards, cut at its last 64-token boundary, instead of one eager forward. Needs every expert resident |
| `--prefill-graph-pad` | off | Prefill: with `--prefill-graph-chunks`, the captured multiples of 64 past 64 tokens are bucket captures; a chunk of exactly such a size runs as one replay of it, which computes what its eager forward computes, and any other chunk of more than 64 tokens runs as one eager forward. Only with the experts resident and no expert cache; elsewhere those chunks run eagerly |
| `--prefill-graph-rows` | 0 | Prefill: rows of a wider verify step captured for `--prefill-graph-tokens` chunks longer than a decode step; 0 = only the decode step's rows |
| `--moe-prefill-demand-tokens` | 0 | Prefill: a chunk of at most this many tokens loads only the experts it routes to, through the expert cache as decode does (GPU decode only); 0 streams whole layers |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |
| `--moe-expert-activation-scales` | off | Research arm, changes the output: serve NVFP4 experts under the `input_scale` tensors the checkpoint stores for them even where its config declares W4A16, so prefill forwards of at least 2048 tokens run the W4A4 expert kernel (sm_120; `FREETOKEN_NVFP4_ACT_SCALES_MIN_TOKENS` sets the length). Shorter prefill forwards and decode keep the W4A16 kernels. `FREETOKEN_NVFP4_W4A4_ATOMIC_SUM=1` makes the W4A4 down kernel add a token's routes with atomics instead of summing stored route rows: the last bits then vary from run to run, and `--reproducible-seeds` refuses it |

### Speculative decoding

| Flag | Default | Meaning |
|---|---|---|
| `--speculative-algorithm`, `--spec-algo` | off | Draft method; `off` keeps it off under a [preset](#presets). `mtp` drafts with the checkpoint's multi-token-prediction layer: Qwen3.5 / Qwen3.6 HF or GGUF checkpoints, and Qwen3.8-Flash-Next HF or GGUF; Gemma-4 (dense, E2B / E4B and MoE) keeps no draft in the target file and drafts from an `mtp-*.gguf` assistant sidecar (architecture `gemma4-assistant`) given with `--speculative-draft-model-path` / `--spec-draft-model`. `dflash` drafts a block of tokens per step with a DFlash 2 draft checkpoint over the target's layer outputs, for the Qwen3.5 / Qwen3.6 / Qwen3.8 family and Qwen3.8-Flash-Next: `incoai/Qwen3.8-27B-DFlash2` for Qwen3.8-27B, `incoai/Qwen3.6-35B-A3B-DFlash2` for Qwen3.6-35B-A3B, `incoai/Qwen3.8-Flash-Next-DFlash2` for Qwen3.8-Flash-Next |
| `--speculative-num-steps`, `--spec-num-steps` | the method's default (3 for `mtp`, the draft's `block_size` - 1 for `dflash`) | Draft tokens per decode step, 1 to 8 for `mtp`; 1 to the draft's `block_size` - 1 for `dflash` |
| `--speculative-draft-model-path`, `--spec-draft-model` | the target checkpoint | Where the draft weights load from. A Qwen3.8-Flash-Next target also takes a Strata MTP pack (a GGUF that keeps the checkpoint's `mtp.*` tensors under their HF names), and a Qwen3.8-Flash-Next GGUF target needs one; a Qwen3.5 / Qwen3.6 GGUF target drafts from its own `nextn` MTP block (the block llama.cpp keeps after the last decoder layer), or from an `mtp-*.gguf` sidecar of the same model (unsloth's `MTP/` releases) given here; for `mtp`, a separate checkpoint folder is an HF-target option, read with the target's quantization config, and a GGUF target refuses a folder draft. `dflash` requires it: the DFlash 2 draft checkpoint (bf16 safetensors with its `config.json`), whose `target_layer_ids`, `num_target_layers`, hidden size and vocabulary must match the target |
| `--speculative-token-map`, `--spec-token-map` | off | `mtp` only. A file of distinct little-endian int32 token ids (`python scripts/make_token_map.py MODEL` writes one, from token ids in order or from text: the checkpoint's own replies, text files, a directory of code and prose; it writes a manifest of what it counted beside the map and refuses a file whose name or text marks it as a benchmark's, since a map built from a test set's tokens raises the acceptance on that test set only): the draft computes its logits over those LM head rows alone, from a copy made at startup whose size is logged. That VRAM comes out of the expert cache when `--moe-cache-auto` sizes it (the default for hybrid and offload without a cache size), and out of the KV cache otherwise. The target still verifies over the whole vocabulary, so the output distribution does not depend on the file, only the acceptance rate does. A tied or block-FP8 LM head is refused |
| `--speculative-draft-quant`, `--spec-draft-quant` | `none` | `mtp` only, for a Qwen3.5 / Qwen3.6 MoE safetensors checkpoint whose MTP layer keeps its routed experts in bf16. `fp8` (e4m3 with 128x128 block scales) or `nvfp4` quantizes those experts at load and keeps them resident in that format, served by the same kernels as a checkpoint's own experts of that format; one startup line gives the bank's bytes before and after. The freed VRAM goes to the expert cache when `--moe-cache-auto` sizes it, and to the KV cache otherwise. The target still verifies every draft, so the output distribution does not depend on it, only the acceptance rate does. Refused for a draft whose experts the checkpoint already quantizes, a GGUF draft, a dense model, Qwen3.8-Flash-Next, Gemma-4 and `dflash` |

Speculative decoding needs `--tp 1` and runs without overlap scheduling; a target with sliding-window KV is served only by the Gemma-4 assistant, which reads the target's cache through a read-only view; every other target with sliding-window KV is refused at startup, and so is every target with DSV4 window-tier KV. The Gemma-4 assistant keeps no KV of its own: it reads the target's cache, so it adds nothing to the KV budget and its weights cost about 149 MiB for the E4B draft, about 425-428 MiB for the 12b and 26B-A4B Q8_0 ones. `mtp` refuses an FTW checkpoint as the draft source, since FTW drops the `mtp.*` tensors, and a Qwen3.5 / Qwen3.6 GGUF converted without its MTP layer (llama.cpp `--no-mtp`) when no `mtp-*.gguf` sidecar is given: such a trunk has no draft of its own to read, while a `--no-mtp` trunk plus a sidecar works (the trunk config's shapes, the sidecar's weights).

`dflash` keeps the draft's context in a per-request ring of the draft's sliding window (2048 positions for the released drafts) and, on a hybrid GDN target, in a window on the GDN state slots, so a prefix-cache hit restores it with the GDN snapshot; the window adds `(window + 64) x hidden` activation-dtype bytes to every GDN slot (`/v1/cache/status` counts them). A target without GDN layers needs `--cache-type naive`. Quantized drafts and DFlash v1 drafts are refused. With sampling, the drafts are drawn from the draft's own distribution over its top-16 candidates and verified by rejection sampling against the target's distribution, so the output distribution is the target's.

### Presets

`--preset single` is for one user at a time on a GPU that cannot hold every expert. It sets
`--max-running-requests 1`, `--embed-weights host`, `--memory-ratio 0.95`, `--moe-prefill-hit-d2d` and
`--moe-prefill-demand-tokens 512`, turns on `--speculative-algorithm mtp` with 3 draft tokens, and raises `--kv-reserve-tokens` to the
default max output tokens plus a 4096-token prompt (36864), so that an answer of the default length is
not cut off with `finish_reason: length`. What each of these costs or buys in expert slots is in
[performance/decode.md](performance/decode.md#getting-more-slots).

A flag on the command line always wins, also when it repeats the default (`--memory-ratio 0.9`);
`--speculative-algorithm off` keeps speculative decoding off. Four settings are dropped where they
cannot apply, each with one log line: speculative decoding where the checkpoint has no MTP layers (or
an explicit `--speculative-algorithm mtp` would be refused), host embeddings where the table is tied
to the LM head, the demand-loaded prefill chunks where every expert has a place on the GPU (resident,
or a slot in the expert cache), the cache decodes on the CPU or `hybrid`, or one of its slot pools is
smaller than a layer's experts, and the KV reserve where the expert cache's minimum does not fit beside it (the
warning names the largest `--kv-reserve-tokens` that does). The KV reserve is never more than the
model's context, and it is lowered, to the default at the least, where it would leave fewer expert
slots than a server without the preset plans. The preset's memory ratio comes with the working memory
of one prefill chunk of `--max-prefill-length` tokens left free on top of its headroom (0.65 GiB for
8192 tokens on Qwen3.6-35B-A3B, counted from the GatedDeltaNet layer's tensor shapes; the layer runs
its chunk rule over one request in calls of 2048 tokens, `FREETOKEN_GDN_RULE_TOKENS`, with the output
of one call over all of them, and as one call, which a value of 0 or several running requests give,
the count is 1.03 GiB): `--memory-ratio
0.95` alone leaves less than that on a 16 GB card, and a long prompt then ends the server with a CUDA
out-of-memory error. An automatically sized expert cache takes that memory from the KV tokens above
`--kv-reserve-tokens` first and from expert slots after, and neither cache goes under its minimum.
Where that cannot leave the whole chunk free, the preset's own KV reserve gives way before the chunk
does: it is lowered to the largest reserve that can, at the least to a prompt as long as the chunk
and 1024 tokens of answer (never under the default reserve). If the plan is still short, the server
starts, the plan line says how much it keeps free, and one warning names the prompt length that
fits: a longer prompt can end the server. For a model
without GatedDeltaNet layers the preset keeps the default ratio. A
typed `--memory-ratio` gets no such reserve. The startup log lists what the preset set,
what each setting did to the `--moe-cache-auto` plan, and the resulting expert slots and KV tokens.

Under `offload` the preset serves without its drafts where the expert cache holds under 20% of all
experts: a `--moe-cache-rate`, `--moe-cache-size` or `--moe-cache-mem` that small, or an automatic
plan that small with the drafts loaded and the preset's KV reserve (counted before the expert banks
load, a slot's bytes taken from the bank layout). At such a cache the drafted rows miss the cache
more often than their accepted tokens pay back. One log line says so and how much GPU memory the
draft's weights gave back to the caches. A `--speculative-num-steps` or `--speculative-algorithm` on
the command line keeps the drafts, and so do `hybrid` and `--moe-cpu-layers`. The 20% comes from four
decode cells of Qwen3.6-35B-A3B NVFP4 on RTX 5090s, the preset with 3 drafts against a server with no
flags: slower at 512 and 701 of 10240 slots, faster at 2048 and 2991. Nothing was measured between
6.8% and 20%.

With its drafts on, the preset quantizes the draft's bf16 routed experts at load
(`--speculative-draft-quant`): `fp8` where every expert is resident (`fused`), `nvfp4` where the
experts are served from an expert cache, whose slots get the bytes that frees. The target still
verifies every draft, so the output distribution is the same; only the acceptance rate can change.
A draft whose experts the checkpoint already quantizes, or any other draft that refuses the flag, is
served as stored, with one log line. A typed `--speculative-draft-quant`, `none` included, is kept.
`ft bench decode` rows filed for the preset before it set the draft quant are another setup and are
not read.

Where every expert fits on the GPU the preset stops streaming them, in one of two ways, and the
plan line says which. Both give the same output as the expert cache does.

- If `--moe-strategy` is not on the command line and the checkpoint's weight files, the KV reserve
  and the prefill chunk above fit in the memory ratio's share of the free GPU memory, the experts
  are served resident (`--moe-strategy fused`): there is no expert cache, and a prefill reads them
  in place. The sizes are counted before the load from the files on disk, so a checkpoint that is
  not a local folder, a `--moe-cache-*` flag or `--moe-cpu-layers` keeps the expert cache. A typed
  `--moe-strategy`, `auto` included, is never replaced.
- With an automatically sized expert cache (`offload`) that has room for every expert, the plan
  takes two layers' worth of slots more than there are experts, out of the KV tokens above the
  reserve. The prefill staging buffers lie over that many slots, and only with every expert in a
  slot outside them does a prefill read the slots as decode does. With fewer slots a prefill keeps
  streaming what is not resident (`--moe-prefill-hit-d2d`), and a chunk of at most 512 tokens (a
  resend, a short turn) loads only the experts it routes to (`--moe-prefill-demand-tokens`). On one
  16 GB card, 512 started second turns of 64 and 256 new tokens 40% and 13% to 18% sooner than 96,
  with the same text and nothing slower, and 1024 made a cold 1000-token prompt 14% slower. A longer
  chunk does the same when the routes of its first layer and the cache's contents say that this copies
  fewer bytes than streaming it, counting an expert it would evict as one more copy; a cache with
  empty slots keeps streaming, which fills them. `FREETOKEN_PREFILL_DEMAND_BY_MISSES=0` leaves the
  choice to the token count alone.

`--preset single` measures nothing: it serves with `fused` where every expert fits as described
above, else with what `--moe-strategy auto` takes from an [`ft bench decode`](#ft-bench-decode)
row, and 3 draft tokens. Where no row answers for the setup it serves `hybrid` if the CPU MoE pool
has 8 threads or more (`--moe-cpu-threads`, or one a physical core within the CPU quota), and
`offload` otherwise: on one box's 12 GiB, 8 GiB, 20% and 5% cells, `offload` there was 31% to 51%
under the fastest typed flags, and `--preset single-measured`, which took `hybrid` with 3 draft
tokens, within 1% to 19%. A typed `--moe-strategy`,
`--max-running-requests` or `--moe-cpu-layers` keeps what it set, and a typed `--reproducible-seeds`
or `--moe-prefill-demand-tokens`, which `hybrid` cannot serve, keeps `offload`; `--preset
single-measured` then times no `hybrid` candidate either. Under `hybrid` a prefill chunk
does not load its experts on demand (above), and hybrid's text can differ from offload's where two
tokens are nearly tied (below). Where it serves `fused` (the
first way above) and the store below holds a draft token map for the checkpoint whose sha256 is
the one its manifest records, it serves 5 MTP draft tokens with that map instead; on two cards
that held every expert of Qwen3.6-35B-A3B NVFP4, 5 drafts with a 32K map decoded 19% and 5%
faster than 3 drafts without one on a math and a coding prompt. With no such map, or a map the
checks of a typed `--speculative-token-map` refuse, it serves 3 draft tokens without one. An
expert cache, `--speculative-algorithm`, `--speculative-num-steps` or `--speculative-token-map`
on the command line, and a typed `--moe-strategy fused`, leave the drafts as they are. One log line
says which and why. `--preset single-fast`, `--preset single-bucket` and `--preset single-measured` do the same: for
`single-measured` that is its static choice, the first candidate it times (below). `--preset single-measured`
is the same preset, and on its first start for a checkpoint, set of flags and machine it times the
candidates the preset cannot choose between, the static preset's choice first. With every expert
resident: `fused` with 5 draft tokens and the store's draft token map (below), where the static
preset serves it, and with 3 draft tokens; with only one of the two, nothing is timed. With an
expert cache that holds part of the experts: `hybrid` and `offload` with 3 draft tokens and `hybrid`
with none; `offload` with none is the `offload` server where that one drops its drafts for the size
of the cache (above), and is not timed where it keeps them. Where an automatic expert cache holds
under 20% of the experts, `--preset single` serves `offload`, and the candidates are `offload` with
3 draft tokens and `hybrid` with none. That `hybrid` is not chosen where its server's plan leaves
less than a prefill chunk's working memory free or lowers the KV reserve the preset asks for. Both
are needed and not shown to be enough: on one box `hybrid` with 3 draft tokens ran out of memory on
long prompts with both kept, so it is not a candidate there until a box run shows it answering them.
A typed `--moe-strategy hybrid` on such a cache is served as typed, untimed. 4 and 5 draft tokens
without the map are not candidates. Each candidate is one server start that decodes two built-in
prompts (a math problem and a coding task) twice over, with the checkpoint's sampling: N timed
tokens after 1024 warm-up tokens that fill the expert cache, then N timed tokens after 256 warm-up
tokens on the cache the first pass left.
N is 512 with every expert resident (256 timed tokens put 5 drafts with a
map 8% ahead of 3 on one prompt and 3% behind on the other, where 2,048 put them 19% and 5% ahead)
and 256 on an expert cache, where a server load takes most of a candidate's time and the shorter
window leaves the 300 s room for a second candidate. A candidate's lead over another is
taken per prompt from the means over the two passes and is the smaller of the two prompts'; a
candidate behind in any single pass has none. The static preset's choice is chosen first; then, in
the order they are timed, a candidate replaces the one chosen so far where it leads that one by more
than 10%, so a candidate barely ahead of another that leads is not taken. What is chosen runs; the
static choice runs with `--moe-strategy` as the command line left it, and with the static preset's
own drafts, also where the preset drops them. The strategy and the number of
draft tokens are chosen together by that one rule; a depth is not chosen per prompt. The launcher logs
the choice in one line (strategy, draft tokens, the KV reserve asked and the expert cache timed), after
a calibration and after a stored one alike. Where a start resolves another strategy or number of draft
tokens than the chosen candidate's server was timed with, it serves the static preset instead, before
any weights load, with one warning that names both settings. A stored calibration is then timed again on
the next start; one timed on that start serves the static preset on every start after it, until
`--calibrate-again`. Once the plan is final the server logs the strategy, draft tokens, KV reserve and
expert slots it serves beside those of the candidate's server; drafts that the preset drops or keeps for
the size of the cache, or an expert cache more than 5% off, are a warning only.

The draft token map is the store's for the checkpoint: under a typed `--moe-strategy fused`, 5 draft
tokens with it are a candidate against the static 3. The store is the directory of the decode profile, and
`python scripts/make_token_map.py MODEL --store ...` writes the map there under the checkpoint's
fingerprint with its manifest; a file whose manifest is missing or names another sha256 is not timed.
`python scripts/make_token_map.py MODEL --install MAP` copies a map built before and its `MAP.json`
there, if the sha256 the manifest records (as `map_sha256` or `sha256`) is the file's and the map
fits MODEL's tokenizer (the vocabulary size the manifest records, where it records one, and every id
below it), and prints where it went. The fingerprint is the checkpoint's folder name and a hash of
its weight files and `config.json`; where no map is stored under the full fingerprint, one stored
under the same hash is read, so the Hugging Face snapshot folder of a checkpoint finds a map
installed from a copy of it with another folder name.
A `--speculative-token-map` on the command line is not a candidate: every server then has that map.
No map is shipped: one has to be built for the checkpoint, from text that no benchmark uses.

The calibration runs before the server loads and has 300 seconds. Every candidate is a model load,
because a running server cannot change its strategy or its draft depth, so the candidates are timed
in the order above and only as many as fit, the static choice first; where it is `hybrid`,
`hybrid` with no draft tokens comes second (the fastest setting with 8 GiB free on one box), then
`offload`. A candidate is started only while 1.1 times the longest candidate so far still fits in
what is left, and its server is stopped if it is not ready with time left for its requests. The end
of the 300 seconds is also a deadline for every load and every request, the static choice's
included: what is still running then is ended, and that server is gone at most 30 seconds later,
by SIGKILL if it does not stop. If the static choice itself is not timed by then, the static preset
runs untimed and the log says so. On a machine that loads the model in 80
seconds that is two or three candidates; the log names the ones that were not started, and prints
the time (`calibration took N s`). A candidate that was not timed is stored as such, is not chosen,
and is not timed on a later start; `--calibrate-again` keeps the stored timings and spends another
300 seconds on the candidates not timed yet. A strategy or a speculative setting given on the command line is not
timed, and a checkpoint that drafts nothing is timed for the strategy alone (two model loads).
Where the static preset serves the experts resident, `fused` is every candidate's strategy: with
every expert on the GPU `hybrid` has nothing to compute on the CPU. Where what the static preset
serves on the GPU cannot be read, nothing is timed and the static preset runs.
SIGTERM, SIGHUP or Ctrl-C during the calibration stops the candidate's server and ends the launcher
with exit code 128 + the signal.

The timings are kept in the decode profile, next to the `ft bench decode` rows, and are read under
the same conditions as those rows (same checkpoint and flags, same machine and release), so later
starts measure nothing and say which calibration they read. `--calibrate-again` times on this
start whatever is stored: the candidates the stored calibration had no time for, or, once every
candidate has been timed, all of them again. A `--moe-cache-rate` or `--moe-cache-size`
is part of those flags. With an automatically sized cache the timings are read only while about as
much GPU memory is free as when they were taken: clearly more or clearly less is timed again. Where
the calibration cannot be done or its
static candidate does not start, the static preset runs and one log line says why; a candidate that
does not start is not chosen, the log names it, and the next start times the candidates again (a
candidate that fails on two starts in a row is left out from then on). The 10% margin, the token
counts and the 1.1 are not tuned on measurements.

`--preset single-fast` is `--preset single` with `--prefill-graph-tokens 64` and
`--prefill-graph-chunks 1-64`: when a prompt continues a cached one by at most 64 tokens (a resend,
a short next turn), those tokens run as one captured forward instead of an eager one, and the first
token comes sooner. A longer continuation runs eagerly, as under `single`.

Its replies are the text `single` gives. A captured forward computes what an eager forward of the
same tokens computes, bit for bit, and `single-fast` replays each chunk of up to 64 tokens whole,
never cut into pieces. Earlier versions of the preset also captured multiples of 64 up to 448 tokens
and cut a longer chunk into a multiple of 64 and the rest; a prompt computed in other pieces differs
in the last bits of its numbers, and where two tokens are nearly tied the reply takes another one.
That is gone: `--prefill-graph-tokens` and `--prefill-graph-chunks` typed on the command line still
allow it, and then the replies can differ from `single`'s.

The captures read the experts where they lie, so they need every expert on the GPU (resident, or
an expert cache with a slot for each outside the prefill staging buffers), and speculative decoding,
whose verify step they are captured beside. They take GPU memory, which the preset leaves free in
addition to the prefill chunk: 2.25 GiB, from 2.1 GiB measured for 70 sizes of Qwen3.6-35B-A3B
NVFP4 on an RTX 5090 (the preset now captures 64 of them), and not measured elsewhere. Where the plan with that memory left free does not
keep every expert resident and the KV reserve, or speculative decoding is off, the server runs as
`--preset single` and one log line says why. `--prefill-graph-tokens` or `--prefill-graph-chunks`
on the command line are kept as typed; typed chunks get no memory left free for them.

`--preset single-bucket` is `--preset single-fast` with `--prefill-graph-tokens 1024`,
`--prefill-graph-chunks 1-64,128,256,512,1024` and `--prefill-graph-pad`: a chunk of exactly 128,
256, 512 or 1,024 tokens (a unique prompt of that length, a next turn of that length) runs as one
captured forward, which writes the snapshot the prefix cache keeps, as the eager forward does. Any
other chunk past 64 tokens runs as one eager forward. A replay for a shorter chunk, padded to the
bucket, would run the matrix kernels at the bucket's row count, which cuBLAS can tile otherwise than
the eager forward of the chunk's tokens; served only for a chunk of its own size, the replay computes
what the eager forward computes, so the replies are `single`'s. The four captures took 170 MiB on an
RTX 5090; the preset leaves 2.5 GiB free for all of its captures and is dropped to `single` by the
same rules as `single-fast`. Where the experts are served from an expert cache, the chunks past 64
tokens run eagerly.

`--preset multi` is `--preset single` for several requests at a time. It sets
`--max-running-requests 8`; a typed number is kept (`--preset multi --max-running-requests 16`), and
the two settings below follow it. Everything else is `single`'s, chosen and dropped by the same
rules.

- `--kv-reserve-tokens` is a 4096-token prompt and as long an answer (the output limit, where
  `--max-output-tokens` is shorter) for each running request, 65536 tokens for 8, and never less
  than `single`'s reserve. It is at most one context for each running request.
- `--cuda-graph-bs` names every batch size up to 16, then the multiples of 8 and the number of
  running requests itself, so that a decode batch of 3, 5, 6 or 7 requests runs on a graph of its
  own size instead of the next captured one with padded rows. A typed `--cuda-graph-bs` or
  `--cuda-graph-max-bs` is kept. The graphs come out of the memory ratio's headroom; what the
  sizes above 8 take has not been measured.

Every running request takes GPU memory before the first token: a hybrid GatedDeltaNet model keeps 4
state slots for each and a snapshot cache of 2 more, and a slot of Qwen3.6-35B-A3B is 61 MiB
(counted from the layer's tensor shapes), so 8 requests hold 49 slots (2.9 GiB) and 16 hold 97
(5.8 GiB), against 9 slots under `single`. That memory counts where the preset decides whether
every expert fits, so the experts stay resident on fewer cards than under `single`, and with more
requests on fewer still; the plan line says which it is and how many KV tokens are left. A typed
`--moe-strategy fused` keeps the experts resident whatever the reserve; the plan line then warns
when the KV pool is smaller than the reserve.

The text of a reply depends on the requests decoded beside it: a decode step computes the rows of
every running request together, and a row's last bits change with the rows beside it. A seeded
request gives one text only when it runs alone (`--max-running-requests 1`).

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--reasoning-budget` | unset | Default `reasoning_budget` for chat requests: thinking tokens before the close tag is forced |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |
| `--enable-timings-report` | off | Attach a llama.cpp-shaped per-request `timings` object to responses |
| `--anthropic-inline-system` | auto | Placement of late Anthropic system instructions: `auto`, `preserve`, or `fold` |
| `--reproducible-seeds` | off | A request with a `seed` is computed without the prefix cache, so its text does not follow earlier requests; needs `--max-running-requests 1` (see below) |

With `--enable-timings-report` every chat completion, completion and Responses response
also carries a llama.cpp-shaped `timings` object next to `usage` (`cache_n`, `prompt_n`,
`prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
`predicted_per_token_ms`, `predicted_per_second`, plus `queue_ms` as a FreeToken
extension and `draft_n` / `draft_n_accepted` under speculative decoding), so frontends
like llama-swap and Open Web UI show per-request prefill/decode rates. All spans are
measured in the scheduler: `queue_ms` covers admission to the first prefill chunk,
`prompt_ms` the first chunk to the first sampled token (chunk gaps included),
`predicted_ms` the first to the last committed token; `prompt_n` counts only the
evaluated tokens and the decode rates divide by `predicted_n - 1`, as in llama.cpp. A chat
completion also splits the time before the queue into spans that follow each other:
`parse_ms` (arrival to the handler, the body read and parsed), `render_ms` (the chat
template), `to_worker_ms`, `tokenize_ms` (the tokenizer worker's encode),
`to_scheduler_ms` and `admit_ms`; a streamed one adds `reply_ms`, from the first sampled
token to the stream's first piece of text. On
streams the object rides the final chunk; a multi-prompt `/v1/completions` request
reports the first prompt's spans, as llama.cpp does. `/v1/messages` never carries it
(llama.cpp emits no timings there either). A streaming chat completion, completion or
Responses request may set the llama.cpp top-level body field `"timings_per_token": true`
to carry the block with growing values on every chunk that follows an ack: text and
reasoning deltas, tool-argument fragments, and the finish chunk (or the usage chunk
with `include_usage`). Without the server flag the field is ignored.

A reasoning model can think for thousands of tokens. A `/v1/chat/completions` request may
set `"reasoning_budget": N`: once the response has generated `N` tokens inside its thinking
span, the server commits the model's close tag (`</think>` for Qwen) as the next token in
place of the sampled one, and the answer is generated as usual. At 80% of the budget it
first forces the notice `(Reasoning budget almost used; conclude.)` into the thinking, once,
unless the budget leaves less than twice the notice's length after that point; the notice
counts toward the budget. `--reasoning-budget` sets the default for every chat protocol; a
request value wins, and `"reasoning_budget": 0` turns a server budget off for that request.
The thinking span is found by the reasoning parser's tag pair, so a budget needs a
`<think>`-style parser: a server on the `gpt_oss` or `muse_glimmer` parser, or with none,
refuses the flag at startup and answers such a request with a 400. Under speculative
decoding a forced token ends that step's accepted drafts. A `/v1/chat/completions` response
under a budget reports `usage.completion_tokens_details.reasoning_tokens`, and
`"reasoning_closed_by": "budget"` in `usage` when the server closed the thinking; on a
stream both ride the usage chunk (`"stream_options": {"include_usage": true}`).

A chat completion or completion request may carry an integer `seed`. Each sampled token is
then drawn from the seed and the token's position, so the draws repeat. The text can still
change, because the logits change in the last bit with what the server did before: a
prompt that reuses a cached prefix continues from stored state, a cold one computes it
again, and the two disagree where two tokens are nearly tied.

`--reproducible-seeds` takes that dependence away for requests that carry a seed. Such a
request reuses no cached prefix and leaves none behind, so its text follows from the model,
the server flags, the prompt, the sampling parameters and the seed, on the same GPU, driver
and build. What it costs:

- Every seeded request prefills its whole prompt. Its first token takes a cold prefill
  however much of the prompt is cached (`cached_tokens` is 0), and a later turn of the same
  conversation cannot reuse what it computed.
- `--max-running-requests` must be 1: a step's logits change with the rows beside it, so
  requests are served one at a time, seeded or not.
- Every expert must be computed on the GPU: `--moe-strategy offload` or `fused`. `hybrid`,
  `cpu` and `--moe-cpu-layers` are refused, because which experts the CPU computes follows
  what is resident and the CPU kernels round differently.
- `--speculative-algorithm dflash` is refused (its drafts draw from the stream all requests
  share). `mtp` works.

The server refuses to start on a combination from this list rather than serve text that
only sometimes repeats. Requests without a seed are served as without the flag and still
use the prefix cache. Text is not promised to match across GPUs, drivers, kernel builds,
speculative depths or other server flags.

### Image input

Experimental. Needs a checkpoint whose family registers a vision encoder ([models.md](models.md#image-input) lists them and how each one
maps the flags below); a request carrying images is rejected otherwise. Images are accepted on all three protocols (OpenAI `image_url`,
Anthropic `image` blocks, Responses `input_image`) as an http(s) URL or base64. Images inside a tool
result (an Anthropic `tool_result` block from Claude Code's Read, a Responses `function_call_output`
from Codex's view_image) are moved to the user turn that follows the tool message, as vLLM does,
because chat templates render tool messages as plain text.
`GET /v1/stats` reports what the server accepts as `model.input_modalities` (`["text"]` or `["text", "image"]`),
so a client can gate its attachment controls without reading the checkpoint config.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--text-model-only` | off | Serve a multimodal checkpoint text-only: no encoder tower is built (its VRAM goes to the KV/expert pools) and every multimodal input is rejected. Same as `--mm-disable` with every encoder kind |
| `--mm-disable` | none | Encoder towers to leave unbuilt (`vision`, `audio`); every input they would serve is rejected |
| `--mmproj` | found next to the model | GGUF models only: the llama.cpp vision tower file (`mmproj-*.gguf`), see [Image input](models.md#image-input). Without it, FreeToken takes a `*mmproj*.gguf` that fits the model (output width, tower type, image tokens) from the model file's own folder, BF16 before F32, F16 and quantized files. With none, or with files of several models it cannot tell apart, the model serves text only; a repo that keeps each quant in a subfolder and the mmproj at the top needs `--mmproj`. The log names the file it loads |
| `--mm-encoder-weights` | host | Where the encoder tower's block weights live. `host` streams them from pinned host banks two blocks at a time behind the compute, so the GPU holds two blocks instead of the whole tower; small images pay the copy time, large ones hide it behind the compute. `gpu` keeps them resident. An encoder without a block stack stays resident either way |
| `--image-min-tokens`, `--image-max-tokens` | processor defaults | Per-image token budget: the image processor resizes every image to take between these many tokens, converted to the family's own units by its processor. A family with fixed budgets honors the maximum only and refuses one below its smallest budget at start-up |
| `--mm-processor-kwargs` | none | JSON object of extra keyword arguments for the checkpoint's image processor call, for knobs the token budget does not cover; applied after the budget, so an explicit key wins |
| `--mm-embed-cache-device` | cpu | Where encoded image embeddings live between prefill chunks. `cpu` keeps them out of the VRAM budget; `cuda` skips the copy back |
| `--allowed-media-domains` | any | Comma-separated hostname allowlist for image URLs; requests for other domains are rejected with a 400. Empty allows any domain |
| `--allowed-local-media-path` | off | Directory `file://` image refs may be read from; unset rejects local files |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `--api-key KEY` (default `$FREETOKEN_API_KEY`) attaches to a server started with `--api-key`; with `--model` the shell uses that server's own key
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--api-key KEY] [--timeout 10] [--json] <subcommand>
```

`--api-key` defaults to `$FREETOKEN_API_KEY`.

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy, accepted input modalities |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.
When `/v1/stats` reports `image` among `model.input_modalities`, the written
config declares the model image-capable, which Codex, OpenCode, OpenClaw and
dsh require before their image tools and attachments send anything; Claude
Code and Hermes need no declaration.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--api-key KEY` | Key of a server started with `--api-key` (default `$FREETOKEN_API_KEY`). The agent gets it in its environment; Hermes and OpenClaw keep it in their config files |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--quant-backend moe.nvfp4=b12x] [--shard-gib 8] [--gpu <uuid-or-index>]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. MoE experts are
always packed into expert banks, which serve the fused and the offload
strategies alike (DeepSeek-V4 experts serve offload only);
`--moe-backend` is kept for compatibility. See the FTW caveats in
[models.md](models.md#notes); FTW files from older builds can be repaired with
[scripts/ftw_hotfix.py](ftw-hotfix.md) instead of reconverting.

## ft bench bw

```bash
ft bench bw --model /path/to/model.gguf   # every MoE layer kind of a checkpoint (GGUF or HF)
ft bench bw                               # canonical geometry per expert format
ft bench bw --gpu 1                       # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes one
row per MoE layer kind, which `ft serve --moe-strategy hybrid --moe-hybrid-max-fetch -1` reads
to split each layer's misses.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`; a profile from other
  hardware or another profile version is ignored.
- A row's key is the layer's CPU format and options, expert shape, top_k and CPU worker
  threads. A layer without an exact row takes the nearest row of the same format and top_k
  whose expert size is within 2x and whose thread count is within 25%, and the server logs
  which; without one it keeps a fetch cap of 1, or, with `FREETOKEN_HYBRID_DEFAULT_FETCH=<share>`
  (above 0, at most 1; off by default), fetches that share of the layer's misses. Bench with the
  same `--cpu-threads` as the server's `--moe-cpu-threads`.
- Each run replaces the rows it benched and keeps the others.
- What to measure: `--model` (checkpoint paths or presets), `--dtype`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets each row's `recommended`: hybrid when CPU bandwidth beats
  PCIe by that factor.

On NVLink-C2C hosts, see [models.md](models.md#nvlink-c2c-hosts-gh200--gb200)
for `FREETOKEN_H2D_BLOCKS_PER_BANK` if `PCIe-gather` sits well below the linear H2D ceiling.

## ft bench decode

```bash
ft bench decode --model /path/to/model                                # offload against hybrid, with the default flags
ft bench decode --model /path/to/model --cache-rates 0.05,0.2,0.5     # one comparison per cache rate
ft bench decode --model /path/to/model -- --moe-cpu-threads 8         # with the flags you serve with, after --
```

Which of `offload` and `hybrid` decodes faster depends on the machine, the model and the cache size
([performance/decode.md](performance/decode.md#hybrid-computing-misses-on-the-cpu)). This command measures it
through the real serving path: it starts `ft serve` once per arm (offload, hybrid with `min-wait`
rounding, hybrid with `stochastic` rounding, offload again), decodes four prompts one at a time with
thinking on and the checkpoint's sampling, then several prompts at once, and prints tok/s per arm
from the server's timings. It does this `--seeds` times per start (5 by default), and a column is
the mean over those passes with its 95% interval.

- A server seeds its sampler once, at start, so each pass decodes its own text and pass k is the
  same draws in every start. Without speculative decoding both arms also draw the same number at
  the same step; with it they share their draws until their texts first differ.
- Every number is of the timed window alone, `--decode` tokens (2,048) after `--warm` tokens
  (1,024), cut by token count. Next to tok/s the bench prints decode steps per second and tokens
  per step (with speculative decoding tok/s moves with the share of accepted drafts, steps per
  second does not), that share, and the rate and the 95th-percentile gap at which tokens reached
  the client. `--warm 0,256,1024,4096` also prints tok/s after each of those warm-ups; the verdict
  reads the longest.
- `--context 32768,65536,100000` adds, per start, one 300-token turn after a prompt of about that
  many tokens, and prints prefill seconds, seconds to the first token and decode tok/s. It is off
  by default and not part of the verdict.
- The prompts are an AIME-25 problem and a coding task (the workload columns), and two that each make
  the model repeat a sentence (the guard columns). Repeated output barely misses the cache, so hybrid
  gains nothing there and still pays its CPU round trip per layer.
- The verdict treats the passes as its samples, because the speed differs from text to text far
  more than between two starts. Two starts of one arm decode the same texts, so they are averaged
  pass by pass. For each workload column the bench prints hybrid's lead over offload and the lower
  bound of that lead, as `lead (at least bound)`: the bound is one-sided at 95%, from the
  differences between the passes of both arms (Welch's standard error, Student's t).
- Hybrid is reported as faster only if that lower bound is above zero on every workload column, the
  lead is at least 1.5% on its worst workload column, and it is no more than 10% behind on either
  guard column. With fewer than three passes there is no bound and hybrid cannot qualify. When it
  does not qualify, the bench prints which of these failed and on which column.
- If two starts of one arm are more than 5% apart over the same passes, the machine was not steady:
  nothing is decided and offload stays. When the lead is at least 1.5% but a lower bound is not
  above zero, the bench adds a round, up to `--rounds`: each hybrid arm and offload again, decoding
  the earlier passes and `--seeds` new ones. An arm whose server fails to start or stops
  answering, in any round, counts as not faster. The 1.5%, 5% and 10% are defaults of the tool, not
  measurements.
- Run `ft bench bw --model` first: hybrid takes its fetch split from that profile.
- Each server start loads the model and decodes about 123,000 tokens at the defaults (five passes of
  four prompts alone and of four at once, 3,072 tokens each), so the first round at one cache rate is
  four model loads. A lead that is not clear of zero adds up to three more per round, each with
  `--seeds` more passes than the round before, and `--cache-rates` runs the whole comparison once
  per rate. `--seeds`, `--warm` and `--decode` shorten a run.
- `--greedy`, `--streams`, `--aime`, `--prompts`, `--csv` and `--json` change what is decoded and
  where the numbers go. `--json` keeps the timed text and the finish reason of every request.
  `--server-cmd` starts the servers of another checkout.

The result is filed as one row per setup in `~/.cache/freetoken/benchdecode/<gpu-uuid>.json`
(`--no-write` only prints it). `ft serve --moe-strategy auto` takes `hybrid` when it finds a row that
says hybrid qualifies with the fetch rounding mode the server runs, and logs the measurement it used.
It uses a row only if all of this holds, and logs which part differs when it does not:

- the same checkpoint and the same flags: the cache size, `--moe-cpu-threads` and the CPUs the
  process may use, speculative decoding, `--max-running-requests`, every other engine setting that
  is not at its default, and the environment switches (`FREETOKEN_*`, and those of the kernels and
  libraries underneath, such as `CUDA_LAUNCH_BLOCKING`);
- the same machine and release: the GPU, the FreeToken version (a new release makes every row
  stale and the log says so; a commit between two releases does not), torch, triton, the NVIDIA
  driver (where `/proc/driver/nvidia/version` exists; not under WSL), the CPU model, the amount of
  RAM, the container's CPU quota and the `ft bench bw` profile;
- with an automatically sized cache, no more free GPU memory than when the row was measured (5%
  more slots at most). Hybrid then also runs with no more slots than it was measured with.

A row keeps the tok/s of every pass of every start, and `auto` applies the rule above to them. A
profile from a build whose rows held one number per column is not read, and the log says so.

`--preset single-measured` keeps its own timings in the same file, under its own key: `auto`
does not read them, and the preset does not read the rows of this command.

Without such a row `auto` stays on `offload` (`--preset single` may take `hybrid` there, see [Presets](#presets)). Rows are Linux only: the machine is read from `/proc`
and `/sys`, and where that cannot be done no row is filed and `auto` stays on `offload`.
Tensor-parallel servers are never picked from rows.
Hybrid computes some experts on the CPU, so where two tokens are nearly tied its output can differ
from offload's; and on output that barely misses the cache it can be up to the guard limit slower.
`--moe-strategy offload` keeps offload whatever was measured.
