#include "models.h"

void llama_model_zgcm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);
    if (n_pattern != 32 || hparams.n_layer() != 32) {
        throw std::runtime_error("zgcm: expected 32 layers and 32 sliding-window pattern entries");
    }
    const auto pattern_key = ml.llm_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN);
    const int64_t pattern_id = gguf_find_key(ml.metadata, pattern_key.c_str());
    if (gguf_get_arr_type(ml.metadata, pattern_id) != GGUF_TYPE_BOOL) {
        throw std::runtime_error("zgcm: sliding-window pattern must be an array of BOOL");
    }
    std::array<uint32_t, LLAMA_MAX_LAYERS> pattern = {};
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    const std::array<uint32_t, 32> expected = {
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0,
        1, 1,
    };
    if (hparams.n_swa != 128) {
        throw std::runtime_error("zgcm: sliding window must be 128");
    }

    std::string bits, global_layers, gated_layers;
    uint32_t n_local = 0;
    for (uint32_t i = 0; i < n_pattern; ++i) {
        if (pattern[i] != expected[i]) {
            throw std::runtime_error(format("zgcm: sliding-window pattern mismatch at layer %u", i));
        }
        if (hparams.n_head(i) != 32 || hparams.n_head_kv(i) != 8 || hparams.n_ff(i) != 11008) {
            throw std::runtime_error(format("zgcm: unsupported attention or FFN dimensions at layer %u", i));
        }
        bits += pattern[i] ? '1' : '0';
        auto & indices = pattern[i] ? gated_layers : global_layers;
        indices += (indices.empty() ? "" : ",") + std::to_string(i);
        n_local += pattern[i];
    }
    if (hparams.n_embd != 4096 || hparams.n_embd_head_k_full != 128 || hparams.n_embd_head_v_full != 128 ||
        hparams.n_embd_head_k_swa != 128 || hparams.n_embd_head_v_swa != 128 ||
        hparams.n_rot_full != 42 || hparams.n_rot_swa != 42 || hparams.f_norm_rms_eps != 1e-6f ||
        hparams.rope_freq_base_train != 1e7f || hparams.rope_freq_scale_train != 1.0f) {
        throw std::runtime_error("zgcm: unsupported embedding, normalization or RoPE parameters");
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    std::copy_n(pattern.begin(), n_pattern, hparams.is_swa_impl.begin());
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    type = LLM_TYPE_7B;

    LLAMA_LOG_INFO("zgcm: SWA window=%u; true=local; local_layers=%u; global_layers=%u\n", hparams.n_swa, n_local, n_pattern - n_local);
    LLAMA_LOG_INFO("zgcm: is_swa=%s\n", bits.c_str());
    LLAMA_LOG_INFO("zgcm: global_layers=[%s]\n", global_layers.c_str());
    LLAMA_LOG_INFO("zgcm: gated_layers=[%s]\n", gated_layers.c_str());
    LLAMA_LOG_INFO("zgcm: head_dim=%u; n_rot=%u; rope=NEOX; global_theta=%.0f; local_theta=%.0f\n",
            hparams.n_embd_head_k_full, hparams.n_rot_full, hparams.rope_freq_base_train, hparams.rope_freq_base_train_swa);
}

void llama_model_zgcm::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        const auto gate_name = tn(LLM_TENSOR_ATTN_GATE, "weight", i);
        if (hparams.is_swa(i)) {
            layer.wqkv_gate = create_tensor(gate_name, {n_embd, n_embd_head_k * n_head}, 0);
        } else if (ml.get_tensor_meta(gate_name.str().c_str()) != nullptr) {
            throw std::runtime_error(format("zgcm: unexpected attention gate on global layer %d", i));
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
    }
    if (ml.n_created != 382 || ml.n_tensors != 382) {
        throw std::runtime_error(format("zgcm: expected 382 bound tensors, created %d of %d", ml.n_created, ml.n_tensors));
    }
    LLAMA_LOG_INFO("zgcm: tensors_bound=%d; tensors_in_file=%d\n", ml.n_created, ml.n_tensors);
}

std::unique_ptr<llm_graph_context> llama_model_zgcm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_zgcm::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_STANDARD);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * attn_inp = cur;
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        if (hparams.is_swa(il)) {
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp);
            cb(gate, "attn_gate_proj", il);
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate", il);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);
        }

        cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
        cb(cur, "attn_out_proj", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up, nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
