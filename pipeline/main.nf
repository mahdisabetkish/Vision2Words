#!/usr/bin/env nextflow
// End-to-end pipeline: download data, cache encoder features, train all
// three decoders in parallel, evaluate each, build the comparison report,
// export decoders A/B to ONNX, and benchmark the C++ port against Python.
//
// Every process body is just the same `v2w` CLI used manually throughout
// this project -- the pipeline's job is orchestration (what runs in
// parallel, what waits for what, where files land), not reimplementing any
// of the actual ML code.
nextflow.enable.dsl = 2

def epochOverride = params.epochs ? "--set train.epochs=${params.epochs} --set train.early_stopping_patience=${params.epochs} --set data.min_word_freq=1" : ''

process DOWNLOAD_OR_LINK_DATA {
    label 'python_stage'
    publishDir params.data_dir, mode: 'copy', overwrite: false

    output:
    path 'raw', emit: raw_dir

    script:
    """
    v2w download-data --raw-dir raw
    """
}

process EXTRACT_FEATURES {
    label 'python_stage'
    publishDir params.data_dir, mode: 'copy', overwrite: true

    input:
    path raw_data

    output:
    path 'features', emit: feature_dir

    script:
    def limitFlag = params.max_images ? "--max-images ${params.max_images}" : ''
    """
    v2w extract-features --raw-dir ${raw_data} --feature-dir features \
        --device ${params.device} --batch-size 32 --num-workers 2 ${limitFlag}
    """
}

process TRAIN {
    label 'python_stage'
    tag "${model_name}"
    publishDir "${params.checkpoints_dir}/${model_name}", mode: 'copy', overwrite: true

    input:
    tuple val(model_name), path(config_file)
    path features
    path raw_data

    output:
    tuple val(model_name), path('checkpoint'), emit: checkpoints

    script:
    """
    v2w train --config ${config_file} \
        --set data.feature_dir=${features} \
        --set data.raw_dir=${raw_data} \
        --set train.ckpt_dir=checkpoint \
        --set train.device=${params.device} \
        ${epochOverride}
    """
}

process EVALUATE {
    label 'python_stage'
    tag "${model_name}"
    publishDir params.results_dir, mode: 'copy', overwrite: true, saveAs: { "${model_name}.json" }

    input:
    tuple val(model_name), path(checkpoint_dir)
    path features
    path raw_data

    output:
    tuple val(model_name), path('result.json'), emit: result

    script:
    """
    v2w evaluate --checkpoint ${checkpoint_dir}/best.pt \
        --raw-dir ${raw_data} --feature-dir ${features} \
        --split test --device ${params.device} --beam-size ${params.beam_size} \
        --out result.json
    """
}

process BUILD_REPORT {
    label 'python_stage'
    publishDir params.results_dir, mode: 'copy', overwrite: true

    input:
    path raw_data
    path features
    path lstm_result
    path lstm_attention_result
    path transformer_result
    tuple val(_a), path(lstm_ckpt)
    tuple val(_b), path(lstm_attention_ckpt)
    tuple val(_c), path(transformer_ckpt)

    output:
    path 'metrics.json'
    path 'comparison.md'
    path 'sample_captions.png'
    path 'attention_maps.png'

    script:
    """
    mkdir -p results_in
    cp ${lstm_result} results_in/lstm_baseline.json
    cp ${lstm_attention_result} results_in/lstm_attention.json
    cp ${transformer_result} results_in/transformer.json
    v2w build-report --raw-dir ${raw_data} --feature-dir ${features} --results-dir results_in \
        --lstm-checkpoint ${lstm_ckpt}/best.pt \
        --lstm-attention-checkpoint ${lstm_attention_ckpt}/best.pt \
        --transformer-checkpoint ${transformer_ckpt}/best.pt \
        --device ${params.device}
    mv results_in/metrics.json results_in/comparison.md results_in/sample_captions.png results_in/attention_maps.png .
    """
}

process EXPORT_ONNX {
    label 'python_stage'
    publishDir params.export_dir, mode: 'copy', overwrite: true

    input:
    path features
    tuple val(_a), path(lstm_attention_ckpt)
    tuple val(_b), path(transformer_ckpt)

    output:
    path '*'

    script:
    """
    v2w export-onnx --feature-dir ${features} \
        --lstm-attention-checkpoint ${lstm_attention_ckpt}/best.pt \
        --transformer-checkpoint ${transformer_ckpt}/best.pt \
        --out-dir . --device cpu
    """
}

process BUILD_CPP {
    label 'cpp_stage'

    output:
    path 'cpp_bin', emit: cpp_bin

    script:
    """
    bash ${params.cpp_dir}/scripts/fetch_third_party.sh
    cmake -S ${params.cpp_dir} -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    mkdir cpp_bin
    cp build/v2w_caption cpp_bin/ 2>/dev/null || cp build/Release/v2w_caption.exe cpp_bin/ || cp build/v2w_caption.exe cpp_bin/
    cp build/*.dll cpp_bin/ 2>/dev/null || true
    cp build/*.so* cpp_bin/ 2>/dev/null || true
    """
}

process CPP_BENCHMARK {
    label 'cpp_stage'
    publishDir params.results_dir, mode: 'copy', overwrite: true

    input:
    path export_dir
    path raw_data
    path features
    path cpp_bin
    tuple val(_a), path(lstm_attention_ckpt)
    tuple val(_b), path(transformer_ckpt)

    output:
    path 'cpp_report.md'
    path 'cpp_parity.json'

    script:
    """
    binary=\$(ls ${cpp_bin}/v2w_caption ${cpp_bin}/v2w_caption.exe 2>/dev/null | head -n 1)
    python ${params.cpp_dir}/../scripts/verify_cpp_parity.py \
        --raw-dir ${raw_data} --feature-dir ${features} --model-dir ${export_dir} --cpp-build-dir ${cpp_bin} \
        --lstm-attention-checkpoint ${lstm_attention_ckpt}/best.pt \
        --transformer-checkpoint ${transformer_ckpt}/best.pt \
        --num-images 5 --beam-size ${params.beam_size} --out cpp_parity.json

    {
        echo "# C++ inference: parity and benchmark (pipeline run)"
        echo
        echo "See cpp_parity.json for per-image results. Latency (mean of 20 runs, one image):"
        echo
        first_image=\$(ls ${raw_data}/images | head -n 1)
        echo '```'
        "\${binary}" ${raw_data}/images/\${first_image} --model-dir ${export_dir} --decoder lstm_attention --benchmark 20
        "\${binary}" ${raw_data}/images/\${first_image} --model-dir ${export_dir} --decoder transformer --benchmark 20
        echo '```'
    } > cpp_report.md
    """
}

process REPORT {
    label 'python_stage'
    publishDir params.results_dir, mode: 'copy', overwrite: true

    input:
    path comparison_md
    path cpp_report_md

    output:
    path 'run_summary.md'

    script:
    """
    {
        echo "# Vision2Words pipeline run"
        echo
        echo "Generated \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
        cat ${comparison_md}
        echo
        cat ${cpp_report_md}
    } > run_summary.md
    """
}

workflow {
    configs_ch = Channel.of(
        tuple('lstm_baseline', file(params.lstm_config)),
        tuple('lstm_attention', file(params.lstm_attention_config)),
        tuple('transformer', file(params.transformer_config)),
    )

    raw_data = DOWNLOAD_OR_LINK_DATA()
    features = EXTRACT_FEATURES(raw_data)
    checkpoints = TRAIN(configs_ch, features, raw_data)

    results = EVALUATE(checkpoints, features, raw_data)

    // Pick each named entry explicitly rather than relying on channel order,
    // so a re-ordering upstream can't silently swap which checkpoint a
    // downstream process treats as which model.
    lstm_ckpt = checkpoints.checkpoints.filter { it[0] == 'lstm_baseline' }
    attn_ckpt = checkpoints.checkpoints.filter { it[0] == 'lstm_attention' }
    tf_ckpt = checkpoints.checkpoints.filter { it[0] == 'transformer' }

    lstm_result = results.result.filter { it[0] == 'lstm_baseline' }.map { it[1] }
    attn_result = results.result.filter { it[0] == 'lstm_attention' }.map { it[1] }
    tf_result = results.result.filter { it[0] == 'transformer' }.map { it[1] }

    report_out = BUILD_REPORT(raw_data, features, lstm_result, attn_result, tf_result, lstm_ckpt, attn_ckpt, tf_ckpt)
    export_out = EXPORT_ONNX(features, attn_ckpt, tf_ckpt)
    cpp_bin = BUILD_CPP()
    cpp_out = CPP_BENCHMARK(export_out, raw_data, features, cpp_bin, attn_ckpt, tf_ckpt)

    REPORT(report_out[1], cpp_out[0])
}
