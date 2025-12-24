#!/bin/bash

# Stress test script for Kimi-Audio model
# Iterates through different audio lengths and concurrency levels

BASE_URL="http://localhost:8000/v1"
MODEL="kimi_audio"
AUDIO_FILE="/home/ec2-user/SageMaker/efs/Projects/Kimi-Audio/test_audios/multiturn/case1/multiturn_a1.wav"
PROMPT="请识别电话沟通场景中如下声音片段的话轮转换意图，判断该片段是否包含明确的开始说话信号。请区分以下两种情况：若检测到清晰语音起始或强烈发言意愿（如语句开头、语气转折），应回复<是>；若仅含附和词（如\"嗯\"、\"yeah\"）、非语言声音（如喷嚏、咳嗽、笑声）、噪声或近似静默等非打断性信号，应回复<否>"
MAX_TOKENS=1
NUM_REQUESTS=500

# Create outputs directory
mkdir -p outputs

# Audio lengths and concurrency levels to test
AUDIO_LENGTHS=("200ms" "400ms" "600ms" "1s" "2s")
CONCURRENCIES=(1 5 10)

echo "=========================================="
echo "Starting Kimi-Audio Stress Test Suite"
echo "=========================================="
echo "Audio file: ${AUDIO_FILE}"
echo "Num requests per test: ${NUM_REQUESTS}"
echo "Max tokens: ${MAX_TOKENS}"
echo ""

for audio_length in "${AUDIO_LENGTHS[@]}"; do
    for concurrency in "${CONCURRENCIES[@]}"; do
        output_file="outputs/results_${audio_length}_c${concurrency}_prefixcache_warmup0.json"

        echo "=========================================="
        echo "Running test: audio_length=${audio_length}, concurrency=${concurrency}"
        echo "Output: ${output_file}"
        echo "=========================================="

        python batch_kimi/stress_test.py \
            --base-url "${BASE_URL}" \
            --model "${MODEL}" \
            --audio-file "${AUDIO_FILE}" \
            --audio-length "${audio_length}" \
            --prompt "${PROMPT}" \
            --max-tokens "${MAX_TOKENS}" \
            --concurrency "${concurrency}" \
            --num-requests "${NUM_REQUESTS}" \
            --streaming \
            --warmup 0 \
            --output "${output_file}"

        echo ""
        echo "Completed: ${output_file}"
        echo ""

        # Brief pause between tests
        sleep 2
    done
done

echo "=========================================="
echo "All tests completed!"
echo "Results saved in outputs/ directory"
echo "=========================================="
