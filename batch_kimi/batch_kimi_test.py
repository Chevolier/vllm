
import os
import librosa

from dataclasses import asdict
from transformers import AutoTokenizer
from vllm.transformers_utils.processors import KimiAudioProcessor
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

messages = [
    {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
    {"role": "user", "message_type": "audio", "content": None}
]


def parse_args():
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=5,
        help="Maximum number of concurrent decode slots.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per output sequence",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = "moonshotai/Kimi-Audio-7B-Instruct"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    engine_args.limit_mm_per_prompt = default_limits | dict(
        engine_args.limit_mm_per_prompt or {}
    )
    llm = LLM(**asdict(engine_args))

    inputs = []
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = KimiAudioProcessor(text_tokenizer=tokenizer)
    files = [item for item in os.listdir(args.input_dir) if item.endswith('.wav')]
    files.sort()
    for file in files:
        filepath = os.path.join(args.input_dir, file)
        messages[1]["content"] = librosa.load(filepath, sr=16000)[0]
        prompts = processor.get_prompt(messages, output_type="text")
        prompt_token_ids = prompts.get("prompt_token_ids")[0]
        multi_modal_data = prompts.get("multi_modal_data")
        inputs.append({
            "multi_modal_data": multi_modal_data,
            "prompt_token_ids": prompt_token_ids
        })

    stop_token_ids = [tokenizer._convert_token_to_id('<|im_kimia_text_eos|>')]
    sampling_params = SamplingParams(
        temperature=0.0, top_k=1, repetition_penalty=1.0,
        max_tokens=args.max_tokens, stop_token_ids=stop_token_ids
    )
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    main()
