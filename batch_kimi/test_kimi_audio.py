#!/usr/bin/env python3
"""
Simple test script for Kimi-Audio model using OpenAI-compatible API.
"""

import argparse
import base64
import io
import json
import os
import time

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI


def parse_audio_length(length_str: str) -> float:
    """Parse audio length string to seconds.

    Supports formats like: 200ms, 500ms, 1s, 2s, 1.5s
    """
    if length_str is None:
        return None

    length_str = length_str.strip().lower()

    if length_str.endswith("ms"):
        return float(length_str[:-2]) / 1000.0
    elif length_str.endswith("s"):
        return float(length_str[:-1])
    else:
        # Assume seconds
        return float(length_str)


def load_audio_as_data_url(audio_file: str) -> str:
    """Load audio file and convert to data URL."""
    ext = os.path.splitext(audio_file)[1].lower()
    format_map = {
        '.pcm': 'pcm',
        '.raw': 'pcm',
        '.wav': 'wav',
        '.mp3': 'mp3',
    }
    audio_format = format_map.get(ext, 'pcm')

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/{audio_format};base64,{audio_b64}"


def load_audio_as_input_audio(audio_file: str, audio_length_sec: float = None) -> tuple:
    """Load audio file and return (input_audio format dict, actual_duration).

    Args:
        audio_file: Path to audio file
        audio_length_sec: If specified, extract a random chunk of this length (seconds)

    Returns:
        Tuple of (input_audio dict, actual_duration in seconds)
    """
    # Load audio using soundfile
    data, sample_rate = sf.read(audio_file)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    total_duration = len(data) / sample_rate

    # Extract chunk if audio_length specified
    if audio_length_sec is not None and audio_length_sec < total_duration:
        chunk_samples = int(audio_length_sec * sample_rate)
        # Random start position
        max_start = len(data) - chunk_samples
        start = np.random.randint(0, max_start + 1)
        data = data[start:start + chunk_samples]
        actual_duration = audio_length_sec
    else:
        actual_duration = total_duration

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format='WAV')
    buffer.seek(0)
    audio_bytes = buffer.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"data": audio_b64, "format": "wav"}, actual_duration


def test_kimi_audio(
    base_url: str,
    model: str,
    audio_file: str = None,
    audio_url: str = None,
    audio_length: str = None,
    prompt: str = None,
    use_raw: bool = False,
    max_tokens: int = 256,
    warmup: int = 0,
    repeat: int = 1,
):
    """Send a single request to test Kimi-Audio model."""

    # Initialize OpenAI client (if not using raw mode)
    client = None
    if not use_raw:
        client = OpenAI(
            base_url=base_url,
            api_key="test",  # vLLM doesn't require real API key
        )

    # Parse audio length
    audio_length_sec = parse_audio_length(audio_length)

    # Prepare audio data
    input_audio = None
    audio_data_url = None
    actual_duration = None

    if audio_url:
        audio_data_url = audio_url
        print(f"Using audio URL: {audio_url}")
    elif audio_file:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return
        input_audio, actual_duration = load_audio_as_input_audio(audio_file, audio_length_sec)
        print(f"Loaded audio file: {audio_file}")
        print(f"Audio duration: {actual_duration:.3f}s" + (f" (extracted from longer audio)" if audio_length_sec else ""))
    else:
        print("Error: Either --audio-file or --audio-url must be provided")
        return

    # Default prompt for audio understanding
    if prompt is None:
        prompt = "Please describe what you hear in this audio."

    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print("-" * 60)

    # Build messages using input_audio format (OpenAI standard)
    if input_audio:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "input_audio",
                        "input_audio": input_audio,
                    }
                ]
            }
        ]
    else:
        # Fallback to audio_url format for URLs
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_data_url},
                    }
                ]
            }
        ]

    # Debug: print message structure
    print("Request messages structure:")
    for i, msg in enumerate(messages):
        print(f"  Message {i}: role={msg['role']}")
        for j, part in enumerate(msg['content']):
            part_type = part.get('type', 'unknown')
            if part_type == 'text':
                text_preview = part.get('text', '')[:50]
                print(f"    Part {j}: type={part_type}, text='{text_preview}...'")
            elif part_type == 'input_audio':
                ia = part.get('input_audio', {})
                print(f"    Part {j}: type={part_type}, format={ia.get('format')}, data_len={len(ia.get('data', ''))}")
            elif part_type == 'audio_url':
                au = part.get('audio_url', {})
                url = au.get('url', '')
                print(f"    Part {j}: type={part_type}, url={url[:80]}...")
            else:
                print(f"    Part {j}: type={part_type}")
    print("-" * 60)

    # First, check if the model supports multimodal by querying the /models endpoint
    try:
        models_url = f"{base_url}/models"
        models_resp = requests.get(models_url)
        if models_resp.status_code == 200:
            models_data = models_resp.json()
            print("Available models:")
            for m in models_data.get("data", []):
                print(f"  - {m.get('id')}")
        print("-" * 60)
    except Exception as e:
        print(f"Warning: Could not fetch models: {e}")

    # Kimi-Audio uses token 151667 (<|im_kimia_text_eos|>) as text EOS
    # This must be included as a stop token for proper generation termination
    KIMIA_TEXT_EOS_TOKEN_ID = 151667

    def send_request(verbose=True):
        """Send a single request and return latency."""
        if use_raw:
            url = f"{base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stop_token_ids": [KIMIA_TEXT_EOS_TOKEN_ID],
                "spaces_between_special_tokens": False
            }
            if verbose:
                print(f"Raw request URL: {url}")
                print(f"Raw request payload (messages structure):")
                print(json.dumps(messages, indent=2, default=str)[:2000])
                print("-" * 60)

            start_time = time.time()
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            latency = time.time() - start_time

            if verbose:
                print(f"HTTP Status: {resp.status_code}")
                print(f"Latency: {latency*1000:.2f}ms")
                if resp.status_code != 200:
                    print(f"Error response: {resp.text}")
                else:
                    result = resp.json()
                    print("Response:")
                    print(json.dumps(result, indent=2))
            return latency
        else:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"stop_token_ids": [KIMIA_TEXT_EOS_TOKEN_ID]},
            )
            latency = time.time() - start_time

            if verbose:
                print("-" * 60)
                print("Response:")
                print(f"  ID: {response.id}")
                print(f"  Model: {response.model}")
                print(f"  Created: {response.created}")
                print(f"  Usage: {response.usage}")
                print(f"  Latency: {latency*1000:.2f}ms")
                print("-" * 60)
                print("Content:")
                for choice in response.choices:
                    print(f"  [{choice.index}] {choice.message.content}")
                    print(f"      finish_reason: {choice.finish_reason}")
            return latency

    try:
        # Warmup requests (silent)
        if warmup > 0:
            print(f"Running {warmup} warmup request(s)...")
            for i in range(warmup):
                warmup_latency = send_request(verbose=False)
                print(f"  Warmup {i+1}: {warmup_latency*1000:.2f}ms")
            print("-" * 60)

        # Main requests
        latencies = []
        for i in range(repeat):
            if repeat > 1:
                print(f"\n--- Request {i+1}/{repeat} ---")
            print("Sending request...")
            latency = send_request(verbose=True)
            latencies.append(latency)

        # Summary for repeated requests
        if repeat > 1:
            print("\n" + "=" * 60)
            print("LATENCY SUMMARY")
            print("=" * 60)
            print(f"  Requests:  {repeat}")
            print(f"  Min:       {min(latencies)*1000:.2f}ms")
            print(f"  Max:       {max(latencies)*1000:.2f}ms")
            print(f"  Mean:      {sum(latencies)/len(latencies)*1000:.2f}ms")
            if len(latencies) > 1:
                sorted_lat = sorted(latencies)
                median = sorted_lat[len(sorted_lat)//2]
                print(f"  Median:    {median*1000:.2f}ms")
            print("=" * 60)

        print("-" * 60)

    except Exception as e:
        print(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Test Kimi-Audio model with OpenAI-compatible API"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="kimi_audio",
        help="Model name as served by vLLM (default: kimi_audio)",
    )
    parser.add_argument(
        "--audio-file",
        default=None,
        help="Path to audio file (pcm, wav, mp3)",
    )
    parser.add_argument(
        "--audio-url",
        default=None,
        help="Audio URL (http:// or data: URL)",
    )
    parser.add_argument(
        "--audio-length",
        default=None,
        help="Audio chunk length to extract (e.g., 200ms, 500ms, 1s, 2s). If not specified, uses entire audio.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw HTTP request instead of OpenAI client (for debugging)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup requests to send before timing (default: 0)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the request for latency measurement (default: 1)",
    )

    args = parser.parse_args()

    test_kimi_audio(
        base_url=args.base_url,
        model=args.model,
        audio_file=args.audio_file,
        audio_url=args.audio_url,
        audio_length=args.audio_length,
        prompt=args.prompt,
        use_raw=args.raw,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
