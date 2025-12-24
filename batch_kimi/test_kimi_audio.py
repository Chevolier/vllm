#!/usr/bin/env python3
"""
Simple test script for Kimi-Audio model using OpenAI-compatible API.
"""

import argparse
import base64
import json
import os

import requests
from openai import OpenAI


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


def load_audio_as_input_audio(audio_file: str) -> dict:
    """Load audio file and return input_audio format dict."""
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
    return {"data": audio_b64, "format": audio_format}


def test_kimi_audio(
    base_url: str,
    model: str,
    audio_file: str = None,
    audio_url: str = None,
    prompt: str = None,
    use_raw: bool = False,
    max_tokens: int = 256,
):
    """Send a single request to test Kimi-Audio model."""

    # Initialize OpenAI client (if not using raw mode)
    client = None
    if not use_raw:
        client = OpenAI(
            base_url=base_url,
            api_key="test",  # vLLM doesn't require real API key
        )

    # Prepare audio data
    input_audio = None
    audio_data_url = None

    if audio_url:
        audio_data_url = audio_url
        print(f"Using audio URL: {audio_url}")
    elif audio_file:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return
        input_audio = load_audio_as_input_audio(audio_file)
        print(f"Loaded audio file: {audio_file}")
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

    try:
        # Send request
        print("Sending request...")

        # Kimi-Audio uses token 151667 (<|im_kimia_text_eos|>) as text EOS
        # This must be included as a stop token for proper generation termination
        KIMIA_TEXT_EOS_TOKEN_ID = 151667

        if use_raw:
            # Send raw HTTP request for debugging
            url = f"{base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stop_token_ids": [KIMIA_TEXT_EOS_TOKEN_ID],
            }
            print(f"Raw request URL: {url}")
            print(f"Raw request payload (messages structure):")
            print(json.dumps(messages, indent=2, default=str)[:2000])
            print("-" * 60)

            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            print(f"HTTP Status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error response: {resp.text}")
            else:
                result = resp.json()
                print("Response:")
                print(json.dumps(result, indent=2))
        else:
            # Use OpenAI client
            # Use extra_body to pass vLLM-specific stop_token_ids parameter
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"stop_token_ids": [KIMIA_TEXT_EOS_TOKEN_ID]},
            )

            print("-" * 60)
            print("Response:")
            print(f"  ID: {response.id}")
            print(f"  Model: {response.model}")
            print(f"  Created: {response.created}")
            print(f"  Usage: {response.usage}")
            print("-" * 60)
            print("Content:")
            for choice in response.choices:
                print(f"  [{choice.index}] {choice.message.content}")
                print(f"      finish_reason: {choice.finish_reason}")
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

    args = parser.parse_args()

    test_kimi_audio(
        base_url=args.base_url,
        model=args.model,
        audio_file=args.audio_file,
        audio_url=args.audio_url,
        prompt=args.prompt,
        use_raw=args.raw,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
