#!/usr/bin/env python3
"""
Stress test script for testing the throughput of Kimi-Audio model.

Measures:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- Latency (end-to-end request time)
- Throughput (requests/sec, tokens/sec)
"""

import argparse
import asyncio
import base64
import io
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import soundfile as sf
import numpy as np


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    start_time: float
    ttft: Optional[float] = None  # Time to first token (seconds)
    latency: float = 0.0  # Total request time (seconds)
    output_tokens: int = 0
    input_tokens: int = 0
    success: bool = False
    error: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all requests."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency metrics (seconds)
    latencies: List[float] = field(default_factory=list)
    ttfts: List[float] = field(default_factory=list)

    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Timing
    total_duration: float = 0.0

    def add_request(self, metrics: RequestMetrics):
        self.total_requests += 1
        if metrics.success:
            self.successful_requests += 1
            self.latencies.append(metrics.latency)
            if metrics.ttft is not None:
                self.ttfts.append(metrics.ttft)
            self.total_input_tokens += metrics.input_tokens
            self.total_output_tokens += metrics.output_tokens
        else:
            self.failed_requests += 1

    def compute_statistics(self) -> dict:
        """Compute summary statistics."""
        stats = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests) * 100,
            "total_duration_sec": self.total_duration,
        }

        # Throughput
        if self.total_duration > 0:
            stats["requests_per_sec"] = self.successful_requests / self.total_duration
            stats["output_tokens_per_sec"] = self.total_output_tokens / self.total_duration
        else:
            stats["requests_per_sec"] = 0
            stats["output_tokens_per_sec"] = 0

        # Latency statistics
        if self.latencies:
            stats["latency_mean_sec"] = statistics.mean(self.latencies)
            stats["latency_median_sec"] = statistics.median(self.latencies)
            stats["latency_std_sec"] = statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0
            stats["latency_min_sec"] = min(self.latencies)
            stats["latency_max_sec"] = max(self.latencies)
            stats["latency_p50_sec"] = np.percentile(self.latencies, 50)
            stats["latency_p90_sec"] = np.percentile(self.latencies, 90)
            stats["latency_p95_sec"] = np.percentile(self.latencies, 95)
            stats["latency_p99_sec"] = np.percentile(self.latencies, 99)

        # TTFT statistics
        if self.ttfts:
            stats["ttft_mean_sec"] = statistics.mean(self.ttfts)
            stats["ttft_median_sec"] = statistics.median(self.ttfts)
            stats["ttft_std_sec"] = statistics.stdev(self.ttfts) if len(self.ttfts) > 1 else 0
            stats["ttft_min_sec"] = min(self.ttfts)
            stats["ttft_max_sec"] = max(self.ttfts)
            stats["ttft_p50_sec"] = np.percentile(self.ttfts, 50)
            stats["ttft_p90_sec"] = np.percentile(self.ttfts, 90)
            stats["ttft_p95_sec"] = np.percentile(self.ttfts, 95)
            stats["ttft_p99_sec"] = np.percentile(self.ttfts, 99)

        # TPOT (Time Per Output Token) - calculated from latency and output tokens
        if self.total_output_tokens > 0 and self.latencies:
            # TPOT = (latency - ttft) / output_tokens for each request
            tpots = []
            for i, (lat, ttft) in enumerate(zip(self.latencies, self.ttfts)):
                if self.total_output_tokens > 0:
                    # Estimate per-request output tokens (average)
                    avg_tokens = self.total_output_tokens / len(self.latencies)
                    if avg_tokens > 0:
                        tpot = (lat - ttft) / avg_tokens
                        tpots.append(tpot)

            if tpots:
                stats["tpot_mean_ms"] = statistics.mean(tpots) * 1000
                stats["tpot_median_ms"] = statistics.median(tpots) * 1000
                stats["tpot_p90_ms"] = np.percentile(tpots, 90) * 1000
                stats["tpot_p95_ms"] = np.percentile(tpots, 95) * 1000

        # Token statistics
        stats["total_input_tokens"] = self.total_input_tokens
        stats["total_output_tokens"] = self.total_output_tokens
        if self.successful_requests > 0:
            stats["avg_input_tokens_per_request"] = self.total_input_tokens / self.successful_requests
            stats["avg_output_tokens_per_request"] = self.total_output_tokens / self.successful_requests

        return stats


def parse_audio_length(length_str: str) -> float:
    """Parse audio length string to seconds.

    Supports formats like: 200ms, 500ms, 1s, 2s, 1.5s
    """
    length_str = length_str.strip().lower()

    if length_str.endswith("ms"):
        return float(length_str[:-2]) / 1000.0
    elif length_str.endswith("s"):
        return float(length_str[:-1])
    else:
        # Assume seconds
        return float(length_str)


def load_audio(audio_file: str) -> tuple:
    """Load audio file and return (data, sample_rate)."""
    data, sample_rate = sf.read(audio_file)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    return data, sample_rate


def extract_random_chunk(data: np.ndarray, sample_rate: int, length_sec: float) -> np.ndarray:
    """Extract a random chunk of specified length from audio data."""
    total_samples = len(data)
    chunk_samples = int(length_sec * sample_rate)

    if chunk_samples >= total_samples:
        # If requested length is longer than audio, return entire audio
        return data

    # Random start position
    max_start = total_samples - chunk_samples
    start = random.randint(0, max_start)

    return data[start:start + chunk_samples]


def audio_to_base64(data: np.ndarray, sample_rate: int) -> str:
    """Convert audio data to base64-encoded WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


async def send_request_streaming(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    audio_b64: str,
    prompt: str,
    request_id: int,
    max_tokens: int = 256,
) -> RequestMetrics:
    """Send a single streaming request and measure metrics."""
    metrics = RequestMetrics(request_id=request_id, start_time=time.time())

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ]
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        first_token_received = False
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                metrics.error = f"HTTP {response.status}: {error_text}"
                return metrics

            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Record TTFT on first token
                    if not first_token_received:
                        metrics.ttft = time.time() - metrics.start_time
                        first_token_received = True

                    # Count output tokens
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            # Rough token count (actual count from usage is more accurate)
                            metrics.output_tokens += 1

                    # Get usage info if available
                    if "usage" in data and data["usage"]:
                        metrics.input_tokens = data["usage"].get("prompt_tokens", 0)
                        metrics.output_tokens = data["usage"].get("completion_tokens", metrics.output_tokens)

                except json.JSONDecodeError:
                    continue

        metrics.latency = time.time() - metrics.start_time
        metrics.success = True

    except Exception as e:
        metrics.error = str(e)
        metrics.latency = time.time() - metrics.start_time

    return metrics


async def send_request_non_streaming(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    audio_b64: str,
    prompt: str,
    request_id: int,
    max_tokens: int = 256,
) -> RequestMetrics:
    """Send a single non-streaming request and measure metrics."""
    metrics = RequestMetrics(request_id=request_id, start_time=time.time())

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ]
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                metrics.error = f"HTTP {response.status}: {error_text}"
                return metrics

            data = await response.json()

            # For non-streaming, TTFT is approximately the full latency
            # (we get all tokens at once)
            metrics.latency = time.time() - metrics.start_time
            metrics.ttft = metrics.latency  # Best approximation for non-streaming

            # Get usage info
            if "usage" in data:
                metrics.input_tokens = data["usage"].get("prompt_tokens", 0)
                metrics.output_tokens = data["usage"].get("completion_tokens", 0)

            metrics.success = True

    except Exception as e:
        metrics.error = str(e)
        metrics.latency = time.time() - metrics.start_time

    return metrics


async def run_stress_test(
    base_url: str,
    model: str,
    audio_file: str,
    audio_length: str,
    concurrency: int,
    num_requests: int,
    prompt: str,
    max_tokens: int,
    streaming: bool,
) -> AggregatedMetrics:
    """Run the stress test with specified parameters."""

    # Parse audio length
    length_sec = parse_audio_length(audio_length)
    print(f"Audio chunk length: {length_sec:.3f}s ({audio_length})")

    # Load source audio
    print(f"Loading audio file: {audio_file}")
    audio_data, sample_rate = load_audio(audio_file)
    audio_duration = len(audio_data) / sample_rate
    print(f"Source audio duration: {audio_duration:.2f}s, sample rate: {sample_rate}Hz")

    if length_sec > audio_duration:
        print(f"Warning: Requested chunk length ({length_sec}s) > audio duration ({audio_duration:.2f}s)")
        print(f"Using full audio instead")
        length_sec = audio_duration

    # Prepare audio chunks
    print(f"Preparing {num_requests} audio chunks...")
    audio_chunks = []
    for i in range(num_requests):
        chunk = extract_random_chunk(audio_data, sample_rate, length_sec)
        audio_b64 = audio_to_base64(chunk, sample_rate)
        audio_chunks.append(audio_b64)

    # Setup
    url = f"{base_url}/chat/completions"
    aggregated = AggregatedMetrics()

    print(f"\nStarting stress test:")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Total requests: {num_requests}")
    print(f"  Streaming: {streaming}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompt: {prompt[:50]}...")
    print("-" * 60)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(session, request_id: int):
        async with semaphore:
            if streaming:
                return await send_request_streaming(
                    session, url, model, audio_chunks[request_id],
                    prompt, request_id, max_tokens
                )
            else:
                return await send_request_non_streaming(
                    session, url, model, audio_chunks[request_id],
                    prompt, request_id, max_tokens
                )

    # Run requests
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout per request

    start_time = time.time()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [limited_request(session, i) for i in range(num_requests)]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            metrics = await coro
            aggregated.add_request(metrics)
            completed += 1

            # Progress update
            if completed % 10 == 0 or completed == num_requests:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{num_requests} requests "
                      f"({rate:.2f} req/s), "
                      f"Success: {aggregated.successful_requests}, "
                      f"Failed: {aggregated.failed_requests}")

    aggregated.total_duration = time.time() - start_time

    return aggregated


def print_results(stats: dict, output_file: Optional[str] = None):
    """Print and optionally save results."""

    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)

    print("\n--- Request Summary ---")
    print(f"Total Requests:      {stats['total_requests']}")
    print(f"Successful:          {stats['successful_requests']}")
    print(f"Failed:              {stats['failed_requests']}")
    print(f"Success Rate:        {stats['success_rate']:.2f}%")
    print(f"Total Duration:      {stats['total_duration_sec']:.2f}s")

    print("\n--- Throughput ---")
    print(f"Requests/sec:        {stats.get('requests_per_sec', 0):.2f}")
    print(f"Output Tokens/sec:   {stats.get('output_tokens_per_sec', 0):.2f}")

    print("\n--- Latency (End-to-End) ---")
    if 'latency_mean_sec' in stats:
        print(f"Mean:                {stats['latency_mean_sec']*1000:.2f}ms")
        print(f"Median (P50):        {stats['latency_p50_sec']*1000:.2f}ms")
        print(f"P90:                 {stats['latency_p90_sec']*1000:.2f}ms")
        print(f"P95:                 {stats['latency_p95_sec']*1000:.2f}ms")
        print(f"P99:                 {stats['latency_p99_sec']*1000:.2f}ms")
        print(f"Min:                 {stats['latency_min_sec']*1000:.2f}ms")
        print(f"Max:                 {stats['latency_max_sec']*1000:.2f}ms")

    print("\n--- TTFT (Time to First Token) ---")
    if 'ttft_mean_sec' in stats:
        print(f"Mean:                {stats['ttft_mean_sec']*1000:.2f}ms")
        print(f"Median (P50):        {stats['ttft_p50_sec']*1000:.2f}ms")
        print(f"P90:                 {stats['ttft_p90_sec']*1000:.2f}ms")
        print(f"P95:                 {stats['ttft_p95_sec']*1000:.2f}ms")
        print(f"P99:                 {stats['ttft_p99_sec']*1000:.2f}ms")
        print(f"Min:                 {stats['ttft_min_sec']*1000:.2f}ms")
        print(f"Max:                 {stats['ttft_max_sec']*1000:.2f}ms")

    print("\n--- TPOT (Time Per Output Token) ---")
    if 'tpot_mean_ms' in stats:
        print(f"Mean:                {stats['tpot_mean_ms']:.2f}ms")
        print(f"Median:              {stats['tpot_median_ms']:.2f}ms")
        print(f"P90:                 {stats['tpot_p90_ms']:.2f}ms")
        print(f"P95:                 {stats['tpot_p95_ms']:.2f}ms")

    print("\n--- Token Statistics ---")
    print(f"Total Input Tokens:  {stats['total_input_tokens']}")
    print(f"Total Output Tokens: {stats['total_output_tokens']}")
    if 'avg_input_tokens_per_request' in stats:
        print(f"Avg Input/Request:   {stats['avg_input_tokens_per_request']:.1f}")
        print(f"Avg Output/Request:  {stats['avg_output_tokens_per_request']:.1f}")

    print("=" * 60)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Stress test for Kimi-Audio model throughput",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        default="kimi_audio",
        help="Model name as served by vLLM",
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        help="Path to source audio file (wav, mp3, etc.)",
    )
    parser.add_argument(
        "--audio-length",
        default="1s",
        help="Audio chunk length (e.g., 200ms, 500ms, 1s, 2s)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--prompt",
        default="请将音频内容转换为文字。",
        help="Prompt to send with each request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (required for accurate TTFT measurement)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Validate audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    # Run stress test
    aggregated = asyncio.run(run_stress_test(
        base_url=args.base_url,
        model=args.model,
        audio_file=args.audio_file,
        audio_length=args.audio_length,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        streaming=args.streaming,
    ))

    # Compute and print statistics
    stats = aggregated.compute_statistics()
    print_results(stats, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
