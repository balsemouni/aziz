"""
test_tts_microservice.py
tests for Piper TTS WebSocket server (services/tts/main.py).
Protocol:
  Client->server: UTF-8 JSON  {text, voice, language}
  Server->client: binary PCM frames + empty b'' sentinel
"""
import asyncio, json, os, sys, time

WS_HOST = os.getenv('TTS_HOST', 'localhost')
WS_PORT = int(os.getenv('TTS_PORT', '8765'))
WS_URL  = f'ws://{WS_HOST}:{WS_PORT}/ws/tts'
TEST_TEXT   = os.getenv('TTS_TEST_TEXT', 'Hello, this is a test of the Piper TTS microservice.')
TEST_VOICE  = os.getenv('TTS_VOICE', 'tara')
SAMPLE_RATE = int(os.getenv('TTS_SAMPLE_RATE', '22050'))

_passed = 0
_failed = 0

def ok(label, detail=''):
    global _passed; _passed += 1
    print(f'  PASS  {label}' + (f'  ({detail})' if detail else ''))

def fail(label, detail=''):
    global _failed; _failed += 1
    print(f'  FAIL  {label}' + (f'  ({detail})' if detail else ''))

def warn(label, detail=''):
    print(f'  WARN  {label}' + (f'  ({detail})' if detail else ''))

def pcm_duration_ms(pcm, sr=SAMPLE_RATE):
    return (len(pcm) // 2 / sr) * 1000.0

async def synthesize(text, voice=None, timeout=15.0):
    import websockets
    payload = {'text': text}
    if voice: payload['voice'] = voice
    pcm_chunks = []; first_latency = 0.0; first = False; t0 = 0.0
    async with websockets.connect(WS_URL, open_timeout=5.0) as ws:
        t0 = time.monotonic()
        await ws.send(json.dumps(payload))
        async for msg in ws:
            if isinstance(msg, bytes):
                if not first: first_latency = (time.monotonic()-t0)*1000; first = True
                if len(msg) == 0: break
                pcm_chunks.append(msg)
    return b''.join(pcm_chunks), first_latency

async def test_basic_synthesis():
    print('\n--- TEST 1: Basic synthesis')
    try:
        pcm, latency = await synthesize(TEST_TEXT, voice=TEST_VOICE)
        ok('PCM received', f'{len(pcm):,} bytes') if pcm else fail('PCM received', '0 bytes')
        dur = pcm_duration_ms(pcm)
        ok('Duration > 0', f'{dur:.0f} ms') if dur > 0 else fail('Duration > 0')
        ok('First-frame latency', f'{latency:.0f} ms')
    except Exception as exc: fail('Basic synthesis', str(exc))

async def test_empty_text_guard():
    print('\n--- TEST 2: Empty-text guard')
    try:
        import websockets
        async with websockets.connect(WS_URL, open_timeout=5.0) as ws:
            await ws.send(json.dumps({'text': ''}))
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(msg, bytes) and len(msg) == 0:
                ok('Empty sentinel received for empty text')
            else:
                fail('Empty sentinel received for empty text', repr(msg))
    except Exception as exc: fail('Empty-text guard', str(exc))

async def test_voice_override():
    print('\n--- TEST 3: Voice override')
    try:
        pcm, _ = await synthesize(TEST_TEXT, voice='lessac')
        if pcm: ok('PCM with voice=lessac', f'{len(pcm):,} bytes')
        else: warn('PCM with voice=lessac', '0 bytes - voice may not be installed')
    except Exception as exc: fail('Voice override', str(exc))

async def test_latency():
    print('\n--- TEST 4: Round-trip latency')
    try:
        _, latency = await synthesize(TEST_TEXT, voice=TEST_VOICE)
        if latency <= 2000: ok('First-frame latency <= 2 s', f'{latency:.0f} ms')
        else: warn('First-frame latency > 2 s', f'{latency:.0f} ms')
    except Exception as exc: fail('Latency test', str(exc))

async def main():
    print('=' * 60)
    print(f'  Piper TTS WebSocket test  |  {WS_URL}')
    print('=' * 60)
    try:
        import websockets
    except ImportError:
        print('ERROR: pip install websockets'); sys.exit(1)
    try:
        import websockets
        async with websockets.connect(WS_URL, open_timeout=5.0): pass
    except Exception as exc:
        print(f'\nERROR: Cannot connect to {WS_URL}: {exc}')
        print('  Start TTS:  cd services/tts && uvicorn main:app --port 8765')
        sys.exit(1)
    await test_basic_synthesis()
    await test_empty_text_guard()
    await test_voice_override()
    await test_latency()
    print('\n' + '=' * 60)
    print(f'  Results: {_passed} passed / {_failed} failed')
    print('=' * 60)
    sys.exit(0 if _failed == 0 else 1)

if __name__ == '__main__':
    asyncio.run(main())
