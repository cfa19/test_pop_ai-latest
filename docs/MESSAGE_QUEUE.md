# Message Queue System

## Overview

The message queue system ensures messages are processed **sequentially** (one at a time) to prevent overwhelming CPU/GPU resources with concurrent model inference.

## Why Sequential Processing?

### Problem: Concurrent Model Inference

Without a queue, multiple concurrent requests can:

1. **GPU Memory Overflow**
   - Each DistilBERT classifier call loads model weights into GPU memory
   - Concurrent calls: 10 requests × 400MB = 4GB GPU memory
   - Result: Out of memory errors, crashes

2. **CPU Overload**
   - Semantic gate uses sentence-transformers (CPU-intensive)
   - Translation uses langdetect + Google Translate
   - Concurrent calls overwhelm CPU cores
   - Result: Slow response times, server unresponsiveness

3. **Model Loading Race Conditions**
   - Multiple requests trying to load the same model simultaneously
   - Can cause file lock errors or memory corruption
   - Result: Inconsistent behavior, crashes

### Solution: Sequential Processing

**Message queue ensures:**
- ✅ **One message processed at a time**
- ✅ **GPU memory usage stays constant** (1× model size)
- ✅ **CPU usage is manageable** (no concurrent overload)
- ✅ **Models loaded once** (singleton + queue)
- ✅ **Predictable response times** (no resource contention)

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Startup Event                                           │
│    └─→ start_message_queue()                            │
│         └─→ MessageQueue.start()                        │
│              └─→ Worker Task (background)               │
│                                                          │
│  Chat Endpoint (/chat)                                  │
│    ├─→ Authenticate user                                │
│    ├─→ Get message queue                                │
│    ├─→ Submit message: queue.process_message()         │
│    │    ├─→ Generate request_id                        │
│    │    ├─→ Create asyncio.Future                      │
│    │    ├─→ Add to queue                               │
│    │    └─→ Wait for result (await future)            │
│    └─→ Return response                                  │
│                                                          │
│  Worker Task (background loop)                          │
│    ├─→ Get message from queue (sequential)             │
│    ├─→ Run workflow (one at a time)                    │
│    │    ├─→ Language detection                         │
│    │    ├─→ Intent classifier (GPU/CPU)                │
│    │    ├─→ Semantic gate (CPU)                        │
│    │    ├─→ RAG retrieval                              │
│    │    └─→ Response generation                        │
│    └─→ Set result on future                            │
│                                                          │
│  Shutdown Event                                         │
│    └─→ stop_message_queue()                            │
│         └─→ MessageQueue.stop()                        │
│              └─→ Cancel worker, reject pending         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Flow Diagram

```
Request 1 arrives → Queue → Worker processes → Response 1
Request 2 arrives → Queue ─┐
Request 3 arrives → Queue ─┤
Request 4 arrives → Queue ─┤
                            └→ Wait in queue

Worker finishes Request 1 → Get Request 2 from queue → Process → Response 2

Worker finishes Request 2 → Get Request 3 from queue → Process → Response 3

...and so on
```

## Implementation

### Message Queue (`src/utils/message_queue.py`)

```python
class MessageQueue:
    """
    Async message queue for sequential workflow processing.

    Ensures only one message is processed at a time.
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.worker_task: asyncio.Task | None = None
        self.is_running = False

    async def process_message(self, workflow_func, **kwargs) -> Any:
        """
        Submit message for processing and wait for result.

        1. Generate unique request_id
        2. Create Future to wait for result
        3. Add to queue
        4. Wait for worker to process
        5. Return result
        """
        request_id = str(uuid.uuid4())
        result_future = asyncio.Future()
        self.pending_requests[request_id] = result_future

        await self.queue.put({
            "request_id": request_id,
            "workflow_func": workflow_func,
            "kwargs": kwargs
        })

        # Wait for result (blocks until worker processes this request)
        result = await result_future
        return result

    async def _worker(self):
        """
        Background worker that processes messages sequentially.

        Loop:
        1. Get next message from queue
        2. Process it (await workflow_func(**kwargs))
        3. Set result on future
        4. Repeat
        """
        while self.is_running:
            message_data = await self.queue.get()

            request_id = message_data["request_id"]
            workflow_func = message_data["workflow_func"]
            kwargs = message_data["kwargs"]

            result_future = self.pending_requests.get(request_id)

            try:
                # Process message (sequential - one at a time)
                result = await workflow_func(**kwargs)
                result_future.set_result(result)
            except Exception as e:
                result_future.set_exception(e)
```

### Integration with FastAPI

**main.py:**
```python
from src.utils.message_queue import start_message_queue, stop_message_queue

@app.on_event("startup")
async def startup_event():
    """Start message queue on server startup."""
    await start_message_queue()
    logger.info("Message queue initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop message queue on server shutdown."""
    await stop_message_queue()
    logger.info("Message queue stopped")
```

**src/api/chat.py:**
```python
from src.utils.message_queue import get_message_queue

@router.post("/chat")
async def chat(request_body: ChatRequest, request: Request):
    # ... authentication ...

    # Submit to queue (instead of calling run_workflow directly)
    message_queue = get_message_queue()

    workflow_state = await message_queue.process_message(
        workflow_func=run_workflow,
        message=request_body.message,
        user_id=user_id,
        # ... other params ...
    )

    # ... process result ...
```

## Singleton Pattern for Models

### DistilBERT Classifier

**Already implemented** in `src/agents/distilbert_classifier.py`:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_distilbert_classifier(model_path: str) -> DistilBertIntentClassifier:
    """
    Get or create cached DistilBERT classifier.

    Loaded once and reused for all requests.
    """
    return DistilBertIntentClassifier(model_path)

class DistilBertIntentClassifier:
    def _load_model(self):
        """Lazy load model (only once)."""
        if self._model is not None:
            return  # Already loaded

        # Load model into GPU/CPU memory
        self._tokenizer, self._model = load_from_local_path(...)
        self._model.eval()
```

**Result:**
- Model loaded once on first request
- Cached for all subsequent requests
- No repeated disk I/O or memory allocation

### Semantic Gate

**Already implemented** in `src/agents/semantic_gate.py`:

```python
_semantic_gate_instance = None

def get_semantic_gate(force_reload: bool = False) -> SemanticGate:
    """
    Get or create the global semantic gate instance (singleton).
    """
    global _semantic_gate_instance

    if _semantic_gate_instance is None or force_reload:
        _semantic_gate_instance = SemanticGate()

    return _semantic_gate_instance
```

**Result:**
- SentenceTransformer loaded once
- Category centroids loaded once
- Reused for all requests

## Performance Comparison

### Without Queue (Concurrent Processing)

```
10 concurrent requests arrive simultaneously:

Request 1: Load model (2s) + Inference (0.5s) = 2.5s
Request 2: Load model (2s) + Inference (0.5s) = 2.5s
Request 3: Load model (2s) + Inference (0.5s) = 2.5s
...
Request 10: Load model (2s) + Inference (0.5s) = 2.5s

GPU Memory: 10 × 400MB = 4GB (CRASH if > GPU memory)
CPU Usage: 10 × 100% = 1000% (system overload)
Total Time: ~2.5s (but system crashes or becomes unresponsive)
```

### With Queue (Sequential Processing)

```
10 requests arrive, processed sequentially:

Request 1: Load model (2s first time) + Inference (0.5s) = 2.5s
Request 2: Inference (0.5s, model already loaded) = 0.5s
Request 3: Inference (0.5s) = 0.5s
...
Request 10: Inference (0.5s) = 0.5s

GPU Memory: 1 × 400MB = 400MB (constant, safe)
CPU Usage: 1 × 100% = 100% (manageable)
Total Time: 2.5s + 9×0.5s = 7s (all requests succeed)

Average latency per request: 7s / 10 = 0.7s
```

**Trade-off:**
- ❌ Slightly higher latency per request (0.7s vs 2.5s)
- ✅ System remains stable (no crashes)
- ✅ All requests succeed (no OOM errors)
- ✅ Predictable performance

## Monitoring

### Queue Metrics

```python
from src.utils.message_queue import get_message_queue

queue = get_message_queue()

# Current queue size (waiting messages)
queue_size = queue.get_queue_size()

# Pending requests (being processed or waiting)
pending_count = queue.get_pending_count()

print(f"Queue size: {queue_size}")
print(f"Pending: {pending_count}")
```

### Logging

The queue system logs important events:

```
[INFO] Starting Pop Skills AI API...
[INFO] Message queue initialized
[INFO] [MESSAGE QUEUE] Started worker

[INFO] [CHAT] Submitting message to queue (queue size: 0)
[DEBUG] [MESSAGE QUEUE] Queued request abc123... (queue size: 1)
[DEBUG] [MESSAGE QUEUE] Processing request abc123... (queue size: 0)
[DEBUG] [MESSAGE QUEUE] Completed request abc123...
[INFO] [CHAT] Message processed (queue size: 0)

[INFO] Shutting down Pop Skills AI API...
[INFO] [MESSAGE QUEUE] Stopping worker...
[INFO] [MESSAGE QUEUE] Stopped
[INFO] Message queue stopped
```

## Testing

### Basic Test

```bash
# Start server
python main.py

# Send 10 concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <token>" \
    -d '{"message": "Hello"}' &
done

# Check logs - should see sequential processing
```

### Load Test

```python
import asyncio
import aiohttp

async def send_message(session, message):
    async with session.post(
        "http://localhost:8000/chat",
        json={"message": message},
        headers={"Authorization": "Bearer <token>"}
    ) as response:
        return await response.json()

async def load_test(num_requests=100):
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_message(session, f"Message {i}")
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        print(f"Completed {len(results)} requests")

asyncio.run(load_test(100))
```

## Configuration

### Environment Variables

```bash
# Enable debug logging for queue
export LOG_LEVEL=DEBUG

# Start server
python main.py
```

### Queue Timeout

Currently using 1-second timeout for queue polling. Can be adjusted in `src/utils/message_queue.py`:

```python
async def _worker(self):
    while self.is_running:
        try:
            message_data = await asyncio.wait_for(
                self.queue.get(),
                timeout=1.0  # Adjust this value
            )
```

## Best Practices

1. **Always use the queue for ML model inference**
   - Intent classification (DistilBERT or OpenAI)
   - Semantic gate (SentenceTransformer)
   - Any GPU/CPU-intensive operations

2. **Don't bypass the queue**
   - Always call `message_queue.process_message(run_workflow, ...)`
   - Never call `run_workflow(...)` directly in endpoints

3. **Monitor queue size**
   - If queue size grows indefinitely, consider:
     - Adding more workers (advanced)
     - Optimizing model inference time
     - Scaling horizontally (multiple servers)

4. **Graceful shutdown**
   - FastAPI shutdown event ensures:
     - Worker is cancelled
     - Pending requests are rejected
     - Resources are cleaned up

## Future Improvements

1. **Multiple Workers**
   - Add N workers instead of 1
   - Each worker processes messages in parallel
   - Still prevents overload (limited to N concurrent)

2. **Priority Queue**
   - Prioritize certain users or message types
   - VIP users get faster responses

3. **Timeout Handling**
   - Automatically timeout requests that take too long
   - Prevent one slow request from blocking others

4. **Queue Metrics Dashboard**
   - Real-time monitoring of queue size
   - Request processing times
   - Error rates

5. **Distributed Queue**
   - Use Redis or RabbitMQ for multi-server deployment
   - Scale horizontally across multiple machines

## Summary

The message queue system provides:

- ✅ **Sequential processing** - One message at a time
- ✅ **Resource protection** - No GPU/CPU overload
- ✅ **Model singleton** - Load once, reuse forever
- ✅ **Graceful handling** - Proper startup/shutdown
- ✅ **Production-ready** - Logging, monitoring, error handling

**Result:** Stable, predictable performance even under high load.
