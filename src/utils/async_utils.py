#!/usr/bin/env python3
"""
Supreme System V5 - Async Utilities

Async helper functions and utilities for concurrent operations.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def run_async(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a synchronous function asynchronously using ThreadPoolExecutor.

    Args:
        func: Synchronous function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, func, *args, **kwargs)


async def gather_with_exception_handling(coroutines: List[Coroutine[Any, Any, T]], timeout: Optional[float] = None) -> List[T]:
    """
    Gather coroutines with proper exception handling.

    Args:
        coroutines: List of coroutines to execute
        timeout: Optional timeout in seconds

    Returns:
        List of results (exceptions are logged but not raised)
    """
    results = []

    try:
        if timeout:
            results = await asyncio.wait_for(asyncio.gather(*coroutines, return_exceptions=True), timeout=timeout)
        else:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Handle exceptions in results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Coroutine {i} failed with exception: {result}")
                results[i] = None  # Replace exception with None

    except asyncio.TimeoutError:
        logger.error(f"Gather operation timed out after {timeout} seconds")
        results = [None] * len(coroutines)
    except Exception as e:
        logger.error(f"Gather operation failed: {e}")
        results = [None] * len(coroutines)

    return results


async def run_concurrent_with_limit(
    tasks: List[Callable[[], Coroutine[Any, Any, T]]], max_concurrent: int = 5, timeout: Optional[float] = None
) -> List[T]:
    """
    Run tasks concurrently with a limit on concurrent executions.

    Args:
        tasks: List of async task functions
        max_concurrent: Maximum number of concurrent tasks
        timeout: Optional timeout for the entire operation

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def run_task(task_func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        async with semaphore:
            try:
                return await task_func()
            except Exception as e:
                logger.error(f"Task failed: {e}")
                raise

    try:
        task_coroutines = [run_task(task) for task in tasks]

        if timeout:
            results = await asyncio.wait_for(asyncio.gather(*task_coroutines, return_exceptions=True), timeout=timeout)
        else:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

    except asyncio.TimeoutError:
        logger.error(f"Concurrent execution timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Concurrent execution failed: {e}")

    return results


async def retry_async(
    func: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of the function

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    raise last_exception


async def run_with_timeout(coroutine: Coroutine[Any, Any, T], timeout: float, default: Optional[T] = None) -> Optional[T]:
    """
    Run a coroutine with a timeout.

    Args:
        coroutine: Coroutine to run
        timeout: Timeout in seconds
        default: Default value to return on timeout

    Returns:
        Result of coroutine or default value
    """
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout} seconds")
        return default
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return default


async def batch_process_async(
    items: List[Any], process_func: Callable[[Any], Coroutine[Any, Any, T]], batch_size: int = 10, max_concurrent: int = 5
) -> List[T]:
    """
    Process items in batches with concurrency control.

    Args:
        items: List of items to process
        process_func: Async function to process each item
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent operations per batch

    Returns:
        List of results
    """
    all_results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} items")

        # Create tasks for this batch
        tasks = [lambda item=item: process_func(item) for item in batch]

        # Process batch concurrently
        batch_results = await run_concurrent_with_limit(tasks, max_concurrent)
        all_results.extend(batch_results)

    return all_results


class AsyncTaskManager:
    """
    Manager for async tasks with lifecycle management.
    """

    def __init__(self):
        self.tasks: List[asyncio.Task] = []
        self.running = False

    async def start_task(self, coroutine: Coroutine[Any, Any, Any], name: str = "") -> asyncio.Task:
        """
        Start a background task.

        Args:
            coroutine: Coroutine to run as task
            name: Optional task name for logging

        Returns:
            Created task
        """
        task = asyncio.create_task(coroutine, name=name)
        self.tasks.append(task)

        # Remove completed tasks
        self.tasks = [t for t in self.tasks if not t.done()]

        if name:
            logger.info(f"Started background task: {name}")

        return task

    async def stop_all_tasks(self):
        """Stop all running tasks."""
        logger.info(f"Stopping {len(self.tasks)} background tasks")

        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        self.tasks.clear()
        logger.info("All background tasks stopped")

    def get_active_tasks(self) -> List[asyncio.Task]:
        """Get list of currently active tasks."""
        return [task for task in self.tasks if not task.done()]

    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        return {
            "total_tasks": len(self.tasks),
            "active_tasks": len(self.get_active_tasks()),
            "completed_tasks": len([t for t in self.tasks if t.done() and not t.cancelled()]),
            "cancelled_tasks": len([t for t in self.tasks if t.cancelled()]),
            "failed_tasks": len([t for t in self.tasks if t.done() and not t.cancelled() and t.exception() is not None]),
        }


async def run_periodic_task(
    func: Callable[[], Coroutine[Any, Any, None]], interval: float, stop_event: Optional[asyncio.Event] = None
):
    """
    Run a function periodically until stopped.

    Args:
        func: Async function to run periodically
        interval: Interval between runs in seconds
        stop_event: Optional event to signal stopping
    """
    while stop_event is None or not stop_event.is_set():
        try:
            await func()
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Periodic task failed: {e}")
            await asyncio.sleep(interval)


async def run_with_deadline(coroutine: Coroutine[Any, Any, T], deadline: float) -> T:
    """
    Run a coroutine with a deadline (absolute time).

    Args:
        coroutine: Coroutine to run
        deadline: Absolute deadline timestamp

    Returns:
        Result of the coroutine

    Raises:
        asyncio.TimeoutError: If deadline is exceeded
    """
    current_time = time.time()
    timeout = max(0, deadline - current_time)

    return await asyncio.wait_for(coroutine, timeout=timeout)
