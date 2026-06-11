# Python Fundamentals

## Variables and Types

Python is a dynamically typed language. Variables are created when you first assign a value to them, and their type is inferred from that value. You can rebind a variable to a value of a different type at any time.

Common built-in types include int, float, str, bool, list, tuple, dict, and set. Type hints can be added optionally using the syntax `name: type = value`, and tools like mypy will check those hints at static analysis time.

## Functions

Functions are first-class objects in Python. They can be assigned to variables, passed as arguments, returned from other functions, and stored in data structures. Functions are defined with the `def` keyword and can accept positional arguments, keyword arguments, default values, *args, and **kwargs.

A function without an explicit return statement returns None implicitly.

## Async and Concurrency

Python supports concurrency through three main models: threads, asyncio, and multiprocessing. Each is suited to different workloads.

### When to use asyncio

Asyncio is best for I/O-bound workloads with many concurrent operations, such as network calls, database queries, or file I/O. The event loop schedules coroutines cooperatively, so thousands of concurrent operations can be handled by a single thread with minimal overhead.

Coroutines are declared with `async def` and awaited with `await`. An `await` point is the only place where the event loop can switch to another coroutine — between awaits, code runs to completion.

### When to use threads

Threads carry per-thread memory overhead and operating-system context-switch costs. They are useful when you need to call into blocking C libraries that release the Global Interpreter Lock (GIL), or when integrating with libraries that don't support asyncio. For pure-Python I/O, asyncio is usually a better fit.

### When to use multiprocessing

For CPU-bound work, neither asyncio nor threads help in CPython because of the GIL — only one thread executes Python bytecode at a time. The multiprocessing module spawns separate Python processes, each with its own GIL, allowing real parallelism on multiple cores at the cost of higher memory usage and slower inter-process communication.

## Exception Handling

Exceptions in Python are raised with `raise` and caught with `try`/`except`. A bare `except:` clause catches all exceptions including system-exit signals — avoid it in favor of specific exception types. The `finally` clause runs whether or not an exception was raised, which makes it the right place for resource cleanup.

Context managers (the `with` statement) automate setup-and-teardown patterns. The standard library provides context managers for file handles, locks, network connections, and many other resources.
