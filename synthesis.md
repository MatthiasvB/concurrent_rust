# Tutorial on Concurrent Rust: Harnessing the Power of Async and Parallelism

![Ferris in a factory weaving yarn]('./ferris_factory.png')

Rust makes writing programs that do more than one thing at a time quite easy, while ensuring that the code we use to do so is free of memory errors. However, writing concurrent programs in Rust may require using unstable features and external crates. In this article, we'll explore two such crates for different purposes, giving you the tools you need to make your Rust code concurrent.

## Understanding Concurrency

_The definitions that follow are debatable. They are given here mainly as a reference for concepts explored in this article._

To grasp concurrent programming in Rust, we first need to distinguish between **concurrency**, **asynchronicity** and **parallelism**:

- **Concurrency** involves executing tasks without a predefined order. To do so, one may want to ensure that the code can potentially run in parallel and is memory-safe for such execution.
- **Asynchronous (Async) Work:** Think of asynchronous work as tasks that can be done without a predetermined sequence, similar to eating and drinking. You eat when you're hungry and drink when you're thirsty, and the order doesn't matter. Async work is ideal for I/O-bound operations and tasks with frequent waiting.
- **Parallel Workload:** This involves tasks that can be done in parallel, such as summing up multiple numbers. The order of execution doesn't affect the result, making it suitable for parallel operations. However, whether parallel execution actually occurs depends on the software configuration, the operating system and the availability of multiple CPU cores.

In essence, while we can design code to be concurrent, actual parallel execution depends on factors like the runtime, operating system, and hardware. Async workload is concurrent without necessarily running in parallel.

## Running Concurrent Workloads

Imagine your CPU as a set of tools on a construction site, with each tool representing a logical CPU core. The tasks you need to perform are like recipes or instructions. However, the workers (threads) are somewhat limited – they can follow instructions precisely but can't multitask effectively by themselves.

### Three Categories of Tasks

1. **Heavy Computation:** Some tasks involve long periods of active work, such as digging a deep hole. In these cases, it's best to match the number of workers to the available tools.

2. **Blocking Calls:** Certain tasks require tools for short durations, like painting a wall with multiple layers, waiting for the paint to dry between iterations. In this scenario, a worker may hold a tool, blocking others from using it. Optimizing these tasks is crucial.

3. **Interrupted Operations:** To improve efficiency, tasks that typically block can be rewritten to switch to different tasks at predefined points. This is where `async`/`await` constructs come into play, allowing for better work scheduling within your program.

## Understanding Threads

In computing, a thread represents a line of execution of commands within a program. Threads can run independently and are like workers on a construction site. The key distinction is who manages the allocation of resources to these workers.

### Kernel Level Threads

When the operating system controls thread scheduling, we call them kernel-level threads or OS threads. While the OS is efficient at scheduling threads, we, as developers, can often optimize scheduling within our program because we have more context.

### Green Threads

Green threads, also known as user-level threads, exist within our program and are managed by our code. They are lighter in terms of resource usage compared to OS threads. Green threads are ideal for cases where multiple tasks need to flexibly switch without true parallelism.

## Understanding Runtimes

Concurrent work would be straightforward if not for two challenges:

1. Running compute-heavy tasks might hog resources needed for other critical tasks, impacting responsiveness.
2. Kernel-level threads are expensive to create and manage.

Async runtimes address these issues by splitting work into green threads and creating a limited number of kernel-level threads, distributing work among them. Different runtimes optimize different types of workloads, including heavy computation, blocking calls (I/O), and async work.

### Tokio

Tokio is optimized for managing tasks like web servers. It's designed to be highly responsive and can handle blocking operations like file system access. Tokio manages two thread pools: one for async work distribution and another for blocking operations. While the latter is useful, it's better style to use non-blocking APIs whenever possible.

### Rayon

Rayon is ideal for compute-heavy tasks, such as simulation software. It maximizes CPU utilization and efficiency. Rayon's parallel iterator traits make it easy to parallelize iterative work.

## Sample Code

Below are code snippets demonstrating different approaches to concurrency in Rust.

### No runtime

We should probably discuss multithreading without a runtime first, because it doesn't require external crates and is conceptually simpler.

```rust
use std;

const NUMBER_OF_THREADS: usize = 3;
const FIBUNACCI_LOAD: usize = 42;

/// Very inefficient fibunacci implementation to put some load on the CPU
fn fibunacci(num: usize) -> usize {
    match num {
        0 => 0,
        1 => 1,
        _ => fibunacci(num - 2) + fibunacci(num - 1),
    }
}

fn work() {
    fibunacci(FIBUNACCI_LOAD);
}

/// Pretend we have to do some CPU bound computation
fn work_then_square(thread_no: usize, to_be_squared: usize) -> usize {
    println! {"Compute thread {thread_no} starts to do some work"}
    work();
    let res = to_be_squared.pow(2);
    println! {"Compute thread {thread_no} done. Result is {res}"};
    res
}

/// Use an explicit amount of kernel level threads to calculate results.
/// The main thread does not get those results, they are only printed to 
/// the console from the worker threads
fn explicit_threads(number_of_threads: usize) {
    let join_handles: Vec<std::thread::JoinHandle<()>> = (0..number_of_threads).map(|num: usize| {
        // spawn thread, return reference to join handle
        std::thread::spawn(move || {
            work_then_square(num, num);
        })
    }).collect();

    // wait for all threads to finish
    join_handles.into_iter().for_each(|thread| {
        thread.join().unwrap();
    });
}

fn main() {
    explicit_threads(NUMBER_OF_THREADS);
}
```

If each thread has the purpose to produce a single result, we can modify this to

```rust
fn explicit_threads_result(number_of_threads: usize) {
    let join_handles: Vec<std::thread::JoinHandle<usize>> = (0..number_of_threads).map(|num: usize| {
        std::thread::spawn(move || {
            work_then_square(num, num)
        })
    }).collect();

    let result: Vec<usize> = join_handles.into_iter().map(|thread| {
        thread.join().unwrap()
    }).collect();

    println!("{:?}", result);
}
```

### Using tokio

With tokio, the focus is async work, not heavy computation.

#### Standard thread pool

Any time `.await` is called, we give tokio the chance to put the green thread to sleep and exchange it with another one that can then run on the worker thread.

```rust
use tokio;

const NUMBER_OF_THREADS: usize = 3;

/// Delay the current green thread without blocking the kernel level thread
async fn sleep_for(seconds: u64) {
    tokio::time::sleep(std::time::Duration::from_secs(seconds)).await;
}

/// Pretend to do short bursts of work and sleep the green thread in between
async fn delay_thread_async_then_square(thread_no: usize, to_be_squared: usize) -> usize {
    let mut wait = 3;
    println!("Async thread {thread_no} sleeping for {wait} seconds");
    sleep_for(wait).await;
    wait = 7;
    println!("Async thread {thread_no} sleeping for {wait} seconds");
    sleep_for(wait).await;
    wait = 1;
    println!("Async thread {thread_no} sleeping for {wait} seconds");
    sleep_for(wait).await;
    wait = 4;
    println!("Async thread {thread_no} sleeping for {wait} seconds");
    sleep_for(wait).await;
    let res = to_be_squared.pow(2);
    println!("Async thread {thread_no} done. Result is {res}");
    res
}

/// Create the tokio runtime and start the work
fn with_tokio_no_communication(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Execute a Future
    rt.block_on(tokio_no_communication(number_of_threads));
}

/// Spawn a number of green threads that sleep often
async fn tokio_no_communication(number_of_threads: usize) {
    let tasks = (0..number_of_threads)
        // `spawn` takes a future as an argument. It will run this future on a
        // separate green thread
        .map(|num| tokio::task::spawn(delay_thread_async_then_square(num, num)));
    futures::future::join_all(tasks).await;
}

fn main() {
    with_tokio_no_communication(NUMBER_OF_THREADS);
}
```

#### Blocking thread pool

Sometimes we may have no choice but to make a call that will block the kernel level thread. To avoid blocking one of tokio's worker threads, we spawn such threads on a separate threadpool:

```rust
use tokio;

const NUMBER_OF_THREADS: usize = 3;

/// This will actually block the current kernel level thread
fn block_thread_for(seconds: u64) {
    std::thread::sleep(std::time::Duration::from_secs(seconds));
}

/// Pretend we do short bursts of work between multiple blocking calls
fn delay_thread_sync_then_square(thread_no: usize, to_be_squared: usize) -> usize {
    let mut wait = 4;
    println!("Blocking thread {thread_no} blocked for {wait} seconds");
    block_thread_for(wait);
    wait = 7;
    println!("Blocking thread {thread_no} blocked for {wait} seconds");
    block_thread_for(wait);
    wait = 1;
    println!("Blocking thread {thread_no} blocked for {wait} seconds");
    block_thread_for(wait);
    wait = 4;
    println!("Blocking thread {thread_no} blocked for {wait} seconds");
    block_thread_for(wait);
    let res = to_be_squared.pow(2);
    println!("Blocking thread {thread_no} done. Result is {res}");
    res
}

/// Create the tokio runtime and start tasks
fn with_tokio_blocking(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(tokio_blocking(number_of_threads));
}

async fn tokio_blocking(number_of_threads: usize) {
    let futures = (0..number_of_threads).map(|num| {
        // We pass a closure instead of a future to `spawn_blocking`
        // That's because blocking work is not expected to be async / return a future
        tokio::task::spawn_blocking(move || {
            let res = delay_thread_sync_then_square(num, num);
            res
        })
    });

    let result: Vec<usize> = futures::future::join_all(futures)
        .await
        .into_iter()
        .collect::<Result<Vec<usize>, _>>()
        .unwrap();
    println!("{:?}", result);
}

fn main() {
    with_tokio_blocking(NUMBER_OF_THREADS);
}
```

### Using rayon

Rayon has a threadpool designed to efficiently parallelize compute heavy tasks.

#### Manually spawning tasks

Rayon allows us to spawn tasks in a similar fashion to kernel level threads and tokio green threads. However, it returns neither a join handle, nor a future. We lose all control over spawned threads. Or rather, if we need control, we have to add it ourselves.

The following approach uses a oneshot channel provided by tokio, which is a channel that will only deliver a value once, to return the task's result _and_ in this case signal that it has run to completion.

```rust
use rayon::prelude::*;
use tokio;

const NUMBER_OF_THREADS: usize = 3;
const FIBUNACCI_LOAD: usize = 42;

/// Very inefficient fibunacci implementation to put some load on the CPU
fn fibunacci(num: usize) -> usize {
    match num {
        0 => 0,
        1 => 1,
        _ => fibunacci(num - 2) + fibunacci(num - 1),
    }
}

fn work() {
    fibunacci(FIBUNACCI_LOAD);
}

/// Pretend we have to do some CPU bound computation
fn work_then_square(thread_no: usize, to_be_squared: usize) -> usize {
    println! {"Compute thread {thread_no} starts to do some work"}
    work();
    let res = to_be_squared.pow(2);
    println! {"Compute thread {thread_no} done. Result is {res}"};
    res
}

fn with_compute_rayon(number_of_threads: usize) {
    futures::executor::block_on(compute_rayon(number_of_threads));
}

async fn compute_rayon(number_of_threads: usize) {
    let receivers = (0..number_of_threads).map(|num| {
        let (send, recv) = tokio::sync::oneshot::channel();
        // We can use a closure here, but we can't get its result
        rayon::spawn(move || {
            let res = work_then_square(num, num);
            let _ = send.send(res);
        });
        // We do return the receiver, which will allow us to wait for the result
        recv
    });
    let result = futures::future::join_all(receivers)
        .await
        .into_iter()
        .collect::<Result<Vec<usize>, _>>()
        .unwrap();
    println!("{:?}", result);
}
```

### Using parallel iteratiors

Rayon provides a method to parallelize work that is incredibly easy to use. By providing a trait for parallel iterators, parallelization may become as easy as replacing `iter()` with `par_iter()`:

```rust
fn parallel_iterators(number_of_threads: usize) {
    let result: Vec<usize> = (0..number_of_threads)
        .into_par_iter() // <- parallelization here!
        .map(|num: usize| {
            let res = work_then_square(num, num);
            res
        })
        .collect();

    println!("{:?}", result);
}
```

## Sharing state and getting results

We have seen that futures and join handles allow us to get the result that a given task produces. We have also seen that this does not work with rayon.

Getting a single result may not always be enough. Maybe we need to get data from a thread multiple times while it is running. Or maybe our threads need to read and/or write shared state. There are two main recipes than can be fairly universally applied.

But why is this even a problem? Basically, because it is very easy to create bugs when multiple threads try to work with the same values at the same time. As long as everybody is just reading, there is no problem. But as soon as reads overlap with writes, or multiple writes happen at the same time, weird things are bound to happen. We must make sure this can't occur to avoid very hard to find and reproduce bugs.

### Mutexes

"Mutex" stands for "mutual exclusion". Think of it as a box that contains a value. Anybody who wants access needs to lock-the-box-and-get-the-value, which is a single operation. We can then read and modify the value, and anybody else who tries just finds a locked box. He will have to wait for the box to become unlocked before he can continue.

Let's look at an example

```rust
// not a full runnable example

async fn tokio_mutex(number_of_threads: usize) {
    // we pass a Mutex to the callback
    // it is wrapped in an Arc (Atomic reference counter) to make it thread-safe
    async fn cb(index: usize, num: usize, arr: std::sync::Arc<std::sync::Mutex<Vec<usize>>>) {
        let res = delay_thread_async_then_square(num, num).await;
        // we (un)lock the mutex
        let mut numbers = arr.lock().unwrap();
        // and modify it
        numbers[index] = res;
        // it is freed when `numbers` is dropped (goes out of scope)
    }

    // create shared state
    let vec: Vec<usize> = (0..number_of_threads).into_iter().collect();
    // wrap it into Arc<Mutex<_>>
    let numbers = std::sync::Arc::new(std::sync::Mutex::new(vec));

    let tasks = (0..number_of_threads).map(|num| {
        // clone the reference
        let numbers = numbers.clone();
        // and pass it to the task
        tokio::task::spawn(cb(num, num, numbers))
    });
    futures::future::join_all(tasks).await;

    // to print the result, we also need to lock the Mutex
    let result = numbers.lock().unwrap();
    println!("{:?}", result);
}
```

Mutexes are neat, because they are fairly easy to wrap your head around. It's shared state. Yes, you need to do this locking thing, but that's not much of a mental overhead.

There's a problem, though. Mutexes introduce the possibility to dead-lock your code. Consider the following scenario

- ThreadA has Mutex1 locked but also needs Mutex2
- ThreadB has Mutex2 locked but also needs Mutex1

ThreadA will keep Mutex1 locked and will block until it has gotten Mutex2 as well. ThreadB will do the same with Mutex2. None of them will ever release the lock on their Mutex, and neither will get a chance to lock the other Mutex. That's a dead-lock. Your program is now frozen.

There is no safe method to ensure that this will never happen in your code as it evolves other than ensuring that any thread will always only lock a single Mutex at a time.

But even that may introduce a significant overhead, if a lot of threads are frequently interested in the same Mutex. They will often block each other, making your code run slower.

That is one reason why the language Go, which is aimed at making concurrency extremely easy, has decided that this is not a good idea. Instead, it uses channels to pass around messages. Instead of holding shared state to mutate it, it would send a message to a central location that instructs another thread that holds the state on how to modify it. We can do the same in Rust.

### Channels

Channels allow us to pass messages between threads. Typically, there can be many senders, but only one receiver. But variations exist.

To modify state, we don't do the modifications in separate threads. Instead, we send messages that contains instructions on how the state needs to be updated to a single thread. Since the update is always performed by the same thread, there is no need to synchronize them.

```rust
fn with_tokio_channels(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // tokio_channels is not async. It's blocking, due to the way it iterates
    // received messages. That's why we don't need the `block_on` call.
    // But we still need to enter the context of the tokio runtime. This is how
    let _guard = rt.enter();

    tokio_channels(number_of_threads);
}

fn tokio_channels(number_of_threads: usize) {
    // The callback gets a sender
    async fn cb(index: usize, num: usize, sender: std::sync::mpsc::Sender<(usize, usize)>) {
        let res = delay_thread_async_then_square(num, num).await;
        sender.send((index, res)).unwrap();
    }

    let (sender, receiver) = std::sync::mpsc::channel::<(usize, usize)>();

    // _Not_ shared state
    let mut vec: Vec<usize> = (0..number_of_threads).into_iter().collect();

    (0..number_of_threads).for_each(|num| {
        // make a clone of the sender for each task
        let sender = sender.clone();
        tokio::task::spawn(cb(num, num, sender));
    });
    // need to drop the sender, because the iterator below 
    // will only complete once all senders are dropped
    drop(sender);

    // iterate all messages from the receiver and modify state accordingly
    receiver.iter().for_each(|(index, res)| {
        vec[index] = res;
    });

    println!("{:?}", vec);
}
```

We can avoid blocking while waiting for messages by using tokio's async channels

```rust
fn with_tokio_async_channels(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(tokio_async_channels(number_of_threads));
}

async fn tokio_async_channels(number_of_threads: usize) {
    async fn cb(index: usize, num: usize, sender: tokio::sync::mpsc::Sender<(usize, usize)>) {
        let res = delay_thread_async_then_square(num, num).await;
        sender.send((index, res)).await.unwrap();
    }

    let (sender, mut receiver) = tokio::sync::mpsc::channel::<(usize, usize)>(number_of_threads);
    let mut vec: Vec<usize> = (0..number_of_threads).into_iter().collect();

    (0..number_of_threads).for_each(|num| {
        let sender = sender.clone();
        tokio::task::spawn(cb(num, num, sender));
    });
    // need to drop the sender, because the iteration below will only complete once all senders are dropped
    drop(sender);

    while let Some((index, res)) = receiver.recv().await {
        vec[index] = res;
    }

    println!("{:?}", vec);
}
```

## Learning more

You can explore [this repositorie's](https://github.com/MatthiasvB/concurrent_rust) `src` directory for more (runnable) snippets on different variations of runtimes and state-sharing mechanisms. Of course, it'd also be a good idea to look at the docs for tokio and rayon. A lot information I listed here was inspired by [this great blog post](https://ryhl.io/blog/async-what-is-blocking/).

## It's not perfect

This article is meant to give you a starting point for concurrency in Rust. While concurrent Rust code is guaranteed not to have memory errors, there is no guarantee that you'll be able to model your problem in Rust. I'm only starting out on the topic myself, but I've heard from people who have tried to do harder things than I have that the language is "just not ready, yet", especially when it comes to concurrency. The team is working hard to resolve these issues, but for now, a few important building blocks simply do not exist, are unstable, or very complicated to use. Beware of that.
