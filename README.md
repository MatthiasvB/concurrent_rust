# Concurrent Rust

Rust makes writing programs that do more than one thing at a time quite easy, but ensuring that the code we use to do so is free of memory errors. However, not all aspects of writing such programs is thoroughly rooted in the language and standard library, yet. So we may have to resort to unstable features and external crates. But that's not a big problem. In this article, we take a look at two such crates, which are used for different purposes.

## What is concurrency?

_The definitions that follow are debatable. They are given here mainly as a reference for concepts explored in this article._

Let's define concurrent work as work that is done in no predefined order. It is important to differentiate "concurrency" and "parallelism". Parallelism is more specific, meaning work that is done at the same time as other work.

Why is this important?

Because, technically, we have no way of ensuring that our code really does run in parallel. All we can do is write it so that it _can_ run concurrently and is (memory) safe to run in parallel. The way we schedule the concurrent work (for example using an async runtime) decides whether our program can potentially be run in parallel. If this is the case, the runtime's configuration, the operating system and the hardware the program runs on determines if any of it will _actually_ run in parallel.

That being said, runtimes like tokio exist to parallelize our code as efficiently as possible, and the basics of how to do this right are the topic of this article.

## What are types of concurrency?

An example for **asynchronous** work is eating and drinking. Sometimes you gotta eat, and sometimes you gotta drink. The order does not matter. You eat when you're hungry and you drink when you're thirsty. Generally, you don't do both at the same time.

What then, is **parallel** workload?

There are things that are fundamentally sequential, like counting up a counter. There is no way to raise the counter from 5 to 6 before it has been raised from 4 to 5. On the other hand, there are things that can easily be done in parallel, like summing up multiple numbers: 3 + 4 + 5 + 6 is the same as (3 + 4) + (5 + 6). The order in which these additions are done has no effect on the result. So we could do 3 + 4 at the same time as 5 + 6, and then finish up with 7 + 11.

Eating and drinking can easily run on single CPU core, if the program running on it dynamically switches between the operations "eat" and "drink".

The second example can also run on a single CPU core, if the program running on it switches from "3 + 4" to "5 + 6" and then to "7 + 11". But, it could also run on two cores, if it outsources "3 + 4" to the second core, does "5 + 6" and then waits for the second core to produce its result and then do "7 + 11". This program might benefit from such parallelization.

Let's distinguish these two cases of "concurrency" as "async" and "parallel".

## How to run concurrent workload?

Let's imagine our CPU is like a set of tools on a construction site. If you have 8 (logical) CPU cores, that's like having 8 tools on the site.

Then there are the things that need to be done. Let's imagine that each thing is described by a recipe called a "task" of consecutive instructions.

To get the work done, you can hire as many workers as you like. Problem is, they are really dumb and can only very precisely follow the instructions that make up a task. To work, they need a tool. For many tasks, you can hire many workers, but only as many as you have tools can work at the same time. The more you have, the more difficult it will become to fairly distribute "tool time" between them.

There are three categories of tasks that you can issue, that fundamentally change the way your workers will behave.

### Heavy computation

One type of task is when your worker will be actively working for a really long time, like digging a very deep hole. He'll just be doing that for hours on end.

### Blocking calls

Another one is when when a worker needs a tool only for a short time, like when painting a wall with multiple layers of paint. He has to start with the first layer, and then wait for it to dry, before the next layer can be applied. If he holds on to the paintbrush for that duration, he's blocking other workers from using it.

### Interrupted operations

It might be smarter if the worker did something else while he's waiting for the paint to dry. If your task is explicitly written to instruct the worker to switch to a different task at predefined breakpoints, a single worker can get a lot more work done. Tasks that would usually keep the worker idle for significant amounts of time can often be rewritten to engage him differently when he would otherwise be waiting. This is what the `async` / `await` constructs in Rust do.

When we describe a task that would usually block in a way that allows it to be interrupted so other work can be done, we enable a good amount of work scheduling to be performed within our own program.

### Dealing with different types of tasks

Let's map our construction site metaphor to the computing domain:

- "tool" = "CPU core"
- "worker" = "(kernel level) thread" (a line of consecutive instructions as viewed by the OS)
- "(interruptible) task" = "(green) thread" (a line of consecutive instructions as viewed by our program)

#### Heavy work tasks

When we are dealing with the heavy computation type of task, the best way to get work done fast is to get as many tools as possible and the same amount of workers. If we only have 8 tools, there is no point in getting 9 workers. If you have more than 8 tasks, you wait for workers to finish their task and then hand them the next one.

#### Interruptable tasks

When we have tasks that require a lot of waiting while preparing to paint the next layer on the wall, it isn't that important how many tools or workers we have. It's much more important to split up the tasks into small chunks, so that workers can often change the job they are working on. Instead of

- paint wall A &hyphen; wait &hyphen; paint wall A &hyphen; wait &hyphen; paint wall A
- paint wall B &hyphen; wait &hyphen; paint wall B &hyphen; wait &hyphen; paint wall B
- paint wall C &hyphen; wait &hyphen; paint wall C &hyphen; wait &hyphen; paint wall C

we instruct them to

- paint wall A
- paint wall B
- paint wall C
- if wall A dry, paint it
- if wall B dry, paint it
- if wall C dry, paint it
- if wall A dry, paint it
- if wall B dry, paint it
- if wall C dry, paint it

In the first case, we need three workers. Or a single one that takes three times as long. Or two that take twice as long. But in the latter case, if we assume that painting is much faster than waiting, a single worker can do the same job in nearly the same amount of time as three workers in the first example.

#### Non-interruptable, blocking tasks

When we only have stupid tasks that make workers unecessarily block tools for extended amounts of time, we are kind of screwed. Luckily, we have an ace up our sleeves in the form of a "overseer" called the operating system. He is able to yank the tool out of the hands of workers that don't currently need them and pass it to others. Then, he yanks it back once the original worker is ready to paint. Unfortunately, that takes some time and makes things a little slower.

- "overseer" = "operating system"

In effect, than means that

- paint wall A &hyphen; wait &hyphen; paint wall A &hyphen; wait &hyphen; paint wall A

type instruction can be performed by three workers with a single tool similarly fast as by a single worker with interruptable instructions, but that still means you are paying for three workers that do the work of one.

## What is a thread?

On a computer, a thread is a line of execution of commands. A program can be made up of multiple threads that advance (more or less) independently of each other, like workers on a construction site.

If the amount of resources that the workers can use to perform their work is limited, somebody has to decide which worker gets to use those resources when.

When that somebody is the operating system, we call the thread an OS thread or a kernel level thread. The operating system is pretty efficient in scheduling threads to run, but we may be able to do even better from within our own program, since we have more knowlegde of what's going on and can avoid some of the work that is associated with switching between OS threads.

They way we are able to perform this scheduling ourselves is by splitting the entirety of instructions that make up our program into distinct green threads or user level threads that are able to run independently of each other. Instead of creating one kernel level thread per green thread, we only create a limited amount (possibly even just one), and then run those green threads consecutively on those. An added bonus is when our green threads are interruptible at strategic breakpoints (whenever something is happening that is not CPU bound), because then we can just do something else while waiting for the harddrive or the network.

### Kernel level threads

As described, kernel level threads are an operating system thing. There are many processes running on a typical computer, and each of them consists of one or more threads. We only have a limited number of tools / CPU cores, so the operating system is busy all the time deciding who gets those tools when. This is no simple task.

### Green threads

Using kernel level threads is the only way to achieve parallelism. But sometimes we just need our program to flexibly jump between different things that need to be done, without doing multiple things at a time. We _could_ do this with kernel level threads. But it might be wasteful of resources, because we may be able to schedule work more efficiently than the OS could.

A green thread is a line of commands that exists within our program. Like, when a HTTP request comes in: "Read the body and make a database query. Then read the DB response, make a computation based on it, and write the result back to the network socket". Querying the database will likely take a signifacant amount of time. That's why we'd better make this thread interruptible

- Read the body and make a database query.
- When the response is ready, read it, make a computation based on it, and write the result back to the network socket

Using green threads makes sense when we have many things running asynchonously and each of them only takes a very short time. A perfect example of this is the event loop some of you may know from Javascript runtimes.

Which green thread executes when is decided by logic that is part of our program, not the OS.

## What is a runtime?

The core problem is that concurrent work would be easy to implement using only kernel level threads if the two following problems wouldn't exist

- Running a lot of compute heavy things may take away resources that are needed more urgently for other things, like keeping UI responsive. If we dismissed this problem, there would be no reason that tabs in our browser can by default only use a single thread. That's a safeguard from miserable code you may download off the internet, that could slow down your entire PC if it's badly written or malicious.
- Kernel level threads are expensive in creation and management

An async runtime may alleviate these problems by

- splitting work into and managing green threads
- creating a limited amount of kernel level threads and distributing work between them (aka using a thread pool)

Different approaches are needed for different types of workloads:

- heavy computation
- blocking calls like I/O
- async work

Different runtimes may specialize on optimizing a subset of these workloads.

### Tokio

The tokio runtime is optimized to manage things like a webserver. It must always be responsive, but may occasionally do blocking operations like file system access. It is not expected to run expensive computations.

To allow this kind of workload to run efficiently, tokio manages two distinct thread pools. One is typically initialized with as many kernel level threads as the machine it's running on has logical CPU cores. It then runs a JS-event-loop like scheme that distributes async work across these threads.

There is one scenario when this becomes problematic. That is when some of the workloads do blocking operations, like wait for a file to be read. This does not only block the green thread that represents the workload, it also blocks the kernel level thread that the green thread is running on. Do this too often at a time, and your webserver is frozen!

Since doing I/O is still a typical job for a webserver, tokio manages a second threadpool. This threadpool starts with zero kernel level threads and for each task you assign to this threadpool, a new kernel level thread is created. Up to a limit of about 500. Whenever a task needs I/O, you spawn it onto this "blocking" thread pool.

While this second thread pool is a neat escape hatch for stuff that blocks, it is better style to find a way to use a non-blocking API instead and stay on the default thread pool. Again, kernel level threads are expensive. Tokio provides async APIs for many things that are traditionally blocking.

### Rayon

A usecase that tokio does not cover is heavy computation, like in simulation software. You really want all your cores to run all the time at maximum efficiency, and responsiveness is not an issue. For this purpose, rayon does the same thing as tokio with its standard threadpool: Creating as many worker threads as there are logical cores and then throwing work at them. The difference is that the workload in this case can not be async, and thread management is tuned toward efficiency down to a level that takes for example CPU caching into account.

Rayon also provides a very simple API in the form of it's parallel iterator traits, which make it incredibly easy to parallelize iterative work.

## Code

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

ThreadA will keep Mutex1 locked and will block until it has gotten Mutex2 as well. ThreadB will do the same with Mutex2. None of them will ever release the lock on their Mutex, and neither will get chance to lock the other Mutex. That's a dead-lock. Your program is now frozen.

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

You can explore this repositories `src` directory for more snippets on different variations of runtimes and state-sharing mechanisms. Of course, it'd also be a good idea to look at the docs for tokio and rayon. A lot information I listed here was inspired by [this great blog post](https://ryhl.io/blog/async-what-is-blocking/).
