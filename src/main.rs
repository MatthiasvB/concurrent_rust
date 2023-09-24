// Examples for different parallel computing approaches, informed by https://ryhl.io/blog/async-what-is-blocking/

#![allow(dead_code)]
//#![allow(unused_imports)]
use futures;
use rayon::prelude::*;
use std;
use tokio;

const NUMBER_OF_THREADS: usize = 60;
const FIBUNACCI_LOAD: usize = 43;

fn main() {
    // explicit_threads_no_communication(NUMBER_OF_THREADS);
    // explicit_threads_result(NUMBER_OF_THREADS);
    // explicit_threads_mutex(NUMBER_OF_THREADS);
    // explicit_threads_channels(NUMBER_OF_THREADS);
    // with_tokio_no_communication(NUMBER_OF_THREADS);
    // with_tokio_mutex(NUMBER_OF_THREADS);
    // with_tokio_channels(NUMBER_OF_THREADS);
    // with_tokio_async_channels(NUMBER_OF_THREADS);
    // with_tokio_blocking(NUMBER_OF_THREADS);
    // with_compute_rayon(NUMBER_OF_THREADS);
    // parallel_iterators(NUMBER_OF_THREADS);

    futures::executor::block_on(parallel_dispatcher());    
}

async fn parallel_dispatcher() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();

    let standalone_jobs: Vec<fn(usize)> = vec![
        explicit_threads_no_communication,
        explicit_threads_mutex,
        explicit_threads_channels,
        with_compute_rayon,
        parallel_iterators,
    ];

    let standalone_threads: Vec<std::thread::JoinHandle<()>> = standalone_jobs
        .into_iter()
        .map(|func| std::thread::spawn(move || func(NUMBER_OF_THREADS)))
        .collect::<Vec<_>>();

    let _ = tokio::join!(
        // async
        tokio_mutex(NUMBER_OF_THREADS),
        tokio_no_communication(NUMBER_OF_THREADS),
        tokio_async_channels(NUMBER_OF_THREADS),
        // `tokio_channels` will block the thread while waiting for spawned, non-blocking threads to finish
        tokio::task::spawn_blocking(|| tokio_channels(NUMBER_OF_THREADS)),
        // blocking
        tokio_blocking(NUMBER_OF_THREADS)
    );

    standalone_threads.into_iter().for_each(|thread| {
        thread.join().unwrap();
    });
}

/// This is the most banal of all examples (though not the easiest to implement)
/// It creates a tokio runtime that executes a number of tasks. We are not interested in their result, only
/// their side-effects.
///
/// The runtime has two aspects:
/// - it schedules async jobs, roughly equivalent to the Javascript event-loop.
/// So this could run code _concurrently_ but not _in parallel_.
/// - it (may) spread this async code across multiple threads, typically using
/// a thread pool with as many threads as your machine has (virtual) cores.
/// _This_ enables the code in parallel
///
/// The details can be configured when creating the runtime
fn with_tokio_no_communication(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Execute a Future
    rt.block_on(tokio_no_communication(number_of_threads));
}

/// We are using `tokio::task::spawn` in here. This is meant for tasks that interrupt (`.await`)
/// very frequently. Another way to put it: This is for code as you'd write it in Javascript. Short
/// bursts of computation and many (potentially long) wait times.
///
/// This means that key here is that tokio will enable you to switch between currently running tasks
/// often and light-weight. Yes, it may also use multiple threads and enable parallelization, but
/// you should still avoid things that
/// - may block the thread for signifcant amounts of time due to heavy computations
/// - may block the thread for significant amounts of time because they are waiting for resources
///
/// So, the way it is used here is _wrong_. But this is more about how to call it, than when to use it.
///
/// What is it for then? Well, for web servers, for example, where you may have like 1000 clients connecting
/// simultaneously, but you have to interrupt frequently to query a DB or whatever else. So you spread
/// 1000 Futures across say 8 threads, but they are constantly switched out before running to completion.
async fn tokio_no_communication(number_of_threads: usize) {
    let tasks = (0..number_of_threads)
        .map(|num| tokio::task::spawn(delay_thread_async_then_square(num, num)));
    futures::future::join_all(tasks).await;
}

fn with_tokio_thread_result(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(tokio_thread_result(number_of_threads));
}

/// It gets a bit more interesting, because now we are interested in the results of the individual threads. Getting this
/// result is relatively easy, apart from some Rust-typical complicated error handling. We just use `collect`'s neat
/// ability to turn a list of `Result`s into a `Result` of a list (🤯), and then just `unwrap` it for brevity.
async fn tokio_thread_result(number_of_threads: usize) {
    async fn cb(num: usize) -> usize {
        let res = delay_thread_async_then_square(num, num).await;
        res
    }

    let tasks = (0..number_of_threads).map(|num| {
        // Can't use a closure because it must return a Future and async closures aren't stable, yet
        tokio::task::spawn(cb(num))
    });
    let result = futures::future::join_all(tasks)
        .await
        .into_iter()
        .collect::<Result<Vec<usize>, _>>()
        .unwrap();

    println!("{:?}", result);
}

/// Using the return value of (multiple) Futures is nice and easy, but obviously limited. A Future
/// can produce only one result. What if need to update state multiple times during the runtime of the thread?
/// Also, not every runtime will even allow you to get a result in this way.
///
/// The following approach is using a `Mutex` (short for mutual exclusion), which is a means to make sure only a single thread can access a
/// value at a given time, making the value thread-safe. To be able to _pass_ the thing to multiple threads,
/// we need an `Arc`.
///
/// Any time a thread wants to read from or write to the shared state, it locks the `Mutex` to get to the contained value.
///
/// When the resulting `MutexGuard` is dropped (goes out of scope), the `Mutex` becomes available for other threads again.
///
/// Even though we mutate state only once here, we _could_ just as well do it multiple times.
fn with_tokio_mutex(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(tokio_mutex(number_of_threads));
}

async fn tokio_mutex(number_of_threads: usize) {
    async fn cb(index: usize, num: usize, arr: std::sync::Arc<std::sync::Mutex<Vec<usize>>>) {
        let res = delay_thread_async_then_square(num, num).await;
        let mut numbers = arr.lock().unwrap();
        numbers[index] = res;
    }

    let vec: Vec<usize> = (0..number_of_threads).into_iter().collect();
    let numbers = std::sync::Arc::new(std::sync::Mutex::new(vec));

    let tasks = (0..number_of_threads).map(|num| {
        let numbers = numbers.clone();
        tokio::task::spawn(cb(num, num, numbers))
    });
    futures::future::join_all(tasks).await;

    let result = numbers.lock().unwrap();
    println!("{:?}", result);
}

/// Mutex's have several problems:
/// - Dead lock: If your threads require two `Mutex`s at a time, there is the possibility to dead lock your program so it
/// will never be able to progress. Say thread A has locked `Mutex` a, and thread B has locked `Mutex` b, but A also needs b
/// and B also needs a to continue. Well, this will never continue, because A won't release a and B won't release b. There you go,
/// program stuck indefinetly.
/// - Overhead: Locking the `Mutex` is an additional step and a thread will be blocked while waiting for the `Mutex` to become available. In
/// cases where multiple threads need access to the `Mutex` frequently, that means a lot of overhead.
///
/// Enter channels: Instead of passing state to multiple threads, you just give them a means to signal that something needs to be done. With
/// channels, threads can communicate with each other. So you can keep your state in the main thread and let the worker threads send you instructions
/// on how to update the state. This scheme can be upgraded to much more complicated scenarios than showcased here, and is such a powerful pattern
/// that it is the default mode of inter-thread communication in the concurrency-focused language Go.
fn with_tokio_channels(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    tokio_channels(number_of_threads);
}

fn tokio_channels(number_of_threads: usize) {
    async fn cb(index: usize, num: usize, sender: std::sync::mpsc::Sender<(usize, usize)>) {
        let res = delay_thread_async_then_square(num, num).await;
        sender.send((index, res)).unwrap();
    }

    let (sender, receiver) = std::sync::mpsc::channel::<(usize, usize)>();
    let mut vec: Vec<usize> = (0..number_of_threads).into_iter().collect();

    (0..number_of_threads).for_each(|num| {
        let sender = sender.clone();
        tokio::task::spawn(cb(num, num, sender));
    });
    // need to drop the sender, because the iterator below will only complete once all senders are dropped
    drop(sender);

    receiver.iter().for_each(|(index, res)| {
        vec[index] = res;
    });

    println!("{:?}", vec);
}

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
    // need to drop the sender, because the iteratoionbelow will only complete once all senders are dropped
    drop(sender);

    while let Some((index, res)) = receiver.recv().await {
        vec[index] = res;
    }

    println!("{:?}", vec);
}

/// Rayon is a concurrency runtime like tokio, but focused on compute-heavy
/// parallel executions of code. As such, this is the runtime that is appropriate for
/// the kind of work we do in this example (which is compute heavy).
///
/// Rayon has two main ways it can be used, and this is the more explicit one of them, where
/// we spawn threads like we do with tokio. Notice though that we don't get a JoinHandle to wait for the spawned
/// thread, and so also have no means to the "result" of the thread.
///
/// Instead, we use a oneshot-channel from tokio, which is a channel that will allow us to send
/// exactly one message. We could have used standard channels or a `Mutex` just as well, but it's interesting
/// to look at this approach, too.
///
/// We also build a list of all the oneshot-receivers. Turns out, they work just like a `JoinHandle` (besides other things)!
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
        recv
    });
    let result = futures::future::join_all(receivers)
        .await
        .into_iter()
        .collect::<Result<Vec<usize>, _>>()
        .unwrap();
    println!("{:?}", result);
}

/// Tokio has a separate, much larger thread-pool for threads that will be blocking due to I/O, meaning
/// waiting for network requests, file access or similar things. Stuff that will block the thread but _not_
/// with heavy compute.
///
/// This is different from the constantly-stopping Javascript event-loop kind of thread, because the thread will
/// not be blocked by a call to `.await`, but instead actually blocked in the way the operating system understands it,
/// which means taken off the processor to make room for other tasks. If we did this on the tokio standard thread pool, blocking the
/// thread effectively reduces the number of (usable) worker threads by one. Do this too often at the same time, and your
/// webserver is frozen!
///
/// It is also different from compute-heavy threads in that it is safe to run many many such kernel level threads at the same time.
/// This is not ideal for compute heavy stuff, because if you spawn 100 threads that do heavy computing, they will actually
/// compete for CPU time, which is not desirable.
///
/// That being said, trying to use async versions of usually blocking operations, for example using file system APIs provided by tokio,
/// should be preferred over synchronous APIs in blocking threads.
///
/// If we still need blocking threads, we spawn them with the `spawn_blocking` function.
fn with_tokio_blocking(number_of_threads: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(tokio_blocking(number_of_threads));
}

async fn tokio_blocking(number_of_threads: usize) {
    let futures = (0..number_of_threads).map(|num| {
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

/// Until now, we have used runtimes (tokio and rayon) to do "concurrent" work. That is work
/// that will be executed in arbitrary order.
///
/// These runtimes _may_ use "parallelism" (in the form of "kernel level threads") to do more
/// of this work in less time (and in the rayon case, this is the main point). But you can't technically be entirely sure about that.
///
/// You don't need a runtime to spawn threads, though! You can spawn kernel level threads yourself.
/// If you do this, you can be absolutely sure that a separate (kernel level) thread is created for each time you call
/// `thread::spawn`!
///
/// Of course, it is still your operating system that decides which thread gets CPU time when, but this is
/// probably the greatest amount of control you are going to get over the parallelism of you program.
///
/// Just keep in mind that spawning too many threads may not be a good idea, and runtimes like rayon
/// exist specifically to take that burden of your shoulders.
fn explicit_threads_no_communication(number_of_threads: usize) {
    let join_handles: Vec<std::thread::JoinHandle<()>> = (0..number_of_threads)
        .map(|num: usize| {
            std::thread::spawn(move || {
                work_then_square(num, num);
            })
        })
        .collect();

    join_handles.into_iter().for_each(|thread| {
        thread.join().unwrap();
    });
}

fn explicit_threads_result(number_of_threads: usize) {
    let join_handles: Vec<std::thread::JoinHandle<usize>> = (0..number_of_threads)
        .map(|num: usize| std::thread::spawn(move || work_then_square(num, num)))
        .collect();

    let result: Vec<usize> = join_handles
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();

    println!("{:?}", result);
}

fn explicit_threads_threat_result(number_of_threads: usize) {
    let mut threads: Vec<std::thread::JoinHandle<usize>> = vec![];
    (0..number_of_threads).for_each(|num: usize| {
        threads.push(std::thread::spawn(move || {
            let res = work_then_square(num, num);
            res
        }));
    });

    let result: Vec<usize> = threads
        .into_iter()
        .map(|thread| thread.join())
        .collect::<Result<Vec<usize>, _>>()
        .unwrap();
    println!("{:?}", result);
}

fn explicit_threads_channels(number_of_threads: usize) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut threads: Vec<std::thread::JoinHandle<()>> = vec![];
    (0..number_of_threads).for_each(|num| {
        let sender = sender.clone();
        threads.push(std::thread::spawn(move || {
            let res = work_then_square(num, num);
            sender.send((num, res)).unwrap();
        }));
    });
    drop(sender); // required: iterator below only finishes when all senders are dropped

    let mut result = vec![0; number_of_threads];
    receiver.iter().for_each(|(index, square)| {
        result[index] = square;
    });
    println!("{:?}", result);
}

fn explicit_threads_mutex(number_of_threads: usize) {
    let mut threads: Vec<std::thread::JoinHandle<()>> = vec![];
    let vec: Vec<usize> = (0..number_of_threads).into_iter().collect();
    let numbers = std::sync::Arc::new(std::sync::Mutex::new(vec.clone()));
    (0..number_of_threads).for_each(|num| {
        let numbers = numbers.clone();
        threads.push(std::thread::spawn(move || {
            let vec = numbers.lock().unwrap();
            let operand = vec[num];
            drop(vec); // to release the lock
            let res = work_then_square(num, operand);
            let mut vec = numbers.lock().unwrap();
            vec[num] = res;
        }))
    });

    threads.into_iter().for_each(|thread| {
        thread.join().unwrap();
    });
    let result = numbers.lock().unwrap();
    println!("{:?}", *result);
}

/// And here a goodie to finish on! Rayon provides traits that allow you to turn (most?!) iterators
/// into parallel iterators, where each iteration spawns a concurrent green thread (user level thread)
/// on rayon's thread pool! You notice hardly any difference between sync and async code here!
///
/// If your parallel workload is just a bunch of near-identical calculations, this is an increadibly easy
/// solution to optimize them.
fn parallel_iterators(number_of_threads: usize) {
    let result: Vec<usize> = (0..number_of_threads)
        .into_par_iter()
        .map(|num: usize| {
            let res = work_then_square(num, num);
            res
        })
        .collect();

    println!("{:?}", result);
}

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

fn work_then_square(thread_no: usize, to_be_squared: usize) -> usize {
    println! {"Compute thread {thread_no} starts to do some work"}
    work();
    let res = to_be_squared.pow(2);
    println! {"Compute thread {thread_no} done. Result is {res}"};
    res
}

fn block_thread_for(seconds: u64) {
    std::thread::sleep(std::time::Duration::from_secs(seconds));
}

async fn sleep_for(seconds: u64) {
    tokio::time::sleep(std::time::Duration::from_secs(seconds)).await;
}

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
