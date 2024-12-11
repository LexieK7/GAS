package com.kingpoint.sliceviewgadget.utils;

import java.util.concurrent.Semaphore;

public class SemaphoreExample {
    private final Semaphore semaphore = new Semaphore(5); // 允许5个线程同时访问共享资源

    public void accessSharedResource() {
        try {
            semaphore.acquire(); // 获取许可
            Thread.sleep(5000);
            System.out.println(Thread.currentThread().getName() + " 正在访问共享资源");
            // 对共享资源进行操作
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            semaphore.release(); // 释放许可
        }
    }

    public static void main(String[] args) {
        SemaphoreExample example = new SemaphoreExample();

        for (int i = 0; i < 10; i++) {
            new Thread(() -> example.accessSharedResource(), "线程" + (i + 1)).start();
        }
    }
}
