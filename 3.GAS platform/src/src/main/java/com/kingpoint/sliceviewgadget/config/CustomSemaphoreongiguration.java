package com.kingpoint.sliceviewgadget.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

@Configuration
public class CustomSemaphoreongiguration {

    @Value("${tile.syncCount}")
    private Integer syncCount;

    @Bean
    public Semaphore customSemaphore(){
        return new Semaphore(syncCount);
    }
}
