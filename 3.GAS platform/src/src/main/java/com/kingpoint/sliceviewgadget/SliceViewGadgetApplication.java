package com.kingpoint.sliceviewgadget;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import java.util.concurrent.ExecutorService;

@SpringBootApplication
public class SliceViewGadgetApplication {

    public static void main(String[] args) {
        SpringApplication.run(SliceViewGadgetApplication.class, args);
    }

}
