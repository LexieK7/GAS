package com.kingpoint.sliceviewgadget.config;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebMvcConfiguration implements WebMvcConfigurer {

    @Value("${tile.path}")
    private String tilePath;


    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 静态资源访问路径
        registry.addResourceHandler("/**")
                // classpath下的静态资源目录
                .addResourceLocations("classpath:/static/")
                // 本地磁盘下的静态资源目录
                .addResourceLocations("file:" + tilePath);
    }


}
