package com.kingpoint.sliceviewgadget.service.impl;

import com.alibaba.fastjson2.JSONObject;
import com.kingpoint.sliceviewgadget.entity.VO.ImgStatus;
import com.kingpoint.sliceviewgadget.service.ViewService;
import com.kingpoint.sliceviewgadget.utils.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;


@Service
public class ViewServiceImpl implements ViewService {

    @Value("${tile.path}")
    private String tilePath;

    @Value("${python.api.url}")
    private String pythonUrl;

    @Autowired
    @Qualifier("customSemaphore")
    private Semaphore semaphore;

    @Autowired
    private RestTemplate restTemplate;

    /**
     * 获取可查看的文件夹列表
     *
     * @return
     */
    @Override
    public List<String> getDirList() {
        File file = new File(tilePath);
        File[] files = file.listFiles(File::isDirectory);
        if (files != null) {
            return Arrays.stream(files).map(File::getName).collect(Collectors.toList());
        }
        return null;
    }

    /**
     * 获取可切割svs文件
     *
     * @return
     */
    @Override
    public List<ImgStatus> getImgList() {
        ArrayList<ImgStatus> imgStatuses = new ArrayList<>();
        File[] files = new File(tilePath).listFiles(o -> o.isFile() && "svs".equals(FileUtils.getFileExtension(o)));
        if (files != null) {
            //所有目录
            List<String> dirList = getDirList();
            for (File file : files) {
                String filePrefix = FileUtils.getFilePrefix(file);
                imgStatuses.add(new ImgStatus(file.getName(), dirList.contains(filePrefix) ? 1 : 0));
            }
        }
        return imgStatuses;
    }

    /**
     *
     * @param fileName
     * @param url
     */
    @Override
    public void sendPython(String fileName, String url) {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("image_path", tilePath + fileName);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> request = new HttpEntity<>(jsonObject.toJSONString(), headers);
        ResponseEntity<String> responseEntity = restTemplate.exchange(pythonUrl + url, HttpMethod.POST, request, String.class);
        if (!Objects.requireNonNull(responseEntity.getBody()).contains("Image processed and saved")) {
            System.out.println(responseEntity.getBody());
            throw new RuntimeException("服务器中无python环境,无法调用");
        }
    }

    /**
     * 重定向到对应的图片
     * @param response
     * @param directory
     * @param z
     * @param x
     * @param y
     * @param type
     * @throws IOException
     */
    @Override
    public void getImg(HttpServletResponse response, String directory, String z, String x, String y, Integer type) throws IOException {
        if (Integer.parseInt(x) < 0 || Integer.parseInt(y) < 0) {
            return;
        }
        //获取低清图片
        if (type == 0) {
            response.sendRedirect("/" + directory + "/" + z + "/" + x + "/" + y + ".png");
            return;
        }
        //获取高清图片
        //校验有无高清图片
        //存在高清图片
        File file = new File(tilePath, directory + "/" + z + "/" + x + "/" + "hd/" + y + ".png");
        if (file.exists()) {
            response.sendRedirect("/" + directory + "/" + z + "/" + x + "/" + "hd/" + y + ".png");
            return;
        }
        //调用模型
        String url = "imageHD/";
        String fileName = directory + "/" + z + "/" + x + "/" + y + ".png";

        try {
            semaphore.acquire();
            // 处理请求的逻辑
            System.out.println("不存在高清图片" + fileName + ",开始调用python");
            sendPython(fileName, url);
            System.out.println(fileName + "调用python完成");
            response.sendRedirect("/" + directory + "/" + z + "/" + x + "/" + "hd/" + y + ".png");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            e.printStackTrace();
        } finally {
            semaphore.release();
        }
    }
}
