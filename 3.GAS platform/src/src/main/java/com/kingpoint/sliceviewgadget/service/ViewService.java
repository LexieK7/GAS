package com.kingpoint.sliceviewgadget.service;

import com.kingpoint.sliceviewgadget.entity.VO.ImgStatus;

import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

public interface ViewService {

    /**
     * 获取可查看的文件夹列表
     * @return
     */
    List<String> getDirList();

    /**
     * 获取可切割svs文件
     * @return
     */
    List<ImgStatus> getImgList();

    /**
     * 调用python
     * @param fileName
     * @param url
     */
    void sendPython(String fileName, String url);

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
    void getImg(HttpServletResponse response, String directory, String z, String x, String y, Integer type) throws IOException;
}
