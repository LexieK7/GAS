package com.kingpoint.sliceviewgadget.controller;

import com.alibaba.fastjson2.JSONObject;
import com.kingpoint.sliceviewgadget.entity.R;
import com.kingpoint.sliceviewgadget.entity.VO.ImgStatus;
import com.kingpoint.sliceviewgadget.service.ViewService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;


@RestController
@RequestMapping("/view")
public class ViewController {

    @Autowired
    private ViewService viewService;

    /**
     * 获取可查看的文件夹列表
     *
     * @return
     */
    @GetMapping("getDirList")
    public R<List<String>> getDirList() {
        List<String> dirList = viewService.getDirList();
        return new R<>(dirList);
    }

    /**
     * 获取可切割svs文件
     *
     * @return
     */
    @GetMapping("getImgList")
    public R<List<ImgStatus>> getImgList() {
        List<ImgStatus> imgList = viewService.getImgList();
        return new R<>(imgList);
    }


    /**
     * 切隔图片
     *
     * @param file 图片名称
     * @return
     */
    @PostMapping("/tileImg")
    public R<String> tileImg(@RequestBody JSONObject file) {
        String url = "getPic/";
        viewService.sendPython(file.getString("fileName"), url);
        //返回目录
        return new R<>("切割完成");
    }


    /**
     * 获取图片
     *
     * @param
     * @return
     */
    @GetMapping("getImg/{directory}/{z}/{x}/{y}/{type}")
    public void getImg(HttpServletResponse response, @PathVariable("directory") String directory, @PathVariable("z") String z, @PathVariable("x") String x, @PathVariable("y") String y, @PathVariable("type") Integer type) throws IOException {
        viewService.getImg(response, directory, z, x, y, type);
    }


}
