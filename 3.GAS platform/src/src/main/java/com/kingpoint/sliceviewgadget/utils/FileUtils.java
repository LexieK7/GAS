package com.kingpoint.sliceviewgadget.utils;

import java.io.File;
import java.nio.file.Path;

public class FileUtils {

    /**
     * 获取文件后缀
     *
     * @param file
     */
    public static String getFileExtension(File file) {
        String fileName = file.getName();
        int lastIndexOfDot = fileName.lastIndexOf('.');
        if (lastIndexOfDot == -1) {
            return "";
        }
        return fileName.substring(lastIndexOfDot + 1);
    }

    /**
     * 获取文件前缀
     *
     * @param file
     * @return
     */
    public static String getFilePrefix(File file) {
        String fileName = file.getName();
        int lastIndexOfDot = fileName.lastIndexOf('.');
        if (lastIndexOfDot == -1) {
            return fileName;
        }
        return fileName.substring(0, lastIndexOfDot);
    }


    public static void main(String[] args) {
        File file = new File("D:\\data\\P20014531-6C-.svs");
        System.out.println(getFileExtension(file));
        System.out.println(getFilePrefix(file));
    }
}
