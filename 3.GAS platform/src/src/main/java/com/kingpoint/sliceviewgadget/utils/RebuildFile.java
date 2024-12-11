package com.kingpoint.sliceviewgadget.utils;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class RebuildFile {

    private static final Map<String, Integer> zoomInterval = new HashMap<>();

    static {
        zoomInterval.put("1", 16384);
        zoomInterval.put("2", 8192);
        zoomInterval.put("5", 4096);
        zoomInterval.put("10", 2048);
        zoomInterval.put("20", 1024);
    }

    public static void main(String[] args) throws IOException {
        File file = new File("D:\\data\\0001");

        reNameZoom(file);

//        Files.walk(file.toPath(),1).forEach(RebuildFile::reName);
    }

    public static void reNameZoom(File file) {
        //获取子文件夹
        File[] files = file.listFiles(File::isDirectory);
        //排序
        List<File> sort = sort(files);
        //判空
        if (sort == null) {
            return;
        }
        //遍历命名
        for (int i = 0; i < sort.size(); i++) {
            File subFolder = sort.get(i);
            String oldName = subFolder.getName();
            //开始重命名
            File newFile = new File(subFolder.getParent(), String.valueOf(i));
            if (subFolder.renameTo(newFile)) {
                System.out.println(oldName + "重命名为" + newFile.getName());
                reNameX(newFile, oldName);
            } else {
                throw new RuntimeException(subFolder.getAbsolutePath() + "文件重命名失败");
            }
        }
    }

    public static void reNameX(File file, String zoom) {
        File[] files = file.listFiles(File::isDirectory);
        List<File> sort = sort(files);
        if (sort == null) {
            return;
        }
        int minX = Integer.parseInt(FileUtils.getFilePrefix(sort.get(0)));
        for (File subFolder : sort) {
            String oldName = subFolder.getName();
            int x = Integer.parseInt(FileUtils.getFilePrefix(subFolder));
            int index = (x - minX) / zoomInterval.get(zoom);
            File newFile = new File(subFolder.getParent(), String.valueOf(index));
            if (subFolder.renameTo(newFile)) {
                System.out.println(oldName + "重命名为" + index);
                reNameY(newFile, zoom);
            } else {
                throw new RuntimeException(subFolder.getAbsolutePath() + "文件重命名失败");
            }
        }
    }

    public static void reNameY(File file, String zoom) {
        File[] files = file.listFiles(File::isFile);
        List<File> sort = sort(files);
        if (sort == null) {
            return;
        }
        for (File subFolder : sort) {
            String oldName = subFolder.getName();
            int y = Integer.parseInt(FileUtils.getFilePrefix(subFolder));
            int index = y / zoomInterval.get(zoom);
            String newName = index + "." + FileUtils.getFileExtension(subFolder);
            if (subFolder.renameTo(new File(subFolder.getParent(), newName))) {
                System.out.println(oldName + "重命名为" + newName);
            } else {
                throw new RuntimeException(subFolder.getAbsolutePath() + "文件重命名失败");
            }
        }
    }


    public static void reName(Path path) {
        File file = path.toFile();
        File[] files = file.listFiles();
        if (files == null) {
            return;
        }
        //files按名字排序
        List<File> sortedFiles = sort(files);

        for (int i = 0; i < sortedFiles.size(); i++) {
            File file1 = sortedFiles.get(i);
            File parentFile = file1.getParentFile();
            if (file1.isFile()) {
                String fileExtension = FileUtils.getFileExtension(file1);
                file1.renameTo(new File(parentFile, i + "." + fileExtension));
            } else {
                file1.renameTo(new File(parentFile, "" + i));
            }
        }

    }


    public static List<File> sort(File[] files) {
        if (files == null || files.length == 0 || files[0] == null) {
            return null;
        }
        return Arrays.stream(files)
                .sorted((file1, file2) -> {
                    long l1 = Long.parseLong(FileUtils.getFilePrefix(file1));
                    long l2 = Long.parseLong(FileUtils.getFilePrefix(file2));
                    return Math.toIntExact(l1 - l2);
                })
                .collect(Collectors.toList());
    }
}
