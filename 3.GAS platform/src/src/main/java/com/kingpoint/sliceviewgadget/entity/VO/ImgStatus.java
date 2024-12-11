package com.kingpoint.sliceviewgadget.entity.VO;

public class ImgStatus {
    String fileName;

    Integer status;

    public ImgStatus() {
    }

    public ImgStatus(String fileName, Integer status) {
        this.fileName = fileName;
        this.status = status;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public Integer getStatus() {
        return status;
    }

    public void setStatus(Integer status) {
        this.status = status;
    }
}
