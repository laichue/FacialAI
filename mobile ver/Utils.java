package com.example.handwritingai;

import android.content.Context;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Utils {
    public static MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        // 모델 파일을 로드하여 ByteBuffer로 변환
        FileInputStream inputStream = new FileInputStream(context.getAssets().openFd(modelFileName).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileChannel.position();
        long declaredLength = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}