package com.example.handwritingai;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private DrawView drawView;
    private Button recognitionButton;
    private TextView resultTextView;
    private Interpreter interpreter;
    private final int INPUT_SIZE = 28;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // DrawView 초기화
        drawView = findViewById(R.id.drawing_view);

        // 인식하기 버튼 초기화
        recognitionButton = findViewById(R.id.recognition_button);

        // 인식 결과를 보여줄 TextView 초기화
        resultTextView = findViewById(R.id.result_text_view);

        try {
            // 모델 파일 로드
            interpreter = new Interpreter(Utils.loadModelFile(MainActivity.this, "mnist.tflite"));
        } catch (IOException e) {
            Log.e("HandwritingAI", "Error loading model file: " + e);
        }


        // 인식하기 버튼 클릭 이벤트 처리
        recognitionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // DrawView에서 그린 이미지를 가져온다.
                Bitmap bitmap = Bitmap.createBitmap(drawView.getWidth(), drawView.getHeight(),
                        Bitmap.Config.ARGB_8888);
                Canvas canvas = new Canvas(bitmap);
                drawView.draw(canvas);

                // 이미지 크기를 INPUT_SIZE x INPUT_SIZE로 변경한다.
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                // 텐서 흐름 객체 생성
                TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

                // 이미지를 텐서 흐름 객체에 넣기 위해 ByteBuffer 생성
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE);
                byteBuffer.order(ByteOrder.nativeOrder());

                // 이미지의 각 픽셀 값을 ByteBuffer에 저장
                int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
                scaledBitmap.getPixels(pixels, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(),
                        scaledBitmap.getHeight());
                for (int pixelValue : pixels) {
                    byteBuffer.putFloat((pixelValue >> 16) & 0xFF);
                    byteBuffer.putFloat((pixelValue >> 8) & 0xFF);
                    byteBuffer.putFloat(pixelValue & 0xFF);
                }

                // ByteBuffer를 TensorBuffer로 변환
                TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(new int[]{1, INPUT_SIZE, INPUT_SIZE, 3},
                        DataType.FLOAT32);
                tensorBuffer.loadBuffer(byteBuffer);

                // 텐서 흐름 객체에 이미지를 넣음
                tensorImage.load(tensorBuffer);

                // 모델에 입력 데이터를 넣고 결과 값을 가져옴
                float[][] result = new float[1][10];
                interpreter.run(tensorImage.getBuffer(), result);

                // 결과 값을 분석하여 화면에 출력
                int recognizedNumber = getRecognizedNumber(result);
                resultTextView.setText(String.valueOf(recognizedNumber));
            }
        });
    }

    // 모델 결과 값을 분석하는 메서드
    private int getRecognizedNumber(float[][] result) {
        float maxProbability = result[0][0];
        int maxProbabilityIndex = 0;

        for (int i = 1; i < result[0].length; i++) {
            if (result[0][i] > maxProbability) {
                maxProbability = result[0][i];
                maxProbabilityIndex = i;
            }
        }

        return maxProbabilityIndex;
    }
}
