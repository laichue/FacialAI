package com.example.handwritingai;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawView extends View {

    // Paint 클래스는 캔버스에 그리기를 할 때 사용하는 페인트 속성을 담는 클래스이다.
    private Paint paint;

    // Path 클래스는 그리기 경로를 나타내는 클래스이다.
    private Path path;

    public DrawView(Context context, AttributeSet attrs) {
        super(context, attrs);

        // Paint 객체를 생성하고 속성을 설정한다.
        paint = new Paint();
        paint.setColor(Color.BLACK);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(10);

        // Path 객체를 생성한다.
        path = new Path();
    }

    // onDraw() 메서드는 뷰가 그려질 때 호출되는 메서드이다.
// 캔버스 객체를 인자로 받아서 캔버스에 그리는 작업을 한다.
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // 생성한 Path 객체와 Paint 객체를 사용하여 캔버스에 그린다.
        canvas.drawPath(path, paint);
    }

    // onTouchEvent() 메서드는 뷰에 터치 이벤트가 발생했을 때 호출되는 메서드이다.
// MotionEvent 객체를 인자로 받아서 터치 이벤트를 처리한다.
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            // 손가락이 눌렸을 때
            case MotionEvent.ACTION_DOWN:
                // Path 객체의 시작점을 설정한다.
                path.moveTo(x, y);
                return true;
            // 손가락이 움직일 때
            case MotionEvent.ACTION_MOVE:
                // Path 객체에 선을 추가한다.
                path.lineTo(x, y);
                break;
            // 손가락이 땠을 때
            case MotionEvent.ACTION_UP:
                break;
            default:
                return false;
        }

        // 화면을 다시 그리도록 한다.
        invalidate();
        return true;
    }
}