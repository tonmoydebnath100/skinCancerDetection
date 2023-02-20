package com.example.skincancerdetection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import com.example.skincancerdetection.ml.ModelUnquant;
import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button selectBtn, captureBtn, predictBtn;
    TextView predictionResult,predtionResult1;
    Bitmap bitmap;
    ImageView imageView;
    int imageSize = 224;
    GoogleSignInOptions gso;
    GoogleSignInClient gsc;
    Button signOutBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getPermission();
        gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestEmail().build();
        gsc = GoogleSignIn.getClient(this,gso);

        selectBtn= findViewById(R.id.selectBtn);
        captureBtn= findViewById(R.id.captureBtn);
        predictionResult= findViewById(R.id.Result);
        predtionResult1= findViewById(R.id.predtionResult1);
        imageView= findViewById(R.id.imageView);
        signOutBtn = findViewById(R.id.signout);

        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });
        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });
        signOutBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                signOut();
            }
        });

    }
    void signOut(){
        gsc.signOut().addOnCompleteListener(new OnCompleteListener<Void>() {
            @Override
            public void onComplete(Task<Void> task) {
                finish();
                startActivity(new Intent(MainActivity.this,Login.class));
            }
        });
    }

    void  getPermission(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if(checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA},11);
            }
        }
    }
    public void classify(Bitmap image){
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer= ByteBuffer.allocateDirect(4* imageSize* imageSize* 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValue = new int [imageSize*imageSize];
            bitmap.getPixels(intValue,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
            int pixel=0;
            for (int i=0; i<imageSize;i++){
                for (int j=0;j<imageSize;j++){
                    int val= intValue[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences=outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Benign - (vasc)", "Benign - (nv)", "Malignant - (mel)","Benign - (df)", "Benign - (bkl)", "Malignant - (bcc)","Malignant - (akiec)"};
            predtionResult1.setText(classes[maxPos]);
            String[] cls ={"vasc","nv","mel","df","bkl","bcc","akiec"};
            String s="";
            for (int i=0; i<cls.length; i++){
                s+= String.format("%s: %.1f%%\n",cls[i],confidences[i]*100);
            }
            predictionResult.setText(s);
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode==11){
            if(grantResults.length>0){
                if (grantResults[0]!=PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==10){
            if(data!=null){
                Uri uri= data.getData();
                try {
                    bitmap= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                    imageView.setImageBitmap(bitmap);
                    bitmap = Bitmap.createScaledBitmap(bitmap,imageSize,imageSize, false);
                    classify(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
         if (requestCode==12){
             bitmap=(Bitmap)data.getExtras().get("data");
             imageView.setImageBitmap(bitmap);
             bitmap = Bitmap.createScaledBitmap(bitmap,imageSize,imageSize, false);
             classify(bitmap);
         }
        super.onActivityResult(requestCode, resultCode, data);
    }
}