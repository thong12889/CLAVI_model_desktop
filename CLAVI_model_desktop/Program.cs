using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace CLAVI_model_desktop
{
    class Program
    {
        static void Main(string[] args)
        {
            int mode = 1;

            if(mode == 1)
            {
                var imagePath_obj = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/img_18.jpg";
                var modelPath_obj = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/technopro_obj.onnx";
                var labelPath_obj = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/technopro_obj_labels.txt";
                var image_obj = Cv2.ImRead(imagePath_obj);
                var objectDetection = new ObjectDetection(modelPath_obj);
                var result = objectDetection.ObjInference(image_obj, labelPath_obj, 0.6f);

                /*Cv2.NamedWindow("Result Obj", WindowFlags.FreeRatio);
                Cv2.ImShow("Result Obj", result);
                Cv2.WaitKey();*/
                Cv2.ImWrite("C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/result.jpg", result);
            }
        }
    }
}
