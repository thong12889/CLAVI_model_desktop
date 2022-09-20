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
            int mode = 3;

            if(mode == 1)
            {
                //Object Detection
                var imagePath_obj = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/img_18.jpg";
                var modelPath_obj = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/technopro_obj.onnx";
                var labelPath_obj = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_obj/technopro_obj_labels.txt";
                var image_obj = Cv2.ImRead(imagePath_obj);
                var objectDetection = new ObjectDetection(modelPath_obj);
                var result_obj = objectDetection.ObjInference(image_obj, labelPath_obj, 0.6f);

                Cv2.NamedWindow("Result Obj", WindowFlags.FreeRatio);
                Cv2.ImShow("Result Obj", result_obj);
                Cv2.WaitKey();
            }
            if(mode == 2)
            {
                //Semantic Segmentation
                var imagePath_semseg = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_semseg/mixed_1.jpg";
                var modelPath_semseg = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_semseg/fruit_semseg.onnx";
                var labelPath_semseg = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_semseg/fruit_semseg_labels.txt";
                var image_semseg = Cv2.ImRead(imagePath_semseg);
                var semanticSegmentation = new SemanticSegmentation(modelPath_semseg);
                var result_semseg = semanticSegmentation.semsegInference(image_semseg, labelPath_semseg, 0.8);

                Cv2.NamedWindow("Result SemSeg", WindowFlags.FreeRatio);
                Cv2.ImShow("Result SemSeg", result_semseg);
                Cv2.WaitKey();
            }
            if(mode == 3)
            {
                //Instance Segmentation
                var imagePath_inseg = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_inseg/img_18.jpg";
                var modelPath_inseg = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_inseg/technopro_inseg.onnx";
                var labelPath_inseg = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_inseg/technopro_inseg_labels.txt";
                var image_inseg = Cv2.ImRead(imagePath_inseg);
                var instanceSegmentation = new InstanceSegmentation(modelPath_inseg);
                var result_inseg = instanceSegmentation.insegInference(image_inseg, labelPath_inseg, 0.6f, 0.8);

                Cv2.NamedWindow("Result Inseg", WindowFlags.FreeRatio);
                Cv2.ImShow("Result Inseg", result_inseg);
                Cv2.WaitKey();
            }
            if(mode == 4)
            {
                //Classification
                var imagePath_cls = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_cls/gorilla.jpg";
                var modelPath_cls = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_cls/animals_cls.onnx";
                var labelPath_cls = "C:/Users/thong/Desktop/Clavi_desktop_onnx/model_cls/animals_cls_labels.txt";
                var image_cls = Cv2.ImRead(imagePath_cls);
                var classification = new Classification(modelPath_cls);
                var (result_cls, result_score) = classification.clsInference(image_cls, 0.5f);

                //Label file
                var label = File.ReadLines(labelPath_cls);
                var labelList = label.ToArray();

                Console.WriteLine(labelList[result_cls]);
                Console.WriteLine(result_score.ToString("0.00"));
                Console.ReadLine();
            }
        }
    }
}
