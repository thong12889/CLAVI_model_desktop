using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace CLAVI_model_desktop
{
    class SemanticSegmentation : IDisposable
    {
        private InferenceSession sess = null;

        public SemanticSegmentation(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }
        public Mat semsegInference(Mat image, string labelPath, double opacity)
        {
            int inputW = 2048;
            int inputH = 1024;
            Size imgSize = new Size(inputW, inputH);

            //Label file
            var label = File.ReadLines(labelPath);
            var labelList = label.ToArray();

            var data = DataPreprocessing(image);
            Mat imageFloat = data.Resize(imgSize);
            var input = new DenseTensor<float>(MatToList(imageFloat), new[] { 1, 3, imgSize.Height, imgSize.Width });

            // Setup inputs and outputs
            var inputMeta = sess.InputMetadata;
            var inputName = inputMeta.Keys.ToArray()[0];
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, input)
            };

            using (var results = sess.Run(inputs))
            {
                //Postprocessing
                var resultsArray = results.ToArray();
                //Pred
                var pred_value = resultsArray[0].AsEnumerable<Int64>().ToArray();
                var pred_dim = resultsArray[0].AsTensor<Int64>().Dimensions.ToArray();

                var output = ConvertSegmentationResult(pred_value, pred_dim, labelList.Length);

                var maskRS = output.Resize(image.Size());
                image = image * (1 - opacity) + maskRS * opacity;
            }

            return image; 
        }
        public static Mat ConvertSegmentationResult(long[] pred, int[] pred_dim, int class_num)
        {
            Vec3b[] palette = GenPalette(class_num);
            palette[0] = new Vec3b(0, 0, 0);
            Mat mat = new Mat(new Size(pred_dim[3], pred_dim[2]), MatType.CV_8UC3);
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int h = 0; h < pred_dim[2]; h++)
                {
                    for (int w = 0; w < pred_dim[3]; w++)
                    {
                        int idx = (batch * pred_dim[1] * pred_dim[2] * pred_dim[3]) + (h * pred_dim[3]) + w;

                        Vec3b pix = mat.At<Vec3b>(h, w);
                        pix = palette[pred[idx]];
                        mat.Set<Vec3b>(h, w, pix);
                    }
                }
            }
            return mat;
        }
        private Mat DataPreprocessing(Mat image)
        {
            Mat data = Mat.Zeros(image.Size(), MatType.CV_32FC3);
            using (var rgbImage = new Mat())
            {
                Cv2.CvtColor(image, rgbImage, ColorConversionCodes.BGR2RGB);
                rgbImage.ConvertTo(data, MatType.CV_32FC3, (float)(1 / 255.0));
                var channelData = Cv2.Split(data);
                channelData[0] = (channelData[0] - 0.485) / 0.229;
                channelData[1] = (channelData[1] - 0.456) / 0.224;
                channelData[2] = (channelData[2] - 0.406) / 0.225;
                Cv2.Merge(channelData, data);
            }
            return data;
        }
        static Vec3b[] GenPalette(int classes)
        {
            Random rnd = new Random(classes);
            Vec3b[] palette = new Vec3b[classes];
            for (int i = 0; i < classes; i++)
            {
                byte v1 = (byte)rnd.Next(0, 255);
                byte v2 = (byte)rnd.Next(0, 255);
                byte v3 = (byte)rnd.Next(0, 255);
                palette[i] = new Vec3b(v1, v2, v3);
            }
            return palette;
        }
        private static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }
        private unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
        {
            float[] array = new float[chn * ih * iw];

            for (int y = 0; y < ih; y++)
            {
                for (int x = 0; x < iw; x++)
                {
                    for (int c = 0; c < chn; c++)
                    {
                        var idx = (y * chn) * iw + (x * chn) + c;
                        var idx2 = (c * iw) * ih + (y * iw) + x;
                        array[idx2] = ptr[idx];
                    }
                }
            }
            return array;
        }
        public void Dispose()
        {
            sess?.Dispose();
            sess = null;
        }
    }
}
