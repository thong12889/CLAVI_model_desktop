using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace CLAVI_model_desktop
{
    class Classification : IDisposable
    {
        private InferenceSession sess = null;

        public Classification(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }
        public (int, float) clsInference(Mat image, float threshold)
        {
            int inputW = 224;
            int inputH = 224;
            Size imgSize = new Size(inputW, inputH);
            int result_class;
            float result_score;

            Mat imageFloat = image.Resize(imgSize);
            imageFloat.ConvertTo(imageFloat, MatType.CV_32FC1);
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
                var resultsArray = results.ToArray();
                var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                float maxValue = pred_value.Max();
                int maxIndex = pred_value.ToList().IndexOf(maxValue);
                //var secondMax = pred_value.OrderByDescending(r => r).Skip(1).FirstOrDefault();

                result_class = maxIndex;
                result_score = pred_value[maxIndex];
            }

            return (result_class, result_score);
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
