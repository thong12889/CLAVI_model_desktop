using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace CLAVI_model_desktop
{
    class Anomaly : IDisposable
    {
        private InferenceSession sess = null;

        public Anomaly(string modelPath)
        {
            var option = new SessionOptions();
            //var option = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            //SessionOptions.MakeSessionOptionWithCudaProvider(0);
            //option.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            sess = new InferenceSession(modelPath, option);

            //sess = new InferenceSession(modelPath);
        }

        public Mat AnomalyInference(Mat image, double opacity = 0.6)
        {
            int inputW = 224;
            int inputH = 224;
            Size imgSize = new Size(inputW, inputH);

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

            Mat result_anomaly = image.Clone();
            result_anomaly = result_anomaly.Resize(imgSize);
            using (var results = sess.Run(inputs))
            {
                var resultsArray = results.ToArray();
                var map_value = resultsArray[0].AsEnumerable<float>().ToArray();
                var map_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                var score_value = resultsArray[1].AsEnumerable<float>().ToArray();
                var score_dim = resultsArray[1].AsTensor<float>().Dimensions.ToArray();

                var min = map_value.Min();
                var max = map_value.Max();

                var normalizeMap = map_value.Select(x => ((x - min) / (max - min)) * 255).ToArray();
                var heatMap = GetHeatmap(map_value, map_dim, normalizeMap);
                Cv2.ApplyColorMap(heatMap, heatMap, ColormapTypes.Jet);
                Cv2.AddWeighted(result_anomaly, opacity, heatMap, 1 - opacity, 0, heatMap);

                Cv2.ImShow("Predicted Heat Map", heatMap);

                var result_mask = GetResultMask(map_value, map_dim, score_value[0]);
                Cv2.ImShow("Predicted Mask", result_mask);

                var gray = new Mat();
                Cv2.CvtColor(result_mask, gray, ColorConversionCodes.BGR2GRAY);
                OpenCvSharp.Point[][] contours;
                OpenCvSharp.HierarchyIndex[] hindex;
                Cv2.FindContours(gray, out contours, out hindex, RetrievalModes.CComp, ContourApproximationModes.ApproxNone);
                Cv2.DrawContours(result_anomaly, contours, -1, new Scalar(0, 0, 255), 2);
            }

            return result_anomaly;
        }
        public static Mat GetHeatmap(float[] output_value, int[] output_dim, float[] HM_value)
        {
            Mat mat = new Mat(new Size(output_dim[2], output_dim[3]), MatType.CV_8UC3);
            for (int batch = 0; batch < output_dim[0]; batch++)
            {
                for (int cls = 0; cls < output_dim[1]; cls++)
                {
                    for (int h = 0; h < output_dim[2]; h++)
                    {
                        for (int w = 0; w < output_dim[3]; w++)
                        {
                            int idx = (batch * output_dim[1] * output_dim[2] * output_dim[3]) + (cls * output_dim[2] * output_dim[3]) + (h * output_dim[3]) + w;

                            Vec3b pix = mat.At<Vec3b>(h, w);
                            pix = new Vec3b((byte)HM_value[idx], (byte)HM_value[idx], (byte)HM_value[idx]);
                            mat.Set<Vec3b>(h, w, pix);
                        }
                    }
                }
            }
            return mat;
        }
        public static Mat GetResultMask(float[] output_value, int[] output_dim, float output2_value)
        {
            Mat mat = new Mat(new Size(output_dim[2], output_dim[3]), MatType.CV_8UC3);
            for (int batch = 0; batch < output_dim[0]; batch++)
            {
                for (int cls = 0; cls < output_dim[1]; cls++)
                {
                    for (int h = 0; h < output_dim[2]; h++)
                    {
                        for (int w = 0; w < output_dim[3]; w++)
                        {
                            int idx = (batch * output_dim[1] * output_dim[2] * output_dim[3]) + (cls * output_dim[2] * output_dim[3]) + (h * output_dim[3]) + w;

                            Vec3b pix = mat.At<Vec3b>(h, w);
                            if (output_value[idx] < output2_value)
                            {
                                pix = new Vec3b(0, 0, 0);
                            }
                            else
                            {
                                pix = new Vec3b(255, 255, 255);
                            }
                            mat.Set<Vec3b>(h, w, pix);
                        }
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
