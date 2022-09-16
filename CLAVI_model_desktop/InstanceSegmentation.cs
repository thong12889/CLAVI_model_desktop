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
    class InstanceSegmentation : IDisposable
    {
        private InferenceSession sess = null;

        public InstanceSegmentation(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }
        public Mat insegInference(Mat image, string labelPath, float threshold)
        {
            float nmsThresh = 0.4f;
            int inputW = 1333;
            int inputH = 800;
            Size imgSize = new Size(inputW, inputH);

            //Label file
            var label = File.ReadLines(labelPath);
            var labelList = label.ToArray();
            var pallete = GenPalette(labelList.Length);

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
                //Postprocessing
                var resultsArray = results.ToArray();
                //Pred
                var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                //Label
                var labels_value = resultsArray[1].AsEnumerable<Int64>().ToArray();
                //Mask
                List<Mat> masks = new List<Mat>();
                var masks_value = resultsArray[2].AsEnumerable<float>().ToArray();
                var masks_dim = resultsArray[2].AsTensor<float>().Dimensions.ToArray();

                var (candidate, label_out, mask_out) = GetInstanceCandidate(pred_value, pred_dim, labels_value, masks_value, masks_dim, threshold);
            }

            return image;
        }
        public static (List<List<float>>, List<int>, List<Mat>) GetInstanceCandidate(float[] pred, int[] pred_dim, long[] labels, float[] masks, int[] masks_dim, float pred_thresh = 0.25f)
        {
            List<List<float>> candidate = new List<List<float>>();
            List<int> labelCand = new List<int>();
            List<Mat> masksCand = new List<Mat>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;//Default 4  // object ness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = pred[idx2];
                    if (value > pred_thresh)
                    {
                        //pred
                        List<float> pred_Cand_value = new List<float>();
                        for (int i = 0; i < pred_dim[2]; i++)
                        {
                            int sub_idx = idx1 + i;
                            pred_Cand_value.Add(pred[sub_idx]);
                        }
                        candidate.Add(pred_Cand_value);

                        int idlabelmask = idx1 / 5;
                        //labels
                        labelCand.Add((int)labels[idlabelmask]);

                        //masks
                        int cls_id = (int)labels[idlabelmask];
                        var maskMats = ConvertInstanceSegmentationResult(idlabelmask, cls_id, masks, masks_dim, pred_thresh);
                        masksCand.Add(maskMats);
                    }
                }
            }
            return (candidate, labelCand, masksCand);
        }
        public static Mat ConvertInstanceSegmentationResult(int id, int cls_id, float[] pred, int[] pred_dim, float threshold = 0.25f)
        {
            
            Mat mat = new Mat(new Size(pred_dim[3], pred_dim[2]), MatType.CV_8UC3);
            for (int batch = 0; batch < pred_dim[1]; batch++)
            {
                for (int h = 0; h < pred_dim[2]; h++)
                {
                    for (int w = 0; w < pred_dim[3]; w++)
                    {
                        int idx = (batch * pred_dim[1] * pred_dim[2] * pred_dim[3]) + (id * pred_dim[2] * pred_dim[3]) + (h * pred_dim[3]) + w;

                        Vec3b pix = mat.At<Vec3b>(h, w);
                        if (pred[idx] < threshold)
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
            return mat;
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
